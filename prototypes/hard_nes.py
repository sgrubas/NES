"""
Hard-constraint NES-OP prototypes, in pure JAX (float64).

The network is NES-OP's (U/V-gated MLP of `improved_mlp`, adaptive Gaussian activations, input = scaled xr - xs);
what changes is what is built into the output, how it starts, and how it is trained.

Factorization (hard constraints on T):
    'nes'    : T = R * (1/vmax + (1/vmin - 1/vmax) * sigmoid(a z))              (NES improved factorization)
    'fermat' : T = T_line - (T_line - R/vmax) * D,   D = q / (1 + q),   q = (R/L)^2 softplus(z)
               T_line = straight-ray traveltime. Fermat's principle gives R/vmax <= T <= T_line exactly, and
               D in [0, 1) keeps T inside. D = O(R^2) near the source, so T = T_line + O(R^3), which is the exact
               local behaviour (ray bending changes traveltime at third order). z -> -inf is the straight ray.

Output shape (hard constraint on the singularities of T):
    heads = 1 : one smooth branch.
    heads > 1 : T = min_k T_k. First-arrival traveltime is semiconcave: every kink (caustic) is a minimum of smooth
                branches and none is convex. Trained through -tau log sum exp(-T_k / tau) with tau -> 0, which is the
                Cole-Hopf / Maslov image of linear superposition of branches.

Initialization:
    'he'    : He-normal kernels, zero biases (NES): every first-layer ridge passes through the source.
    'ridge' : first-layer ridges spread over the domain (directions, offsets, widths).

Vanishing viscosity (visc=True): residual of v^2 |grad T|^2 - 1 = eps v^2 lap T, eps -> 0. With T = -eps log phi this
    is the linear equation eps^2 lap phi = phi / v^2, whose -eps log converges to the first arrival (Varadhan).

Optimizers:
    'adam' : NES defaults: L1 of the Hamiltonian residual (v^2 |grad T|^2 - 1)/2, minibatches,
             inverse-time learning-rate decay.
    'lm'   : Levenberg-Marquardt for the same L1 loss (iteratively reweighted least squares), full batch,
             full Jacobian; damped steps from the eigendecomposition of the (small) normal matrix.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

_GL_T, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_T, _GL_W = 0.5 * (_GL_T + 1), 0.5 * _GL_W


def straight_ray(vel, xs, x):
    """
        Straight-ray traveltime T_line = |d| int_0^1 s(xs + t d) dt (s = 1/v, d = x - xs) at x (N, dim), with its
        gradient (N, dim) and laplacian (N,) w.r.t. x:
            grad T_line = d/|d| I0 + |d| I1,   lap T_line = (dim - 1) I0 / |d| + 2 d/|d| . I1 + |d| I2,
            I0 = int s, I1 = int t grad s, I2 = int t^2 lap s.
    """
    d = x - xs
    dim = d.shape[-1]
    R = np.linalg.norm(d, axis=-1)
    P = xs + _GL_T[:, None, None] * d[None]                      # (q, N, dim)
    v = vel(P)
    gv = vel.gradient(P)
    h = 1e-5 * np.max(np.abs(np.concatenate([vel.xmin, vel.xmax])))
    lap_v = sum((vel.gradient(P + h * e)[..., i] - vel.gradient(P - h * e)[..., i]) / (2 * h)
                for i, e in enumerate(np.eye(dim)))
    s = 1.0 / v
    gs = -gv / (v * v)[..., None]
    lap_s = -lap_v / v ** 2 + 2 * np.sum(gv * gv, axis=-1) / v ** 3
    I0 = np.tensordot(_GL_W, s, axes=1)
    I1 = np.tensordot(_GL_W * _GL_T, gs, axes=1)
    I2 = np.tensordot(_GL_W * _GL_T ** 2, lap_s, axes=1)
    pos = R > 0
    safe = np.where(pos, R, 1.0)
    TL = R * I0
    gTL = np.where(pos[:, None], d / safe[:, None] * I0[:, None] + R[:, None] * I1, 0.0)
    LTL = np.where(pos, (dim - 1) * I0 / safe + 2 * np.sum(d / safe[:, None] * I1, axis=-1) + R * I2, 0.0)
    return TL, gTL, LTL


class Problem:
    """ Velocity model, source, collocation points, evaluation grid and reference traveltimes """
    def __init__(self, name, title, vel, xs, T_ref, n_train=4000, seed=0, caustic=False, reference='exact'):
        self.name, self.title, self.vel = name, title, vel
        self.xs = np.asarray(xs, dtype=float)
        self.xmin, self.xmax = np.asarray(vel.xmin, float), np.asarray(vel.xmax, float)
        self.vmin, self.vmax = float(vel.min), float(vel.max)
        self.scale = 1.0 / np.max(np.abs([self.xmin, self.xmax]))
        self.caustic, self.reference = caustic, reference
        n = T_ref.shape[0]
        self.axes = [np.linspace(lo, hi, n) for lo, hi in zip(self.xmin, self.xmax)]
        self.X = np.stack(np.meshgrid(*self.axes, indexing='ij'), axis=-1)
        self.T_ref = T_ref
        self.eval = self.features(self.X.reshape(-1, 2))
        R = np.linalg.norm(self.X.reshape(-1, 2) - self.xs, axis=-1)
        self.eval_ok = (R > 0) & np.isfinite(T_ref.ravel())
        self.eval_far = self.eval_ok & (R > 0.05)
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.xmin, self.xmax, size=(n_train, 2))
        x = x[np.abs(x - self.xs).sum(-1) > 1e-5]
        self.train = self.features(x)
        self.t_scale = float(np.median(np.asarray(self.train['TL'])))

    def features(self, x):
        TL, gTL, LTL = straight_ray(self.vel, self.xs, x)
        return dict(x=jnp.asarray(x), v=jnp.asarray(self.vel(x)), TL=jnp.asarray(TL), gTL=jnp.asarray(gTL),
                    LTL=jnp.asarray(LTL))


class Config:
    def __init__(self, factor='nes', heads=1, opt='adam', init='he', visc=False, nl=3, nu=16):
        self.factor, self.heads, self.opt, self.init, self.visc = factor, heads, opt, init, visc
        self.nl, self.nu = nl, nu

    @property
    def id(self):
        return f"{self.factor}-K{self.heads}-{self.opt}-{self.init}{'-visc' if self.visc else ''}-{self.nl}x{self.nu}"

    def as_dict(self):
        return dict(factor=self.factor, heads=self.heads, opt=self.opt, init=self.init, visc=self.visc,
                    nl=self.nl, nu=self.nu, id=self.id)


###############################################################################
#                                   NETWORK                                   #
###############################################################################


def init_params(key, cfg, prob):
    def dense(k, n_in, n_out):
        return dict(W=jax.random.normal(k, (n_in, n_out)) * np.sqrt(2.0 / n_in), b=jnp.zeros(n_out))

    lo, hi = (prob.xmin - prob.xs) * prob.scale, (prob.xmax - prob.xs) * prob.scale
    corners = jnp.asarray(np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]]))

    def ridge(k, n_out):
        k1, k2, k3 = jax.random.split(k, 3)
        theta = (jnp.arange(n_out) + jax.random.uniform(k1, (n_out,))) * jnp.pi / n_out
        e = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)            # (n_out, 2)
        proj = corners @ e.T                                                # (4, n_out)
        p_lo, p_hi = proj.min(0), proj.max(0)
        c = p_lo + (p_hi - p_lo) * (jax.random.permutation(k2, n_out) + 0.5) / n_out
        w = (2.0 + 4.0 * jax.random.uniform(k3, (n_out,))) / (p_hi - p_lo)
        return dict(W=(e * w[:, None]).T, b=-w * c)

    first = ridge if cfg.init == 'ridge' else (lambda k, n: dense(k, 2, n))
    ks = jax.random.split(key, cfg.nl + 3)
    head = dense(ks[cfg.nl + 2], cfg.nu, cfg.heads)
    if cfg.factor == 'fermat':                                   # start close to the straight-ray solution (D ~ 0)
        head = dict(W=0.1 * head['W'], b=jnp.full(cfg.heads, -5.0))
    return dict(h=[first(ks[0], cfg.nu)] + [dense(ks[i], cfg.nu, cfg.nu) for i in range(1, cfg.nl)],
                u=first(ks[cfg.nl], cfg.nu), v=first(ks[cfg.nl + 1], cfg.nu), head=head,
                a=jnp.ones(cfg.nl), au=jnp.ones(()), av=jnp.ones(()), aout=jnp.ones(()))


def count_params(p):
    return int(ravel_pytree(p)[0].size)


def _gauss(z, a):
    return jnp.exp(-(a * z) ** 2)


def trunk(p, f):
    """ NES `improved_mlp` network: features (dim,) -> logits (heads,) """
    h = _gauss(f @ p['h'][0]['W'] + p['h'][0]['b'], p['a'][0])
    U = _gauss(f @ p['u']['W'] + p['u']['b'], p['au'])
    V = _gauss(f @ p['v']['W'] + p['v']['b'], p['av'])
    for i in range(1, len(p['h'])):
        h = _gauss(h @ p['h'][i]['W'] + p['h'][i]['b'], p['a'][i])
        h = (1.0 - h) * U + h * V
    return h @ p['head']['W'] + p['head']['b']


def make_model(prob, cfg):
    """ traveltime(p, x, TL, gTL, LTL, tau) and residual(p, x, v, TL, gTL, LTL, tau, eps) for one point """
    xs = jnp.asarray(prob.xs)
    smin, smax, scale = 1.0 / prob.vmax, 1.0 / prob.vmin, prob.scale

    def traveltime(p, x, TL, gTL, LTL, tau):
        d = x - xs
        R = jnp.sqrt(jnp.sum(d * d) + 1e-300)
        z = trunk(p, d * scale)
        if cfg.factor == 'nes':
            Tk = R * (smin + (smax - smin) * jax.nn.sigmoid(p['aout'] * z))
        else:
            q = (R * scale) ** 2 * jax.nn.softplus(z)
            D = q / (1.0 + q)
            dx = x - jax.lax.stop_gradient(x)                    # zero, but carries the derivatives of T_line:
            tl = TL + jnp.dot(gTL, dx) + 0.25 * LTL * jnp.dot(dx, dx)   # value, gradient and laplacian (2D)
            Tk = tl - (tl - R * smin) * D
        if cfg.heads == 1:
            return Tk[0]
        t = jnp.where(tau > 0, tau, 1.0)                        # finite in both branches of the where
        return jnp.where(tau > 0, -t * jax.nn.logsumexp(-Tk / t), jnp.min(Tk))

    def residual(p, x, v, TL, gTL, LTL, tau, eps):
        g = jax.grad(traveltime, argnums=1)(p, x, TL, gTL, LTL, tau)
        r = v * v * jnp.sum(g * g) - 1.0
        if cfg.visc:
            H = jax.jacfwd(jax.grad(traveltime, argnums=1), argnums=1)(p, x, TL, gTL, LTL, tau)
            r = r - eps * v * v * jnp.trace(H)
        return 0.5 * r

    return traveltime, residual


def schedule(prob, cfg, progress, stop_tau=0.85, stop_eps=0.8):
    """ Temperature of the soft minimum and viscosity, both -> 0 during training """
    tau = eps = 0.0
    if cfg.heads > 1 and progress <= stop_tau:
        tau = 0.02 * prob.t_scale * 10 ** (-6 * min(1.0, progress / (stop_tau - 0.15)))
    if cfg.visc and progress <= stop_eps:
        eps = 0.02 * prob.t_scale * 10 ** (-6 * min(1.0, progress / (stop_eps - 0.2)))
    return tau, eps


###############################################################################
#                                  EVALUATION                                 #
###############################################################################


class Evaluator:
    def __init__(self, prob, cfg):
        self.prob = prob
        tt, _ = make_model(prob, cfg)
        e = prob.eval
        T = jax.vmap(tt, in_axes=(None, 0, 0, 0, 0, None))
        G = jax.vmap(jax.grad(tt, argnums=1), in_axes=(None, 0, 0, 0, 0, None))
        self._T = jax.jit(lambda p: T(p, e['x'], e['TL'], e['gTL'], e['LTL'], 0.0))
        self._G = jax.jit(lambda p: G(p, e['x'], e['TL'], e['gTL'], e['LTL'], 0.0))
        self.ref = prob.T_ref.ravel()

    def rmae(self, p):
        T = np.asarray(self._T(p))
        ok = self.prob.eval_ok
        return float(np.abs(T - self.ref)[ok].sum() / np.abs(self.ref[ok]).sum())

    def full(self, p):
        T = np.asarray(self._T(p))
        G = np.asarray(self._G(p))
        ok, far = self.prob.eval_ok, self.prob.eval_far
        rel = np.where(ok, (T - self.ref) / np.where(ok, self.ref, 1.0), 0.0)
        H1 = np.asarray(self.prob.eval['v']) * np.linalg.norm(G, axis=-1) - 1.0
        return dict(T=T, rel=rel, H1=H1,
                    rmae=float(np.abs(T - self.ref)[ok].sum() / np.abs(self.ref[ok]).sum()),
                    max_over=float(max(rel[far].max(), 0.0)), max_under=float(max(-rel[far].min(), 0.0)),
                    cert_over=float(max(H1[far].max(), 0.0)), cert_under=float(max(-H1[far].min(), 0.0)))


###############################################################################
#                                  TRAINING                                   #
###############################################################################


def _vm(fn, n_const=0):
    return jax.vmap(fn, in_axes=(None, 0, 0, 0, 0, 0) + (None,) * n_const)


def train_adam(prob, cfg, seed=0, epochs=3000, lr=6e-3, decay=3e-4, n_batches=8, log_every=25, callback=None):
    """ NES-style training. Returns params, the curve [(step, seconds, loss, rmae)] and compile seconds """
    p = init_params(jax.random.PRNGKey(seed), cfg, prob)
    _, res = make_model(prob, cfg)
    res_b = _vm(res, 2)
    data = prob.train
    N = data['x'].shape[0]
    B = int(np.ceil(N / n_batches))
    ev = Evaluator(prob, cfg)

    def loss_fn(p, idx, tau, eps):
        r = res_b(p, data['x'][idx], data['v'][idx], data['TL'][idx], data['gTL'][idx], data['LTL'][idx], tau, eps)
        return jnp.mean(jnp.abs(r))

    grad_fn = jax.value_and_grad(loss_fn)
    b1, b2, adam_eps = 0.9, 0.999, 1e-7

    @jax.jit
    def epoch(p, m, v, step, perm, tau, eps):
        def body(carry, idx):
            p, m, v, step = carry
            loss, g = grad_fn(p, idx, tau, eps)
            step = step + 1
            m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
            v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
            lr_t = lr / (1.0 + decay * (step - 1)) * jnp.sqrt(1 - b2 ** step) / (1 - b1 ** step)
            p = jax.tree_util.tree_map(lambda a, mm, vv: a - lr_t * mm / (jnp.sqrt(vv) + adam_eps), p, m, v)
            return (p, m, v, step), loss
        (p, m, v, step), losses = jax.lax.scan(body, (p, m, v, step), perm)
        return p, m, v, step, jnp.mean(losses)

    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    step = jnp.zeros((), dtype=jnp.int64)
    rng = np.random.default_rng(seed)
    perm_of = lambda: jnp.asarray(np.resize(rng.permutation(N), n_batches * B).reshape(n_batches, B))

    tc = time.perf_counter()
    jax.block_until_ready(epoch(p, m, v, step, perm_of(), 1.0, 0.0))
    compile_s = time.perf_counter() - tc

    curve, t_train = [(0, 0.0, float('nan'), ev.rmae(p))], 0.0
    for e in range(1, epochs + 1):
        tau, eps = schedule(prob, cfg, e / epochs)
        ts = time.perf_counter()
        p, m, v, step, loss = epoch(p, m, v, step, perm_of(), tau, eps)
        loss = float(loss)
        t_train += time.perf_counter() - ts
        if e % log_every == 0 or e == epochs:
            curve.append((int(step), t_train, loss, ev.rmae(p)))
            if callback:
                callback(curve)
    return p, curve, compile_s


def train_lm(prob, cfg, seed=0, iters=200, lam0=1e-3, log_every=2, callback=None, max_seconds=600, w_eps=1e-3):
    """
        Levenberg-Marquardt for the L1 loss mean |r| by iteratively reweighted least squares: each iteration
        minimizes |W (r + J d)|^2 + lam |d|^2 with W = diag(1 / sqrt(max(|r|, w_eps * mean|r|))), so |W r|^2 = sum |r|.
        The normal matrix (J^T W^2 J, P x P) is eigendecomposed once per iteration, so a new lam costs a matvec.
        A step is kept when it lowers sum |r|; lam falls by 3 on success and grows 2x, 4x, ... on failure.
    """
    p = init_params(jax.random.PRNGKey(seed), cfg, prob)
    flat, unravel = ravel_pytree(p)
    _, res = make_model(prob, cfg)
    data = prob.train
    N = data['x'].shape[0]
    ev = Evaluator(prob, cfg)
    args = (data['x'], data['v'], data['TL'], data['gTL'], data['LTL'])

    def point(fl, x, v, TL, gTL, LTL, tau, eps):
        return res(unravel(fl), x, v, TL, gTL, LTL, tau, eps)

    rvec = jax.jit(lambda fl, tau, eps: _vm(point, 2)(fl, *args, tau, eps))

    @jax.jit
    def linearize(fl, tau, eps):
        r = _vm(point, 2)(fl, *args, tau, eps)
        J = _vm(jax.grad(point), 2)(fl, *args, tau, eps)
        w = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(r), w_eps * jnp.mean(jnp.abs(r))))
        Jw = w[:, None] * J
        S2, V = jnp.linalg.eigh(Jw.T @ Jw)
        return V.T @ (Jw.T @ (w * r)), jnp.maximum(S2, 0.0), V, jnp.sum(jnp.abs(r))

    @jax.jit
    def step_of(gv, S2, V, lam):
        return -V @ (gv / (S2 + lam))

    tc = time.perf_counter()
    gv, S2, V, f = linearize(flat, 1.0, 0.0)
    jax.block_until_ready(step_of(gv, S2, V, 1.0))
    jax.block_until_ready(rvec(flat, 1.0, 0.0))
    compile_s = time.perf_counter() - tc

    lam = None
    curve, t_train = [(0, 0.0, float('nan'), ev.rmae(p))], 0.0
    for it in range(1, iters + 1):
        tau, eps = schedule(prob, cfg, it / iters, stop_tau=0.75, stop_eps=0.7)
        ts = time.perf_counter()
        gv, S2, V, f = linearize(flat, tau, eps)
        if lam is None:
            lam = lam0 * float(S2[-1])
        f, accepted, nu = float(f), False, 2.0
        for _ in range(16):
            new = flat + step_of(gv, S2, V, lam)
            f_new = float(jnp.sum(jnp.abs(rvec(new, tau, eps))))
            if np.isfinite(f_new) and f_new < f:
                flat, lam, accepted = new, lam / 3.0, True
                break
            lam, nu = lam * nu, nu * 2
        jax.block_until_ready(flat)
        t_train += time.perf_counter() - ts
        if it % log_every == 0 or it == iters or not accepted:
            curve.append((it, t_train, (f_new if accepted else f) / N, ev.rmae(unravel(flat))))
            if callback:
                callback(curve)
        if not accepted or t_train > max_seconds:
            break
    return unravel(flat), curve, compile_s


def train(prob, cfg, seed=0, **kw):
    return (train_adam if cfg.opt == 'adam' else train_lm)(prob, cfg, seed=seed, **kw)
