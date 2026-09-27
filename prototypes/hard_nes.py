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
    'fermat-exp' : the same with q = (R/L)^2 exp(z): a log-deficit, whose relative sensitivity does not vanish
               where the deficit is small (softplus behaves like exp there, but saturates to linear above)
    'fermat-sq' : D = u^2 / (1 + u^2) with u = f(d) - f(0). Near the source the time saved by ray bending is
               T_line - T = R^3 |grad_perp s|^2 / (24 s0) + O(R^4), so D is a smooth positive semidefinite quadratic
               form in d (rank 1 in 2-D) that vanishes along grad s: u is smooth and vanishes at the source, while
               log q in the two forms above must go to -inf along grad s.

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
    'lms'  : the same with Marquardt scaling (damping lam * diag(J^T J) instead of lam * I).
    'lbfgsb', 'bfgs' : SciPy L-BFGS-B / BFGS on the L2 loss mean r^2, full batch.
    'ssbfgs' : self-scaled BFGS (Oren-Luenberger scaling, dense inverse Hessian, strong Wolfe line search) on the
             L2 loss, as in the PINN studies that reach near machine precision (Urban et al., 2024).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

_GL_T, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_T, _GL_W = 0.5 * (_GL_T + 1), 0.5 * _GL_W


def straight_ray(vel, xs, x):
    """
        Straight-ray traveltime T_line = |d| int_0^1 s(xs + t d) dt (s = 1/v, d = x - xs) at x (N, dim), with its
        gradient (N, dim) and hessian (N, dim, dim) w.r.t. x:
            grad T_line = d^ I0 + |d| I1,   hess T_line = (1 - d^ d^T) I0 / |d| + d^ I1^T + I1 d^T + |d| I2,
            I0 = int s, I1 = int t grad s, I2 = int t^2 hess s,   hess s = -hess v / v^2 + 2 grad v grad v^T / v^3.
        hess v is a central difference of `vel.gradient`.
    """
    d = x - xs
    dim = d.shape[-1]
    R = np.linalg.norm(d, axis=-1)
    P = xs + _GL_T[:, None, None] * d[None]                      # (q, N, dim)
    v = vel(P)
    gv = vel.gradient(P)
    h = 1e-5 * np.max(np.abs(np.concatenate([vel.xmin, vel.xmax])))
    Hv = np.stack([(vel.gradient(P + h * e) - vel.gradient(P - h * e)) / (2 * h) for e in np.eye(dim)], axis=-1)
    Hv = 0.5 * (Hv + np.swapaxes(Hv, -1, -2))
    s = 1.0 / v
    gs = -gv / (v * v)[..., None]
    Hs = -Hv / (v ** 2)[..., None, None] + 2 * gv[..., :, None] * gv[..., None, :] / (v ** 3)[..., None, None]
    I0 = np.tensordot(_GL_W, s, axes=1)
    I1 = np.tensordot(_GL_W * _GL_T, gs, axes=1)
    I2 = np.tensordot(_GL_W * _GL_T ** 2, Hs, axes=1)
    pos = R > 0
    safe = np.where(pos, R, 1.0)
    dh = d / safe[:, None]
    TL = R * I0
    gTL = np.where(pos[:, None], dh * I0[:, None] + R[:, None] * I1, 0.0)
    eye = np.eye(dim)[None]
    HTL = ((eye - dh[:, :, None] * dh[:, None, :]) * (I0 / safe)[:, None, None]
           + dh[:, :, None] * I1[:, None, :] + I1[:, :, None] * dh[:, None, :] + R[:, None, None] * I2)
    HTL = np.where(pos[:, None, None], HTL, 0.0)
    return TL, gTL, HTL


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
        TL, gTL, HTL = straight_ray(self.vel, self.xs, x)
        return dict(x=jnp.asarray(x), v=jnp.asarray(self.vel(x)), TL=jnp.asarray(TL), gTL=jnp.asarray(gTL),
                    HTL=jnp.asarray(HTL))


class Config:
    def __init__(self, factor='nes', heads=1, opt='adam', init='he', visc=False, nl=3, nu=16, act='gauss'):
        self.factor, self.heads, self.opt, self.init, self.visc = factor, heads, opt, init, visc
        self.nl, self.nu, self.act = nl, nu, act

    @property
    def id(self):
        return (f"{self.factor}-K{self.heads}-{self.opt}-{self.init}{'-visc' if self.visc else ''}-{self.nl}x{self.nu}"
                + ('' if self.act == 'gauss' else f'-{self.act}'))

    def as_dict(self):
        return dict(factor=self.factor, heads=self.heads, opt=self.opt, init=self.init, visc=self.visc,
                    nl=self.nl, nu=self.nu, act=self.act, id=self.id)


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
    if cfg.factor.startswith('fermat'):                          # start close to the straight-ray solution (D ~ 0)
        head = dict(W=0.1 * head['W'], b=jnp.full(cfg.heads, -5.0))
    return dict(h=[first(ks[0], cfg.nu)] + [dense(ks[i], cfg.nu, cfg.nu) for i in range(1, cfg.nl)],
                u=first(ks[cfg.nl], cfg.nu), v=first(ks[cfg.nl + 1], cfg.nu), head=head,
                a=jnp.ones(cfg.nl), au=jnp.ones(()), av=jnp.ones(()), aout=jnp.ones(()),
                **(dict(c1=jnp.zeros(cfg.nl + 2), c2=jnp.ones(cfg.nl + 2)) if cfg.act == 'cauchy' else {}))


def count_params(p):
    return int(ravel_pytree(p)[0].size)


def _gauss(z, a):
    return jnp.exp(-(a * z) ** 2)


def _act(p, z, a, i, act):
    """ Activation of group i (layers 0..nl-1, then U, V): gauss exp(-(az)^2), lorentz 1/(1+(az)^2) (a pole at
        z = +-i/a), cauchy (c1 az + c2)/(1+(az)^2) (Li, Xia & Zhang 2024; starts equal to lorentz) """
    if act == 'gauss':
        return _gauss(z, a)
    t = a * z
    r = 1.0 / (1.0 + t * t)
    return r if act == 'lorentz' else (p['c1'][i] * t + p['c2'][i]) * r


def trunk(p, f, act='gauss'):
    """ NES `improved_mlp` network: features (dim,) -> logits (heads,) """
    nl = len(p['h'])
    h = _act(p, f @ p['h'][0]['W'] + p['h'][0]['b'], p['a'][0], 0, act)
    U = _act(p, f @ p['u']['W'] + p['u']['b'], p['au'], nl, act)
    V = _act(p, f @ p['v']['W'] + p['v']['b'], p['av'], nl + 1, act)
    for i in range(1, nl):
        h = _act(p, h @ p['h'][i]['W'] + p['h'][i]['b'], p['a'][i], i, act)
        h = (1.0 - h) * U + h * V
    return h @ p['head']['W'] + p['head']['b']


def make_model(prob, cfg):
    """ traveltime(p, x, TL, gTL, HTL, tau) and residual(p, x, v, TL, gTL, HTL, tau, eps) for one point """
    xs = jnp.asarray(prob.xs)
    smin, smax, scale = 1.0 / prob.vmax, 1.0 / prob.vmin, prob.scale

    def traveltime(p, x, TL, gTL, HTL, tau):
        d = x - xs
        R = jnp.sqrt(jnp.sum(d * d) + 1e-300)
        z = trunk(p, d * scale, cfg.act)
        if cfg.factor == 'nes':
            Tk = R * (smin + (smax - smin) * jax.nn.sigmoid(p['aout'] * z))
        else:
            if cfg.factor == 'fermat-sq':
                u = z - trunk(p, jnp.zeros_like(d), cfg.act)
                q = u * u
            else:
                q = (R * scale) ** 2 * (jnp.exp(z) if cfg.factor == 'fermat-exp' else jax.nn.softplus(z))
            D = q / (1.0 + q)
            dx = x - jax.lax.stop_gradient(x)                    # zero, but carries the derivatives of T_line:
            tl = TL + jnp.dot(gTL, dx) + 0.5 * dx @ HTL @ dx    # value, gradient and hessian
            Tk = tl - (tl - R * smin) * D
        if cfg.heads == 1:
            return Tk[0]
        t = jnp.where(tau > 0, tau, 1.0)                        # finite in both branches of the where
        return jnp.where(tau > 0, -t * jax.nn.logsumexp(-Tk / t), jnp.min(Tk))

    def residual(p, x, v, TL, gTL, HTL, tau, eps):
        g = jax.grad(traveltime, argnums=1)(p, x, TL, gTL, HTL, tau)
        r = v * v * jnp.sum(g * g) - 1.0
        if cfg.visc:
            H = jax.jacfwd(jax.grad(traveltime, argnums=1), argnums=1)(p, x, TL, gTL, HTL, tau)
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
        self._T = jax.jit(lambda p: T(p, e['x'], e['TL'], e['gTL'], e['HTL'], 0.0))
        self._G = jax.jit(lambda p: G(p, e['x'], e['TL'], e['gTL'], e['HTL'], 0.0))
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
        r = res_b(p, data['x'][idx], data['v'][idx], data['TL'][idx], data['gTL'][idx], data['HTL'][idx], tau, eps)
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


def train_lm(prob, cfg, seed=0, iters=200, lam0=1e-3, log_every=2, callback=None, max_seconds=600, w_eps=1e-3,
             p0=None, curve0=None, sampler=None, resample_every=40, eig_max=1500, freeze_after=None):
    """
        Levenberg-Marquardt for the L1 loss mean |r| by iteratively reweighted least squares: each iteration
        minimizes |W (r + J d)|^2 + lam |d|^2 with W = diag(1 / sqrt(max(|r|, w_eps * mean|r|))), so |W r|^2 = sum |r|.
        The normal matrix (J^T W^2 J, P x P) is eigendecomposed once per iteration, so a new lam costs a matvec;
        above `eig_max` parameters it is Cholesky-factorized for each lam instead (about 25x cheaper per factorization).
        A step is kept when it lowers sum |r|; lam falls by 3 on success and grows 2x, 4x, ... on failure.
        `sampler(params)` (see Resampler) replaces the collocation set every `resample_every` iterations; a gradual
        sampler (kind '*-g') instead swaps a few points every iteration. No resampling after `freeze_after` seconds.
    """
    scaled = cfg.opt == 'lms'
    p = init_params(jax.random.PRNGKey(seed), cfg, prob) if p0 is None else p0
    flat, unravel = ravel_pytree(p)
    eig = flat.size <= eig_max
    _, res = make_model(prob, cfg)
    data = prob.train
    N = data['x'].shape[0]
    ev = Evaluator(prob, cfg)
    args = (data['x'], data['v'], data['TL'], data['gTL'], data['HTL'])

    def point(fl, x, v, TL, gTL, HTL, tau, eps):
        return res(unravel(fl), x, v, TL, gTL, HTL, tau, eps)

    rvec = jax.jit(lambda fl, args, tau, eps: _vm(point, 2)(fl, *args, tau, eps))

    @jax.jit
    def linearize(fl, args, tau, eps):
        r = _vm(point, 2)(fl, *args, tau, eps)
        J = _vm(jax.grad(point), 2)(fl, *args, tau, eps)
        w = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(r), w_eps * jnp.mean(jnp.abs(r))))
        Jw = w[:, None] * J
        col = jnp.sqrt(jnp.sum(Jw * Jw, axis=0)) if scaled else jnp.ones(Jw.shape[1])
        col = jnp.maximum(col, 1e-12 * jnp.max(col))
        Js = Jw / col                                           # Marquardt scaling: unit columns
        A, g = Js.T @ Js, Js.T @ (w * r)
        if eig:
            S2, V = jnp.linalg.eigh(A)
            return V.T @ g, jnp.maximum(S2, 0.0), V, jnp.sum(jnp.abs(r)), col
        return g, jnp.trace(A)[None], A, jnp.sum(jnp.abs(r)), col   # trace bounds the largest eigenvalue

    @jax.jit
    def step_of(gv, S2, V, lam, col):
        if eig:
            return -(V @ (gv / (S2 + lam))) / col
        c = jax.scipy.linalg.cho_factor(V + lam * jnp.eye(V.shape[0]))   # V is the normal matrix here
        return -jax.scipy.linalg.cho_solve(c, gv) / col

    tc = time.perf_counter()
    gv, S2, V, f, col = linearize(flat, args, 1.0, 0.0)
    jax.block_until_ready(step_of(gv, S2, V, 1.0, col))
    jax.block_until_ready(rvec(flat, args, 1.0, 0.0))
    compile_s = time.perf_counter() - tc

    lam = None
    curve = list(curve0) if curve0 else [(0, 0.0, float('nan'), ev.rmae(p))]
    t_train, it0 = curve[-1][1], curve[-1][0]
    for it in range(1, iters + 1):
        tau, eps = schedule(prob, cfg, it / iters, stop_tau=0.75, stop_eps=0.7)
        ts = time.perf_counter()
        live = sampler is not None and it > 1 and (freeze_after is None or t_train < freeze_after)
        if live and sampler.gradual:
            args = sampler.evolve(unravel(flat), args, f / N)
        elif live and (it - 1) % resample_every == 0:
            d = sampler(unravel(flat))
            args = (d['x'], d['v'], d['TL'], d['gTL'], d['HTL'])
        gv, S2, V, f, col = linearize(flat, args, tau, eps)
        if lam is None:
            lam = lam0 * float(S2[-1])
        f, accepted, nu = float(f), False, 2.0
        for _ in range(16):
            new = flat + step_of(gv, S2, V, lam, col)
            f_new = float(jnp.sum(jnp.abs(rvec(new, args, tau, eps))))
            if np.isfinite(f_new) and f_new < f:
                flat, lam, accepted = new, lam / 3.0, True
                break
            lam, nu = lam * nu, nu * 2
        jax.block_until_ready(flat)
        t_train += time.perf_counter() - ts
        if it % log_every == 0 or it == iters or not accepted:
            curve.append((it0 + it, t_train, (f_new if accepted else f) / N, ev.rmae(unravel(flat))))
            if callback:
                callback(curve)
        if not accepted or t_train > max_seconds:
            break
    return unravel(flat), curve, compile_s


def ssbfgs(fg, x0, callback, max_iters=100000, c2=0.9):
    """
        Self-scaled BFGS on the inverse Hessian H: H+ = tau (H - Hy yH / yHy + yHy v v^T) + s s^T / s^T y,
        v = s / s^T y - Hy / yHy, tau = min(1, s^T y / yHy) (Oren-Luenberger scaling, capped as in Al-Baali),
        with SciPy's strong Wolfe line search. `callback(x, f)` may raise StopIteration.
    """
    from scipy.optimize import line_search
    cache = {}

    def f_of(z):
        fz, gz = fg(z)
        cache.clear()
        cache[z.tobytes()] = gz
        return fz

    def g_of(z):
        key = z.tobytes()
        return cache[key] if key in cache else fg(z)[1]

    x = np.array(x0, dtype=np.float64)
    f, g = fg(x)
    n, H, scaled_once, fails = x.size, np.eye(x.size), False, 0
    for _ in range(max_iters):
        d = -H @ g
        if g @ d >= 0:
            H, d = np.eye(n), -g
        alpha, _, _, f_new, _, _ = line_search(f_of, g_of, x, d, gfk=g, old_fval=f, c1=1e-4, c2=c2, maxiter=40)
        if alpha is None:
            fails += 1
            if fails > 2:
                break
            H = np.eye(n)
            continue
        fails = 0
        s = alpha * d
        x_new = x + s
        g_new = g_of(x_new)
        y = g_new - g
        sy = s @ y
        if not scaled_once and sy > 0:
            H, scaled_once = (sy / (y @ y)) * np.eye(n), True
        if sy > 1e-14 * np.linalg.norm(s) * np.linalg.norm(y):
            Hy = H @ y
            yHy = y @ Hy
            tau = min(1.0, sy / yHy)
            v = s / sy - Hy / yHy
            H = tau * (H - np.outer(Hy, Hy) / yHy + yHy * np.outer(v, v)) + np.outer(s, s) / sy
        x, f, g = x_new, f_new, g_new
        try:
            callback(x, f)
        except StopIteration:
            break
    return x


def train_qn(prob, cfg, seed=0, max_seconds=90, log_every=25, warmup=0, max_iters=200000, callback=None):
    """
        Quasi-Newton training on the L2 loss mean r^2 (full batch, float64): 'lbfgsb' and 'bfgs' from SciPy,
        'ssbfgs' above. `warmup` > 0 runs that many Adam epochs first (the curve then starts with them).
        Stops after `max_seconds` of optimizer time (logging excluded) or when the method stops.
    """
    from scipy.optimize import minimize
    if warmup:
        p, curve, compile_s = train_adam(prob, cfg, seed=seed, epochs=warmup, log_every=log_every)
        curve, t0 = list(curve), curve[-1][1]
    else:
        p, curve, compile_s, t0 = init_params(jax.random.PRNGKey(seed), cfg, prob), None, 0.0, 0.0
    flat0, unravel = ravel_pytree(p)
    _, res = make_model(prob, cfg)
    data = prob.train
    args = (data['x'], data['v'], data['TL'], data['gTL'], data['HTL'])
    ev = Evaluator(prob, cfg)

    def point(fl, x, v, TL, gTL, HTL, tau, eps):
        return res(unravel(fl), x, v, TL, gTL, HTL, tau, eps)

    def loss(fl):
        r = _vm(point, 2)(fl, *args, 0.0, 0.0)
        return jnp.mean(r * r)

    vg = jax.jit(jax.value_and_grad(loss))
    tc = time.perf_counter()
    jax.block_until_ready(vg(flat0))
    compile_s += time.perf_counter() - tc
    if curve is None:
        curve = [(0, 0.0, float('nan'), ev.rmae(p))]

    def fg(x):
        f, g = vg(jnp.asarray(x))
        return float(f), np.asarray(g, dtype=np.float64)

    st = dict(it=0, t=0.0, mark=time.perf_counter(), f=float('nan'))

    def on_iter(x, f=None):
        now = time.perf_counter()
        st['t'] += now - st['mark']
        st['it'] += 1
        if f is not None:
            st['f'] = float(f)
        if st['it'] % log_every == 0:
            curve.append((st['it'], t0 + st['t'], st['f'], ev.rmae(unravel(jnp.asarray(x)))))
            if callback:
                callback(curve)
        st['mark'] = time.perf_counter()
        if st['t'] > max_seconds:
            raise StopIteration

    if cfg.opt == 'ssbfgs':
        x = ssbfgs(fg, np.asarray(flat0), on_iter, max_iters=max_iters)
    else:
        method = {'lbfgsb': 'L-BFGS-B', 'bfgs': 'BFGS'}[cfg.opt]
        opts = dict(maxiter=max_iters, gtol=0.0)
        if method == 'L-BFGS-B':
            opts.update(maxcor=50, ftol=0.0, maxfun=10 * max_iters, maxls=50)
        out = minimize(fg, np.asarray(flat0), jac=True, method=method, options=opts,
                       callback=lambda intermediate_result: on_iter(intermediate_result.x, intermediate_result.fun))
        x = out.x
    st['t'] += time.perf_counter() - st['mark']
    pf = unravel(jnp.asarray(x))
    curve.append((st['it'], t0 + st['t'], float(fg(x)[0]), ev.rmae(pf)))
    return pf, curve, compile_s



class Resampler:
    """
        New collocation sets for `train_lm(sampler=...)`, drawn from a fixed pool of uniform candidates:
            'uniform'  : a fresh uniform set
            'rad'      : density ~ |r| / mean|r| + c (residual-based adaptive distribution, Wu et al. 2023)
            'rad-curv' : the same, but points where the smallest hessian eigenvalue times R is below -kappa (caustic
                         smoothing zones: first arrivals are semiconcave, their smooth fits bend down sharply there)
                         keep only the uniform part c
            '<kind>-g' : gradual version of uniform, rad or rad-curv: every iteration k slots get a uniform candidate,
                         accepted with probability min(1, w(new) / w(old)) (Metropolis-Hastings with independent
                         proposals), so the set drifts toward the density w without jumps
            'dwr'      : density ~ |r| A / mean(|r| A) + c, A = number of the network's descent paths (backward rays)
                         through the point: a residual matters for every receiver downstream of it along the rays,
                         and caustic crests have none (dual-weighted residual, the adjoint of the linearized eikonal
                         is transport along rays)
    """
    def __init__(self, prob, cfg, kind, seed=0, pool=40000, n=None, c=1.0, kappa=5.0, n_acc=81, n_paths=8000,
                 k=None):
        self.gradual = kind.endswith('-g')
        kind = kind[:-2] if self.gradual else kind
        assert not (self.gradual and kind == 'dwr'), 'no gradual dwr'
        self.prob, self.kind, self.c, self.kappa, self.n_acc, self.n_paths = prob, kind, c, kappa, n_acc, n_paths
        self.rng = np.random.default_rng(seed + 12345)
        xp = self.rng.uniform(prob.xmin, prob.xmax, size=(pool, 2))
        xp = xp[np.abs(xp - prob.xs).sum(-1) > 1e-5]
        self.pool = prob.features(xp)
        self.xp = xp
        self.n = n or int(prob.train['x'].shape[0])
        self.k = k or int(np.ceil(0.01 * self.n))
        tt, res = make_model(prob, cfg)
        pl = self.pool
        self._r_at = jax.jit(jax.vmap(res, in_axes=(None, 0, 0, 0, 0, 0, None, None)))
        self._h_at = jax.jit(jax.vmap(jax.hessian(tt, argnums=1), in_axes=(None, 0, 0, 0, 0, None)))
        self._host = None
        self._r = jax.jit(lambda p: jax.vmap(res, in_axes=(None, 0, 0, 0, 0, 0, None, None))(
            p, pl['x'], pl['v'], pl['TL'], pl['gTL'], pl['HTL'], 0.0, 0.0))
        if kind == 'rad-curv':
            self._h = jax.jit(lambda p: jax.vmap(jax.hessian(tt, argnums=1), in_axes=(None, 0, 0, 0, 0, None))(
                p, pl['x'], pl['TL'], pl['gTL'], pl['HTL'], 0.0))
        if kind == 'dwr':
            e = prob.eval
            self._g = jax.jit(lambda p: jax.vmap(jax.grad(tt, argnums=1), in_axes=(None, 0, 0, 0, 0, None))(
                p, e['x'], e['TL'], e['gTL'], e['HTL'], 0.0))

    def flow_accumulation(self, params):
        """ Visits of descent paths of the network, started from n_paths pool points, per cell of an n_acc^2 grid,
            scaled to paths from every pool point """
        from scipy.ndimage import map_coordinates
        prob = self.prob
        n = prob.X.shape[0]
        G = np.asarray(self._g(params)).reshape(n, n, 2)
        lo, hi = prob.xmin, prob.xmax
        cell = (hi - lo) / (self.n_acc - 1)
        ds = 0.5 * cell.min()
        x = self.xp[:self.n_paths].copy()                      # the pool is already uniform random
        alive = np.ones(len(x), bool)
        acc = np.zeros((self.n_acc, self.n_acc))
        for _ in range(int(2 * np.linalg.norm(hi - lo) / ds)):
            if not alive.any():
                break
            fi = ((x[alive] - lo) / (hi - lo) * (n - 1)).T
            g = np.stack([map_coordinates(G[..., k], fi, order=1, mode='nearest') for k in range(2)], -1)
            x[alive] -= ds * g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
            x = np.clip(x, lo, hi)
            idx = np.clip(np.round((x[alive] - lo) / cell).astype(int), 0, self.n_acc - 1)
            np.add.at(acc, (idx[:, 0], idx[:, 1]), 1.0)
            alive[alive] = np.linalg.norm(x[alive] - prob.xs, axis=-1) > 2 * ds
        return acc * len(self.xp) / len(x)

    def _weight(self, params, pts, mean_r):
        """ Target density (unnormalized) at the points pts = (x, v, TL, gTL, HTL) """
        r = np.abs(np.asarray(self._r_at(params, *pts, 0.0, 0.0)))
        w = r / max(mean_r, 1e-300) + self.c
        if self.kind == 'rad-curv':
            x, v, TL, gTL, HTL = pts
            lam_min = np.linalg.eigvalsh(np.asarray(self._h_at(params, x, TL, gTL, HTL, 0.0)))[:, 0]
            R = np.linalg.norm(np.asarray(x) - self.prob.xs, axis=-1)
            w = np.where(lam_min * R < -self.kappa, self.c, w)
        return w

    def evolve(self, params, args, mean_r):
        """ One gradual step: k slots of the collocation set `args` get Metropolis-Hastings moves """
        if self._host is None or self._host[0].shape[0] != args[0].shape[0]:
            self._host = [np.array(a) for a in args]
        h, k = self._host, self.k
        slots = self.rng.choice(h[0].shape[0], size=k, replace=False)
        cand = self.rng.integers(len(self.xp), size=k)
        keys = ('x', 'v', 'TL', 'gTL', 'HTL')
        new = [np.asarray(self.pool[key])[cand] for key in keys]
        if self.kind == 'uniform':
            acc = np.ones(k, bool)
        else:
            w = self._weight(params, tuple(jnp.asarray(np.concatenate([a[slots], b])) for a, b in zip(h, new)), mean_r)
            acc = self.rng.uniform(size=k) * w[:k] < w[k:]
        for a, b in zip(h, new):
            a[slots[acc]] = b[acc]
        self.accepted = getattr(self, 'accepted', 0) + int(acc.sum())
        return tuple(jnp.asarray(a) for a in h)

    def __call__(self, params):
        if self.kind == 'uniform':
            w = np.ones(len(self.xp))
        else:
            r = np.abs(np.asarray(self._r(params)))
            if self.kind == 'dwr':
                acc = self.flow_accumulation(params)
                lo, hi = self.prob.xmin, self.prob.xmax
                idx = np.clip(np.round((self.xp - lo) / ((hi - lo) / (self.n_acc - 1))).astype(int), 0, self.n_acc - 1)
                r = r * (1.0 + acc[idx[:, 0], idx[:, 1]])
            w = r / max(r.mean(), 1e-300) + self.c
            if self.kind == 'rad-curv':
                Hs = np.asarray(self._h(params))
                lam_min = np.linalg.eigvalsh(Hs)[:, 0]
                R = np.linalg.norm(self.xp - self.prob.xs, axis=-1)
                w = np.where(lam_min * R < -self.kappa, self.c, w)
        idx = self.rng.choice(len(self.xp), size=self.n, replace=False, p=w / w.sum())
        return {k: v[idx] for k, v in self.pool.items()}


def train(prob, cfg, seed=0, warmup=0, **kw):
    """ Trains with cfg.opt; `warmup` > 0 runs that many Adam epochs first (for 'lm', 'lms' and quasi-Newton) """
    if cfg.opt == 'adam':
        return train_adam(prob, cfg, seed=seed, **kw)
    if cfg.opt in ('lm', 'lms'):
        if warmup:
            p, curve, cs = train_adam(prob, cfg, seed=seed, epochs=warmup)
            p, curve, cs2 = train_lm(prob, cfg, seed=seed, p0=p, curve0=curve, **kw)
            return p, curve, cs + cs2
        return train_lm(prob, cfg, seed=seed, **kw)
    return train_qn(prob, cfg, seed=seed, warmup=warmup, **kw)
