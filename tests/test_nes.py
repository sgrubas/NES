"""
NES test suite. Runs on the backend selected by KERAS_BACKEND (jax, tensorflow, torch):

    KERAS_BACKEND=jax pytest tests
"""
import pathlib
import pickle
import numpy as np
import pytest
import keras

import NES
from NES.velocity import VerticalGradient, LocAnomaly, Interpolator, MaxwellFishEye, LuneburgLens
from NES.eikonalLayers import IsoEikonal
from NES.layers import RECIPROCITY_MODES, ACTS

DATA = pathlib.Path(__file__).parent / 'data'
RNG = np.random.default_rng(0)


@pytest.fixture(scope='module')
def vel():
    return VerticalGradient(v0=2.0, a=1.0, xmin=[0., 0.], xmax=[1., 1.])


def central_diff(f, x, eps=1e-3):
    """ Jacobian of f: (N, D) -> (N, ...) by central differences, stacked on the last axis """
    cols = []
    for i in range(x.shape[-1]):
        e = np.zeros(x.shape[-1])
        e[i] = eps
        cols.append((f(x + e) - f(x - e)) / (2 * eps))
    return np.stack(cols, axis=-1)


def assert_close(a, b, rtol):
    scale = np.abs(b).max() + 1e-12
    assert np.abs(np.asarray(a) - np.asarray(b)).max() / scale < rtol


########################################################################
#                      derivatives & reciprocity
########################################################################


@pytest.mark.parametrize('reciprocity', RECIPROCITY_MODES)
@pytest.mark.parametrize('improved_mlp', [False, True])
def test_tp_derivatives(vel, reciprocity, improved_mlp):
    keras.utils.set_random_seed(1)
    nes = NES.NES_TP(vel)
    nes.build_model(nl=3, nu=16, reciprocity=reciprocity, improved_mlp=improved_mlp)
    x = RNG.uniform(0.1, 0.9, (8, 4))

    T, Gs, Gr = nes.predict(x, ('T', 'Gs', 'Gr'))
    assert_close(T, nes.Traveltime(x), 1e-6)
    G_fd = central_diff(nes.Traveltime, x)
    assert_close(Gs, G_fd[:, :2], 3e-3)
    assert_close(Gr, G_fd[:, 2:], 3e-3)

    H_fd = central_diff(nes.GradientR, x)                               # d(Gr)/d[xs, xr]: (N, 2, 4)
    assert_close(nes.HessianSR(x), H_fd[:, :, :2].transpose(0, 2, 1), 3e-3)
    Hrr = H_fd[:, :, 2:]
    assert_close(nes.HessianR(x), np.stack([Hrr[:, 0, 0], Hrr[:, 0, 1], Hrr[:, 1, 1]], -1), 3e-3)
    assert_close(nes.LaplacianR(x), Hrr[:, 0, 0] + Hrr[:, 1, 1], 3e-3)
    Hss = central_diff(nes.GradientS, x)[:, :, :2]
    assert_close(nes.LaplacianS(x), Hss[:, 0, 0] + Hss[:, 1, 1], 3e-3)

    v = vel(x[:, 2:])
    Er = (np.sum(Gr**2, -1) * v**2 - 1) / 2
    assert_close(nes._predict(x, 'Er'), Er, 1e-5)

    swapped = np.concatenate([x[:, 2:], x[:, :2]], axis=-1)
    if reciprocity is not None:  # exact reciprocity (up to float32 rounding), also of the gradients
        assert_close(nes.Traveltime(swapped), T, 1e-6)
        assert_close(nes.GradientS(swapped), Gr, 1e-6)


def test_op_derivatives(vel):
    keras.utils.set_random_seed(2)
    nes = NES.NES_OP(xs=[0.3, 0.2], velocity=vel)
    nes.build_model(nl=3, nu=16, improved_mlp=True)
    x = RNG.uniform(0.4, 0.9, (8, 2))
    G = nes.Gradient(x)
    assert_close(G, central_diff(nes.Traveltime, x), 3e-3)
    H_fd = central_diff(nes.Gradient, x)
    assert_close(nes.Hessian(x), np.stack([H_fd[:, 0, 0], H_fd[:, 0, 1], H_fd[:, 1, 1]], -1), 3e-3)
    assert_close(nes.Laplacian(x), H_fd[:, 0, 0] + H_fd[:, 1, 1], 3e-3)
    assert_close(nes.Velocity(x), 1 / np.linalg.norm(G, axis=-1), 1e-6)
    assert nes.Traveltime(x.reshape(2, 4, 2)).shape == (2, 4)


def test_eikonal_residual_vanishes_for_analytic_solution(vel):
    xs = np.array([0.3, 0.2])
    xr = RNG.uniform(0, 1, (50, 2))
    grad = vel.dtime(xr, xs)
    assert_close(grad, central_diff(lambda z: vel.time(z, xs), xr, eps=1e-6), 1e-6)
    res = IsoEikonal()(keras.ops.convert_to_tensor(grad), keras.ops.convert_to_tensor(vel(xr)[:, None]))
    assert np.abs(keras.ops.convert_to_numpy(res)).max() < 1e-6


def test_velocity_fixes():
    loc = LocAnomaly(vmin=2.0, vmax=3.0, mus=[0.5, 0.5], sigmas=[0.2, 0.3], xmin=[0, 0], xmax=[1, 1])
    x = RNG.uniform(0, 1, (20, 2))
    assert_close(loc.gradient(x), central_diff(loc, x, eps=1e-6), 1e-6)

    F = RNG.uniform(1, 2, (10, 12))
    interp = Interpolator(F, np.linspace(0, 1, 10), np.linspace(0, 2, 12), method='nearest')
    restored = pickle.loads(pickle.dumps(interp))
    x = RNG.uniform([0, 0], [1, 2], (20, 2))
    assert np.array_equal(restored(x), interp(x))
    assert restored.Func.method == 'nearest'


LENSES = {  # kind: (factory, half-width of the square domain)
    'fisheye': (lambda b: MaxwellFishEye(2.0, 1.0, xmin=[-b, -b], xmax=[b, b]), 1.0),
    'hyperbolic': (lambda b: MaxwellFishEye(2.0, 1.0, high_velocity=True, xmin=[-b, -b], xmax=[b, b]), 0.65),
    'luneburg': (lambda b: LuneburgLens(2.0, 1.0, xmin=[-b, -b], xmax=[b, b]), 1.2),
    'luneburg-fast': (lambda b: LuneburgLens(2.0, 1.0, n0=0.6, xmin=[-b, -b], xmax=[b, b]), 1.2),
}


def make_lens(kind):
    factory, b = LENSES[kind]
    return factory(b), b


def rmae(t, t_ref):
    ok = np.isfinite(t_ref)  # Luneburg: closed form inside the lens only
    return np.abs(t - t_ref)[ok].sum() / np.abs(t_ref)[ok].sum()


@pytest.mark.parametrize('kind', LENSES)
def test_lens_closed_form(kind):
    lens, b = make_lens(kind)
    xs = np.array([0.3, -0.2])
    x = RNG.uniform(-b, b, (200, 2))
    x = x[np.isfinite(lens.time(x, xs)) & (np.abs(x - xs).sum(-1) > 1e-3)]
    grad = lens.dtime(x, xs)
    assert_close(grad, central_diff(lambda z: lens.time(z, xs), x, eps=1e-6), 1e-6)
    assert np.abs(np.linalg.norm(grad, axis=-1) * lens(x) - 1).max() < 1e-10   # |grad T| = 1/v exactly
    assert np.array_equal(lens.time(x, xs), lens.time(xs, x))                  # reciprocity

    # independent check: 2nd-order factored fast marching converges to the closed form
    eikonalfm = pytest.importorskip('eikonalfm')
    ax = np.linspace(-b, b, 401)
    h = ax[1] - ax[0]
    X = np.stack(np.meshgrid(ax, ax, indexing='ij'), -1)
    src = tuple(int(round((c + b) / h)) for c in xs)
    T_fmm = eikonalfm.factored_fast_marching(lens(X), src, (h, h), 2) * \
        eikonalfm.distance(X.shape[:-1], (h, h), src, indexing='ij')
    assert rmae(T_fmm, lens.time(X, np.array([ax[i] for i in src]))) < 1e-4


@pytest.mark.parametrize('kind', LENSES)
def test_nes_learns_lens(kind):
    """ End-to-end physics check: NES-OP and NES-TP against the exact traveltimes (RMAE, eq. 15 of the paper) """
    lens, b = make_lens(kind)
    keras.utils.set_random_seed(7)
    xs = np.array([0.3, -0.2])
    op = NES.NES_OP(xs, lens)
    op.build_model(nl=4, nu=32)
    op.train(4000, epochs=150, batch_size=1000, verbose=0)
    x = RNG.uniform(-b, b, (4000, 2))
    x = x[np.abs(x - xs).sum(-1) > 1e-2]
    assert rmae(op.Traveltime(x), lens.time(x, xs)) < 0.02

    tp = NES.NES_TP(lens)
    tp.build_model(nl=4, nu=32, reciprocity='first_layer')
    tp.train(8000, epochs=150, batch_size=2000, verbose=0)
    pairs = np.concatenate([RNG.uniform(-b / 2, b / 2, (4000, 2)), RNG.uniform(-b, b, (4000, 2))], -1)
    pairs = pairs[np.abs(pairs[:, 2:] - pairs[:, :2]).sum(-1) > 1e-2]
    assert rmae(tp.Traveltime(pairs), lens.time(pairs[:, 2:], pairs[:, :2])) < 0.03


########################################################################
#                             float64 mode
########################################################################


@pytest.fixture
def float64():
    """
        Computes in float64 during the test. Besides floatx, the default dtype policy of layers has to be set:
        Keras fixes it at the first layer ever built. JAX also needs 64-bit arrays enabled.
    """
    floatx, policy = keras.config.floatx(), keras.config.dtype_policy()
    keras.config.set_floatx('float64')
    keras.config.set_dtype_policy('float64')
    if keras.backend.backend() == 'jax':
        import jax
        x64 = jax.config.jax_enable_x64
        jax.config.update('jax_enable_x64', True)
    yield
    keras.config.set_floatx(floatx)
    keras.config.set_dtype_policy(policy)
    if keras.backend.backend() == 'jax':
        jax.config.update('jax_enable_x64', x64)


def test_float64_homogeneous(float64):
    """
        vmin = vmax gives T = R / v whatever the network. Any float32 step on the way (Keras 3.15 `ops.norm`
        for R, the inputs, the source position) would limit the error to ~1e-7.
    """
    hom = VerticalGradient(v0=3.0, a=0.0, xmin=[0., 0.], xmax=[1., 1.])
    xs = np.array([0.3, 0.2])                           # not representable in float32
    xr = RNG.uniform(0, 1, (8, 2))
    R = np.linalg.norm(xr - xs, axis=-1)

    op = NES.NES_OP(xs, hom)
    op.build_model(nl=2, nu=8, act='tanh')
    T, G = op.predict(xr, ('T', 'G'))
    assert T.dtype == np.float64
    assert np.abs(3 * T / R - 1).max() < 1e-14
    assert np.abs(3 * G - (xr - xs) / R[:, None]).max() < 1e-14

    tp = NES.NES_TP(hom)
    tp.build_model(nl=2, nu=8, reciprocity='first_layer')
    T = tp.Traveltime(np.concatenate([np.tile(xs, (len(xr), 1)), xr], axis=-1))
    assert np.abs(3 * T / R - 1).max() < 1e-14


@pytest.mark.parametrize('act', ['tanh', 'atan', 'sin', 'sinc'])
def test_float64_activations(float64, act):
    z = RNG.uniform(-2, 2, 64)
    ref = {'tanh': np.tanh, 'atan': np.arctan, 'sin': np.sin, 'sinc': lambda z: np.sin(z) / z}[act]
    out = keras.ops.convert_to_numpy(ACTS[act](keras.ops.convert_to_tensor(z)))
    assert out.dtype == np.float64
    assert np.abs(out - ref(z)).max() < 1e-14


def test_finite_at_source(vel):
    """ T = 0 and zero (sub)gradient at the source, where R = |xr - xs| is not differentiable """
    keras.utils.set_random_seed(8)
    nes = NES.NES_OP(xs=[0.3, 0.2], velocity=vel)
    nes.build_model(nl=2, nu=8)
    T, G = nes.predict(nes.xs, ('T', 'G'))
    assert T == 0 and np.array_equal(G, [0, 0])


########################################################################
#                         legacy (TF / Keras 2) models
########################################################################


REFS = dict(np.load(DATA / 'legacy_refs.npz'))


@pytest.mark.parametrize('tag', sorted({k.split('/')[0] for k in REFS}))
def test_legacy_models(tag):
    cls = NES.NES_OP if tag.startswith('OP') else NES.NES_TP
    nes = cls.load(DATA / 'legacy_models' / tag)
    x = REFS[f'{tag}/x']
    for key in [k.split('/')[1] for k in REFS if k.startswith(tag + '/') and not k.endswith('/x')]:
        assert_close(nes._predict(x, key), REFS[f'{tag}/{key}'], 1e-4)


########################################################################
#                       training, saving, transfer
########################################################################


@pytest.mark.parametrize('reciprocity', ['output', 'first_layer', 'invariant'])
def test_tp_training_reduces_loss(vel, reciprocity):
    keras.utils.set_random_seed(3)
    nes = NES.NES_TP(vel)
    nes.build_model(nl=2, nu=16, reciprocity=reciprocity, losses=['Er', 'Es'])
    h = nes.train(2000, epochs=8, batch_size=500, verbose=0)
    assert h.history['loss'][-1] < 0.7 * h.history['loss'][0]


def test_save_load_resume(vel, tmp_path):
    keras.utils.set_random_seed(4)
    nes = NES.NES_TP(vel, name='tp')
    nes.build_model(nl=2, nu=16, reciprocity='first_layer')
    nes.train(1000, epochs=2, batch_size=250, verbose=0)
    nes.save(tmp_path / 'tp', save_optimizer=True, training_data=True)

    loaded = NES.NES_TP.load(tmp_path / 'tp')
    x = RNG.uniform(0, 1, (16, 4))
    assert np.array_equal(loaded.Traveltime(x), nes.Traveltime(x))
    assert loaded.config == nes.config
    assert int(loaded.model.optimizer.iterations) == int(nes.model.optimizer.iterations) == 8
    for a, b in zip(loaded.model.optimizer.variables, nes.model.optimizer.variables):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(loaded.x_train['x'], nes.x_train['x'])

    # identical states train identically
    for model in (nes, loaded):
        model.model.fit(nes.x_train, batch_size=250, epochs=1, shuffle=False, verbose=0)
    assert_close(loaded.Traveltime(x), nes.Traveltime(x), 1e-5)


def test_op_save_load_transfer(vel, tmp_path):
    keras.utils.set_random_seed(5)
    nes = NES.NES_OP(xs=[0.5, 0.5], velocity=vel)
    nes.build_model(nl=2, nu=16)
    nes.train(1000, epochs=2, batch_size=250, verbose=0)
    nes.save(tmp_path / 'op')
    x = RNG.uniform(0, 1, (16, 2))
    assert np.array_equal(NES.NES_OP.load(tmp_path / 'op').Traveltime(x), nes.Traveltime(x))

    moved = nes.transfer(xs=[0.2, 0.2])
    assert np.allclose(moved.xs, [0.2, 0.2])
    assert all(np.array_equal(a, b) for a, b in zip(moved.net.get_weights(), nes.net.get_weights()))
    moved.train(500, epochs=1, batch_size=250, verbose=0)


def test_adaptive_sampling_callbacks(vel):
    from NES.experimental import RARsampling, FromCoarseToFineResampling, LRScheduler
    keras.utils.set_random_seed(6)
    nes = NES.NES_TP(vel)
    nes.build_model(nl=2, nu=16)
    rar = RARsampling(nes, m=50, res_pts=500, freq=1, eps=0.0, verbose=0)
    nes.train(1000, epochs=3, batch_size=250, callbacks=[rar], verbose=0)
    assert nes.data_generator.size == len(nes.x_train['x']) + 2 * 50

    nes.compile(decay=0)
    c2f = FromCoarseToFineResampling(nes, set_pts=(500, 700), tolerance=1e3, patience=1, verbose=0)
    lrs = LRScheduler(bounds=(1e3,), lrs=(1e-3, 5e-4), patience=1, cooldown=0)
    nes.train(500, epochs=3, batch_size=100, callbacks=[c2f, lrs], verbose=0)
    assert nes.data_generator.size == len(nes.x_train['x']) + 200
    assert np.isclose(float(keras.ops.convert_to_numpy(nes.model.optimizer.learning_rate)), 5e-4)


def test_legacy_api_surface(vel):
    nes = NES.NES_TP(vel)
    nes.build_model(nl=2, nu=8, reciprocity=True)
    assert set(nes.outs.keys()) == set()           # built lazily
    assert isinstance(nes.outs['T'], keras.Model)
    assert NES.misc.Marmousi is NES.velocity.Marmousi
    vm = NES.misc.MarmousiSmoothedPart()
    assert vm.dim == 2 and vm.min < vm.max
    with pytest.raises(ValueError):
        nes.build_model(losses=['T'])


########################################################################
#                       FLOPs and hyperparameter search
########################################################################


def test_flops_reflect_reciprocity(vel):
    flops = {}
    for mode in RECIPROCITY_MODES:
        nes = NES.NES_TP(vel)
        nes.build_model(nl=4, nu=50, reciprocity=mode)
        flops[mode] = nes.net.flops()
    assert flops['output'] > 1.9 * flops[None]
    assert flops[None] < flops['first_layer'] < 1.1 * flops[None]
    assert flops['invariant'] < 1.05 * flops[None]


def test_median_stopping_rule():
    optuna = pytest.importorskip('optuna', minversion='5.0')
    from NES.hpo import MedianStoppingRule
    study = optuna.create_study(directions=['minimize', 'minimize'])
    dist = {'x': optuna.distributions.FloatDistribution(0, 1)}
    for curve, cost in [([1.0, 0.5, 0.2], 10), ([1.0, 0.6, 0.3], 10), ([2.0, 1.0, 0.8], 10), ([0.1, 0.05, 0.01], 1e6)]:
        study.add_trial(optuna.trial.create_trial(values=[curve[-1], cost], params={'x': 0.5}, distributions=dist,
                                                  user_attrs={'curve': curve}))
    rule = MedianStoppingRule(n_startup_trials=3, n_warmup_steps=1, reference='cheaper')
    # running averages at step 1 of the three cheap trials: 0.75, 0.8, 1.5 -> median 0.8
    assert rule.should_prune(study, [1.0, 0.9], flops=100)
    assert not rule.should_prune(study, [1.0, 0.7], flops=100)
    assert not rule.should_prune(study, [1.0], flops=100)                  # warm-up
    assert not rule.should_prune(study, [1.0, 0.9], flops=5)               # fewer than 3 cheaper trials
    assert rule.should_prune(study, [1.0, float('nan')], flops=5)           # diverged
    rule_all = MedianStoppingRule(n_startup_trials=3, n_warmup_steps=1, reference='all')
    assert rule_all.should_prune(study, [1.0, 0.8], flops=5)               # median of 0.075, 0.75, 0.8, 1.5 = 0.775


def test_tune_smoke(vel):
    pytest.importorskip('optuna', minversion='5.0')
    import NES.hpo as hpo
    space = hpo.search_space(nl=(2, 3), nu=(8, 16), act='ad-gauss-1', improved_mlp=False, n_train=(300, 600))
    study = hpo.tune(vel, 'TP', n_trials=3, epochs=4, eval_every=2, n_val=300, search_space=space, verbose=False)
    front = hpo.pareto_front(study)
    assert front and all(np.isfinite(r['loss']) and r['flops'] > 0 for r in front)
    assert 'act' not in front[0]['params']                                  # fixed values are not searched
    nes, train_kw = hpo.build(vel, front[0]['params'], 'TP', search_space=space)
    assert nes.config['nl'] == front[0]['params']['nl'] and nes.config['act'] == 'ad-gauss-1'
