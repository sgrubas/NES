"""
Multi-objective hyperparameter optimization (HPO) and importance (HPI) for NES-OP / NES-TP with Optuna >= 5.

Objectives, both minimized:
    loss  : mean |H_2| (eqs. 13-14 of Grubas et al., 2023) on a fixed validation set of collocation points.
            The loss is non-dimensional, and its relation to RMAE is close to model-independent (paper, Fig. 10),
            so it serves as a proxy of the RMAE w.r.t. a reference solver (e.g. 2nd-order factored FMM).
            Evaluated with the same metric for every trial, whatever its training set or learning rate.
    flops : training cost = FLOPs of one forward evaluation (`TraveltimeNet.flops`) x collocation points x epochs,
            a deterministic, backend-independent proxy of the training time. It ignores per-step overhead, so it
            does not penalize many small batches; cap `n_batches` in the search space if that matters (GPU).

Sampler : `optuna.samplers.TPESampler`, the multi-objective TPE (MOTPE; Ozaki et al., 2022), which is the default
          for multi-objective studies since Optuna 5.0.
Pruning : median stopping rule (Golovin et al., 2017, sec. 3.2.2) on the validation-loss learning curve.
          Optuna does not support `trial.report` in multi-objective studies, so the rule is applied by a Keras
          callback that raises `optuna.TrialPruned`. Pruned trials are kept by MOTPE as bad examples.
          Two-objective adaptation (default `reference='cheaper'`): a trial is compared only with completed trials
          of lower or equal FLOPs, since it can reach the Pareto front only by beating the cheaper ones on loss;
          comparing with all trials would prune the small networks that form the low-cost end of the front.
HPI     : PED-ANOVA (Watanabe et al., 2023), the default of `optuna.importance` since Optuna 5.0. Without a target
          it measures importance for reaching the Pareto front; with a target, for one objective.

References
    Golovin D., Solnik B., Moitra S., Kochanski G., Karro J., Sculley D. (2017). Google Vizier: A service for
        black-box optimization. Proc. 23rd ACM SIGKDD (KDD 2017).
        https://doi.org/10.1145/3097983.3098043
    Ozaki Y., Tanigaki Y., Watanabe S., Nomura M., Onishi M. (2022). Multiobjective tree-structured Parzen estimator.
        JAIR 73, 1209-1250. https://doi.org/10.1613/jair.1.13188
    Watanabe S., Bansal A., Hutter F. (2023). PED-ANOVA: Efficiently quantifying hyperparameter importance in
        arbitrary subspaces. IJCAI. https://arxiv.org/abs/2304.10255
    Grubas S., Duchkov A., Loginov G. (2023). Neural Eikonal solver. JCP 474, 111789.
        https://doi.org/10.1016/j.jcp.2022.111789

Example
    study = NES.hpo.tune(velocity, solver='TP', n_trials=100, epochs=500)
    front = NES.hpo.pareto_front(study)
    nes, train_kw = NES.hpo.build(velocity, front[3]['params'], solver='TP')
    nes.train(**train_kw, epochs=3000)
"""
import time
import warnings
import numpy as np
import keras

from .NeuralEikonalSolver import NES_OP, NES_TP

try:
    import optuna
except ImportError as e:  # optional dependency
    raise ImportError("NES.hpo needs Optuna >= 5: pip install 'optuna>=5'") from e

OBJECTIVES = ('loss', 'flops')


def loss_to_rmae(loss):
    """ Empirical relation between the non-dimensional loss and RMAE (NES_EarlyStopping default, paper Fig. 10) """
    return np.asarray(loss) * 10**(-0.16)


#######################################################################
                        ### SEARCH SPACE ###
#######################################################################


def _suggest(trial, name, spec, log=False):
    if isinstance(spec, list):                                   # categorical
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, tuple):                                  # range (low, high)
        lo, hi = spec
        if isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)):
            return trial.suggest_int(name, lo, hi, log=log)
        return trial.suggest_float(name, lo, hi, log=log)
    return spec                                                  # fixed value


def search_space(nl=(2, 8), nu=(16, 128), act=['ad-gauss-1', 'gauss', 'ad-tanh-1', 'ad-sin-1'],
                 improved_mlp=[False, True], reciprocity=['output', 'first_layer', 'invariant'],
                 lr=(1e-3, 2e-2), decay=(1e-5, 1e-2), n_train=None, n_batches=[1, 2, 4, 8], **fixed_build):
    """
        Search space for `tune`: a (low, high) tuple is a range (log scale for nu, lr, decay, n_train),
        a list is categorical, any other value is fixed. `n_train` defaults to (1e3, 2e4) points for NES-OP
        and (1e4, 2e5) pairs for NES-TP. Extra keyword arguments are fixed arguments of `build_model`.
        Returns callable(trial, solver) -> {'build': {...}, 'compile': {...}, 'train': {...}}.

        The Hamiltonian power p and the loss metric stay fixed (p=2, L1): changing them changes the scale of
        the loss and would make the loss objective incomparable between trials.
    """
    def space(trial, solver):
        op = solver == 'OP'
        build = dict(nl=_suggest(trial, 'nl', nl), nu=_suggest(trial, 'nu', nu, log=True),
                     act=_suggest(trial, 'act', act), improved_mlp=_suggest(trial, 'improved_mlp', improved_mlp),
                     **fixed_build)
        if not op:
            build['reciprocity'] = _suggest(trial, 'reciprocity', reciprocity)
        compile_kw = dict(lr=_suggest(trial, 'lr', lr, log=True), decay=_suggest(trial, 'decay', decay, log=True))
        n_range = n_train if n_train is not None else ((1000, 20000) if op else (10000, 200000))
        n = _suggest(trial, 'n_train', n_range, log=True)
        n_b = _suggest(trial, 'n_batches', n_batches)
        return dict(build=build, compile=compile_kw, train=dict(x_train=n, batch_size=int(np.ceil(n / n_b))))
    return space


default_search_space = search_space()


def _make(velocity, space, solver, xs):
    nes = NES_OP(xs, velocity) if solver == 'OP' else NES_TP(velocity)
    nes.build_model(**space['build'])
    nes.compile(**space['compile'])
    return nes, space['train']


def build(velocity, params, solver='TP', xs=None, search_space=default_search_space):
    """
        Builds and compiles a solver with the hyperparameters `params` of a trial (e.g. from `pareto_front`).
        Returns (solver instance, train kwargs); call `nes.train(**train_kw, epochs=...)`.
    """
    return _make(velocity, search_space(optuna.trial.FixedTrial(params), solver), solver, xs)


#######################################################################
                            ### PRUNING ###
#######################################################################


class MedianStoppingRule:
    """
        Median stopping rule of Golovin et al. (2017, sec. 3.2.2): stop a trial at step s if its best loss by step s
        is strictly worse than the median of the running averages (over steps 1..s) of the completed trials'
        loss curves. Steps are the validation checkpoints.

        n_startup_trials : minimum number of comparable completed trials before pruning starts
        n_warmup_steps : no pruning before this many checkpoints
        reference : 'cheaper' (completed trials with FLOPs <= the trial's FLOPs, two-objective adaptation, see module
                    docstring) or 'all' (the original rule, single-objective behaviour)
    """
    def __init__(self, n_startup_trials=5, n_warmup_steps=3, reference='cheaper'):
        assert reference in ('cheaper', 'all')
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.reference = reference

    def should_prune(self, study, curve, flops):
        step = len(curve) - 1
        if not np.all(np.isfinite(curve)):
            return True  # diverged
        if step < self.n_warmup_steps:
            return False
        running = []
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            other = t.user_attrs.get('curve')
            if other is None or len(other) <= step:
                continue
            if self.reference == 'cheaper' and t.values[1] > flops:
                continue
            running.append(np.mean(other[:step + 1]))
        if len(running) < self.n_startup_trials:
            return False
        return min(curve) > np.median(running)


class _PruningCallback(keras.callbacks.Callback):
    """ Evaluates the validation loss every `every` epochs, records the curve and applies the stopping rule """
    def __init__(self, trial, study, rule, evaluate, flops, every):
        super().__init__()
        self.trial, self.study, self.rule = trial, study, rule
        self.evaluate, self.flops, self.every = evaluate, flops, every
        self.curve = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every:
            return
        self.curve.append(float(self.evaluate()))
        self.trial.set_user_attr('curve', self.curve)
        if self.rule is not None and self.rule.should_prune(self.study, self.curve, self.flops):
            raise optuna.TrialPruned(f'epoch {epoch + 1}: loss {self.curve[-1]:.3e}')


#######################################################################
                        ### OPTIMIZATION ###
#######################################################################


def validation_points(velocity, solver='TP', xs=None, n=10000, seed=0, sing_eps=1e-2):
    """ Fixed random collocation points (receivers for NES-OP, pairs for NES-TP), away from the source singularity """
    rep = 1 if solver == 'OP' else 2
    x = np.random.default_rng(seed).uniform(np.tile(velocity.xmin, rep), np.tile(velocity.xmax, rep),
                                            size=(n, rep * velocity.dim))
    d = velocity.dim
    src = np.asarray(xs) if solver == 'OP' else x[:, :d]
    return x[np.abs(x[:, -d:] - src).sum(-1) > sing_eps]


def tune(velocity, solver='TP', xs=None, n_trials=50, epochs=300, timeout=None, study=None,
         search_space=default_search_space, pruning=MedianStoppingRule(), eval_every=10,
         n_val=10000, reference=None, seed=0, verbose=True, **study_kwargs):
    """
        Runs (or continues) a two-objective study (loss, flops) for NES-OP or NES-TP on `velocity`.

        velocity : NES.velocity.BaseVelocity instance
        solver : 'OP' (needs source `xs`) or 'TP'
        n_trials, timeout : budget of the study (see optuna.Study.optimize)
        epochs : training epochs of every trial
        study : existing study to continue; otherwise created with `study_kwargs` (e.g. storage, study_name,
                load_if_exists) and TPESampler(seed=seed)
        search_space : `search_space(...)` result, or any callable(trial, solver) returning the same dict
        pruning : MedianStoppingRule instance or None
        eval_every : epochs between validation checkpoints (steps of the stopping rule)
        n_val : number of validation points
        reference : optional (x, T_ref) to record the true RMAE of every trial as user attribute 'rmae'
                    (not an objective), e.g. from an analytic solution or `fmm_reference`
        Returns the optuna.Study
    """
    op = solver == 'OP'
    if op and xs is None:
        raise ValueError("NES-OP needs the source location `xs`")
    if study is None:
        study = optuna.create_study(directions=['minimize', 'minimize'],
                                    sampler=optuna.samplers.TPESampler(seed=seed), **study_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')          # experimental API
            study.set_metric_names(list(OBJECTIVES))
    x_val = validation_points(velocity, solver, xs, n_val, seed)
    residual = 'E' if op else 'Er'

    def objective(trial):
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed + trial.number)
        nes, train_kw = _make(velocity, search_space(trial, solver), solver, xs)
        flops = nes.net.flops() * train_kw['x_train'] * epochs
        trial.set_user_attr('flops_per_eval', nes.net.flops())
        trial.set_user_attr('n_params', int(nes.net.count_params()))
        evaluate = lambda: np.mean(np.abs(nes._predict(x_val, residual)))
        callback = _PruningCallback(trial, study, pruning, evaluate, flops, eval_every)
        t0 = time.perf_counter()
        nes.train(**train_kw, epochs=epochs, callbacks=[callback], verbose=0)
        trial.set_user_attr('train_seconds', time.perf_counter() - t0)
        loss = float(evaluate())
        if reference is not None:
            x_ref, t_ref = reference
            ok = np.isfinite(t_ref)
            trial.set_user_attr('rmae', float(np.abs(nes.Traveltime(x_ref)[ok] - t_ref[ok]).sum()
                                              / np.abs(t_ref[ok]).sum()))
        if not np.isfinite(loss):
            raise optuna.TrialPruned('diverged')
        return loss, float(flops)

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


#######################################################################
                        ### ANALYSIS ###
#######################################################################


def pareto_front(study):
    """ Pareto-optimal trials sorted by cost: list of dicts (number, loss, flops, approx_rmae, rmae, params, ...) """
    rows = []
    for t in sorted(study.best_trials, key=lambda t: t.values[1]):
        rows.append(dict(number=t.number, loss=t.values[0], flops=t.values[1],
                         approx_rmae=float(loss_to_rmae(t.values[0])), rmae=t.user_attrs.get('rmae'),
                         n_params=t.user_attrs.get('n_params'), train_seconds=t.user_attrs.get('train_seconds'),
                         params=dict(t.params)))
    return rows


def param_importances(study, evaluator=None):
    """
        Hyperparameter importances with PED-ANOVA (Optuna 5 default evaluator):
        'pareto' (for reaching the Pareto front) and 'loss' (for the loss alone). The FLOPs objective is not
        included: it is a known function of nl, nu, n_train, improved_mlp and reciprocity.
    """
    get = optuna.importance.get_param_importances
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {'pareto': get(study, evaluator=evaluator),
                'loss': get(study, evaluator=evaluator, target=lambda t: t.values[0])}


def fmm_reference(velocity, sources, shape):
    """
        Reference traveltimes by the 2nd-order factored fast marching method (eikonalfm, Treister & Haber 2016),
        as in the paper. Sources are snapped to the grid.

        sources : (Ns, dim) source locations;  shape : grid shape, e.g. (101, 101)
        Returns (x, T): x (Ns, *shape, 2*dim) source-receiver pairs, T (Ns, *shape).
        For NES-OP with one source use x[0, ..., dim:] and T[0].
    """
    import eikonalfm
    axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(velocity.xmin, velocity.xmax, shape)]
    h = [a[1] - a[0] for a in axes]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    v = velocity(grid)
    xs_all, t_all = [], []
    for s in np.atleast_2d(sources):
        idx = tuple(int(np.clip(round((c - a[0]) / hi), 0, len(a) - 1)) for c, a, hi in zip(s, axes, h))
        tau1 = eikonalfm.factored_fast_marching(v, idx, h, 2)
        t_all.append(tau1 * eikonalfm.distance(v.shape, h, idx, indexing='ij'))
        src = np.array([a[i] for a, i in zip(axes, idx)])
        xs_all.append(np.concatenate([np.broadcast_to(src, grid.shape), grid], axis=-1))
    return np.stack(xs_all), np.stack(t_all)
