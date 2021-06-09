from warnings import simplefilter
import numpy as np
import numbers
import math
from scipy.optimize import minimize
import pandas as pd
import time

# copy-pasted from scikit-learn utils/validation.py
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None (or np.random), return the RandomState singleton used
    by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    If seed is a new-style np.random.Generator, return it.
    Otherwise, raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    try:
        # Generator is only available in numpy >= 1.17
        if isinstance(seed, np.random.Generator):
            return seed
    except AttributeError:
        pass
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

class OptimizeResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.
    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

class ObjectiveFunWrapper(object):

    def __init__(self, func, maxfun=1e7, *args):
        self.func = func
        self.args = args
        # Number of objective function evaluations
        self.nfev = 0
        # Number of gradient function evaluation if used
        self.ngev = 0
        # Number of hessian of the objective function if used
        self.nhev = 0
        self.maxfun = maxfun

    def fun(self, x):
        self.nfev += 1
        return self.func(x, *self.args)

class LocalSearchWrapper(object):
    """
    Class used to wrap around the minimizer used for local search
    Default local minimizer is SciPy minimizer L-BFGS-B
    """

    LS_MAXITER_RATIO = 6
    LS_MAXITER_MIN = 100
    LS_MAXITER_MAX = 1000

    def __init__(self, search_bounds, func_wrapper, **kwargs):
        self.func_wrapper = func_wrapper
        self.kwargs = kwargs
        self.minimizer = minimize
        bounds_list = list(zip(*search_bounds))
        self.lower = np.array(bounds_list[0])
        self.upper = np.array(bounds_list[1])

        # If no minimizer specified, use SciPy minimize with 'L-BFGS-B' method
        if not self.kwargs:
            n = len(self.lower)
            ls_max_iter = min(max(n * self.LS_MAXITER_RATIO,
                                  self.LS_MAXITER_MIN),
                              self.LS_MAXITER_MAX)
            self.kwargs['method'] = 'L-BFGS-B'
            self.kwargs['options'] = {
                'maxiter': ls_max_iter,
            }
            self.kwargs['bounds'] = list(zip(self.lower, self.upper))

    def local_search(self, x, e):
        # Run local search from the given x location where energy value is e
        x_tmp = np.copy(x)
        mres = self.minimizer(self.func_wrapper.fun, x, **self.kwargs)
        if 'njev' in mres:
            self.func_wrapper.ngev += mres.njev
        if 'nhev' in mres:
            self.func_wrapper.nhev += mres.nhev
        # Check if is valid value
        is_finite = np.all(np.isfinite(mres.x)) and np.isfinite(mres.fun)
        in_bounds = np.all(mres.x >= self.lower) and np.all(
            mres.x <= self.upper)
        is_valid = is_finite and in_bounds

        # Use the new point only if it is valid and return a better results
        if is_valid and mres.fun < e:
            return mres.fun, mres.x
        else:
            return e, x_tmp

class EnergyState(object):
    """
    Class used to record the energy state. At any time, it knows what is the
    currently used coordinates and the most recent best location.
    Parameters
    ----------
    lower : array_like
        A 1-D NumPy ndarray containing lower bounds for generating an initial
        random components in the `reset` method.
    upper : array_like
        A 1-D NumPy ndarray containing upper bounds for generating an initial
        random components in the `reset` method
        components. Neither NaN or inf are allowed.
    callback : callable, ``callback(x, f, context)``, optional
        A callback function which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and `context` has value in [0, 1, 2]
    """
    # Maximimum number of trials for generating a valid starting point
    MAX_REINIT_COUNT = 1000

    def __init__(self, lower, upper, callback=None):
        self.ebest = None
        self.current_energy = None
        self.current_location = None
        self.xbest = None
        self.lower = lower
        self.upper = upper
        self.callback = callback

    def reset(self, func_wrapper, rand_gen, x0=None):
        """
        Initialize current location is the search domain. If `x0` is not
        provided, a random location within the bounds is generated.
        """
        if x0 is None:
            # self.current_location = rand_gen.uniform(self.lower, self.upper,
            #                                          size=len(self.lower))
            self.current_location = rand_gen.randint(self.lower, self.upper,
                                                     size=len(self.lower))
        else:
            self.current_location = np.copy(x0)
        init_error = True
        reinit_counter = 0
        while init_error:
            self.current_energy = func_wrapper.fun(self.current_location)
            if self.current_energy is None:
                raise ValueError('Objective function is returning None')
            if (not np.isfinite(self.current_energy) or np.isnan(
                    self.current_energy)):
                if reinit_counter >= EnergyState.MAX_REINIT_COUNT:
                    init_error = False
                    message = (
                        'Stopping algorithm because function '
                        'create NaN or (+/-) infinity values even with '
                        'trying new random parameters'
                    )
                    raise ValueError(message)
                self.current_location = rand_gen.uniform(self.lower,
                                                         self.upper,
                                                         size=self.lower.size)
                reinit_counter += 1
            else:
                init_error = False
            # If first time reset, initialize ebest and xbest
            if self.ebest is None and self.xbest is None:
                self.ebest = self.current_energy
                self.xbest = np.copy(self.current_location)
            # Otherwise, we keep them in case of reannealing reset

    def update_best(self, e, x, context):
        self.ebest = e
        self.xbest = np.copy(x)
        if self.callback is not None:
            val = self.callback(x, e, context)
            if val is not None:
                if val:
                    return('Callback function requested to stop early by '
                           'returning True')

    def update_current(self, e, x):
        self.current_energy = e
        self.current_location = np.copy(x)


class VisitingDistribution(object):
    """
    Class used to generate new coordinates based on the distorted
    Cauchy-Lorentz distribution. Depending on the steps within the strategy
    chain, the class implements the strategy for generating new location
    changes.
    Parameters
    ----------
    lb : array_like
        A 1-D NumPy ndarray containing lower bounds of the generated
        components. Neither NaN or inf are allowed.
    ub : array_like
        A 1-D NumPy ndarray containing upper bounds for the generated
        components. Neither NaN or inf are allowed.
    visiting_param : float
        Parameter for visiting distribution. Default value is 2.62.
        Higher values give the visiting distribution a heavier tail, this
        makes the algorithm jump to a more distant region.
        The value range is (0, 3]. It's value is fixed for the life of the
        object.
    rand_gen : {`~numpy.random.RandomState`, `~numpy.random.Generator`}
        A `~numpy.random.RandomState`, `~numpy.random.Generator` object
        for using the current state of the created random generator container.
    """
    TAIL_LIMIT = 1.e8
    MIN_VISIT_BOUND = 1.e-10

    def __init__(self, lb, ub, visiting_param, rand_gen):
        # if you wish to make _visiting_param adjustable during the life of
        # the object then _factor2, _factor3, _factor5, _d1, _factor6 will
        # have to be dynamically calculated in `visit_fn`. They're factored
        # out here so they don't need to be recalculated all the time.
        self._visiting_param = visiting_param
        self.rand_gen = rand_gen
        self.lower = lb
        self.upper = ub
        self.bound_range = ub - lb

        # these are invariant numbers unless visiting_param changes
        self._factor2 = np.exp((4.0 - self._visiting_param) * np.log(
            self._visiting_param - 1.0))
        self._factor3 = np.exp((2.0 - self._visiting_param) * np.log(2.0)
                               / (self._visiting_param - 1.0))
        self._factor4_p = np.sqrt(np.pi) * self._factor2 / (self._factor3 * (
            3.0 - self._visiting_param))

        self._factor5 = 1.0 / (self._visiting_param - 1.0) - 0.5
        self._d1 = 2.0 - self._factor5
        self._factor6 = np.pi * (1.0 - self._factor5) / np.sin(
            np.pi * (1.0 - self._factor5)) / np.exp(math.lgamma(self._d1))

    def visiting(self, x, step, temperature):
        """ Based on the step in the strategy chain, new coordinated are
        generated by changing all components is the same time or only
        one of them, the new values are computed with visit_fn method
        """
        dim = x.size
        if step < dim:
            # Changing all coordinates with a new visiting value
            visits = self.visit_fn(temperature, dim)
            upper_sample, lower_sample = self.rand_gen.uniform(size=2)
            visits[visits > self.TAIL_LIMIT] = self.TAIL_LIMIT * upper_sample
            visits[visits < -self.TAIL_LIMIT] = -self.TAIL_LIMIT * lower_sample
            x_visit = visits + x
            a = x_visit - self.lower
            b = np.fmod(a, self.bound_range) + self.bound_range
            x_visit = np.fmod(b, self.bound_range) + self.lower
            x_visit[np.fabs(
                x_visit - self.lower) < self.MIN_VISIT_BOUND] += 1.e-10
        else:
            # Changing only one coordinate at a time based on strategy
            # chain step
            x_visit = np.copy(x)
            visit = self.visit_fn(temperature, 1)
            if visit > self.TAIL_LIMIT:
                visit = self.TAIL_LIMIT * self.rand_gen.uniform()
                # visit = self.TAIL_LIMIT * self.rand_gen.randint()
            elif visit < -self.TAIL_LIMIT:
                visit = -self.TAIL_LIMIT * self.rand_gen.uniform()
                # visit = -self.TAIL_LIMIT * self.rand_gen.randint()
            index = step - dim
            x_visit[index] = visit + x[index]
            a = x_visit[index] - self.lower[index]
            b = np.fmod(a, self.bound_range[index]) + self.bound_range[index]
            x_visit[index] = np.fmod(b, self.bound_range[
                index]) + self.lower[index]
            if np.fabs(x_visit[index] - self.lower[
                    index]) < self.MIN_VISIT_BOUND:
                x_visit[index] += self.MIN_VISIT_BOUND
        return x_visit

    def visit_fn(self, temperature, dim):
        """ Formula Visita from p. 405 of reference [2] """
        x, y = self.rand_gen.normal(size=(dim, 2)).T

        factor1 = np.exp(np.log(temperature) / (self._visiting_param - 1.0))
        factor4 = self._factor4_p * factor1

        # sigmax
        x *= np.exp(-(self._visiting_param - 1.0) * np.log(
            self._factor6 / factor4) / (3.0 - self._visiting_param))

        den = np.exp((self._visiting_param - 1.0) * np.log(np.fabs(y)) /
                     (3.0 - self._visiting_param))

        return x / den

class StrategyChain(object):
    """
    Class that implements within a Markov chain the strategy for location
    acceptance and local search decision making.
    Parameters
    ----------
    acceptance_param : float
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    visit_dist : VisitingDistribution
        Instance of `VisitingDistribution` class.
    func_wrapper : ObjectiveFunWrapper
        Instance of `ObjectiveFunWrapper` class.
    minimizer_wrapper: LocalSearchWrapper
        Instance of `LocalSearchWrapper` class.
    rand_gen : {`~numpy.random.RandomState`, `~numpy.random.Generator`}
        A `~numpy.random.RandomState` or `~numpy.random.Generator`
        object for using the current state of the created random generator
        container.
    energy_state: EnergyState
        Instance of `EnergyState` class.
    """
    def __init__(self, acceptance_param, visit_dist, func_wrapper,
                 minimizer_wrapper, rand_gen, energy_state):
        # Local strategy chain minimum energy and location
        self.emin = energy_state.current_energy
        self.xmin = np.array(energy_state.current_location)
        # Global optimizer state
        self.energy_state = energy_state
        # Acceptance parameter
        self.acceptance_param = acceptance_param
        # Visiting distribution instance
        self.visit_dist = visit_dist
        # Wrapper to objective function
        self.func_wrapper = func_wrapper
        # Wrapper to the local minimizer
        self.minimizer_wrapper = minimizer_wrapper
        self.not_improved_idx = 0
        self.not_improved_max_idx = 1000
        self._rand_gen = rand_gen
        self.temperature_step = 0
        self.K = 100 * len(energy_state.current_location)
        self.lastvisit = []

    def accept_reject(self, j, e, x_visit):
        r = self._rand_gen.uniform()
        pqv_temp = 1.0 - ((1.0 - self.acceptance_param) *
            (e - self.energy_state.current_energy) / self.temperature_step)
        if pqv_temp <= 0.:
            pqv = 0.
        else:
            pqv = np.exp(np.log(pqv_temp) / (
                1. - self.acceptance_param))

        if r <= pqv:
            # We accept the new location and update state
            self.energy_state.update_current(e, x_visit)
            self.xmin = np.copy(self.energy_state.current_location)

        # No improvement for a long time
        if self.not_improved_idx >= self.not_improved_max_idx:
            if j == 0 or self.energy_state.current_energy < self.emin:
                self.emin = self.energy_state.current_energy
                self.xmin = np.copy(self.energy_state.current_location)

    def run(self, step, temperature):
        self.temperature_step = temperature / float(step + 1)
        self.not_improved_idx += 1
        for j in range(self.energy_state.current_location.size * 2):
            if j == 0:
                if step == 0:
                    self.energy_state_improved = True
                else:
                    self.energy_state_improved = False
            x_visit = self.visit_dist.visiting(
                self.energy_state.current_location, j, temperature)
            for i in range(len(x_visit)):
                x_visit[i] = round(x_visit[i])
            if np.array_equal(x_visit, self.lastvisit) :
                return
            self.lastvisit = x_visit
            # Calling the objective function
            e = self.func_wrapper.fun(x_visit)
            if e < self.energy_state.current_energy:
                # We have got a better energy value
                self.energy_state.update_current(e, x_visit)
                if e < self.energy_state.ebest:
                    val = self.energy_state.update_best(e, x_visit, 0)
                    if val is not None:
                        if val:
                            return val
                    self.energy_state_improved = True
                    self.not_improved_idx = 0
            else:
                # We have not improved but do we accept the new location?
                self.accept_reject(j, e, x_visit)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during annealing')
        # End of StrategyChain loop

    def local_search(self):
        # Decision making for performing a local search
        # based on strategy chain results
        # If energy has been improved or no improvement since too long,
        # performing a local search with the best strategy chain location
        if self.energy_state_improved:
            # Global energy has improved, let's see if LS improves further
            e, x = self.minimizer_wrapper.local_search(self.energy_state.xbest,
                                                       self.energy_state.ebest)
            if e < self.energy_state.ebest:
                self.not_improved_idx = 0
                val = self.energy_state.update_best(e, x, 1)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during local search')
        # Check probability of a need to perform a LS even if no improvement
        do_ls = False
        if self.K < 90 * len(self.energy_state.current_location):
            pls = np.exp(self.K * (
                self.energy_state.ebest - self.energy_state.current_energy) /
                self.temperature_step)
            if pls >= self._rand_gen.uniform():
                do_ls = True
        # Global energy not improved, let's see what LS gives
        # on the best strategy chain location
        if self.not_improved_idx >= self.not_improved_max_idx:
            do_ls = True
        if do_ls:
            e, x = self.minimizer_wrapper.local_search(self.xmin, self.emin)
            self.xmin = np.copy(x)
            self.emin = e
            self.not_improved_idx = 0
            self.not_improved_max_idx = self.energy_state.current_location.size
            if e < self.energy_state.ebest:
                val = self.energy_state.update_best(
                    self.emin, self.xmin, 2)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during dual annealing')

def dual_annealing(func, bounds, args=(), maxiter=1000,
                   local_search_options={}, initial_temp=5230.,
                   restart_temp_ratio=2.e-5, visit=2.62, accept=-5.0,
                   maxfun=1e7, seed=None, no_local_search=False,
                   callback=None, x0=None):
    """
    Find the global minimum of a function using Dual Annealing.
    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence, shape (n, 2)
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining bounds for the objective function parameter.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the
        objective function.
    maxiter : int, optional
        The maximum number of global search iterations. Default value is 1000.
    local_search_options : dict, optional
        Extra keyword arguments to be passed to the local minimizer
        (`minimize`). Some important options could be:
        ``method`` for the minimizer method to use and ``args`` for
        objective function additional arguments.
    initial_temp : float, optional
        The initial temperature, use higher values to facilitates a wider
        search of the energy landscape, allowing dual_annealing to escape
        local minima that it is trapped in. Default value is 5230. Range is
        (0.01, 5.e4].
    restart_temp_ratio : float, optional
        During the annealing process, temperature is decreasing, when it
        reaches ``initial_temp * restart_temp_ratio``, the reannealing process
        is triggered. Default value of the ratio is 2e-5. Range is (0, 1).
    visit : float, optional
        Parameter for visiting distribution. Default value is 2.62. Higher
        values give the visiting distribution a heavier tail, this makes
        the algorithm jump to a more distant region. The value range is (0, 3].
    accept : float, optional
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    maxfun : int, optional
        Soft limit for the number of objective function calls. If the
        algorithm is in the middle of a local search, this number will be
        exceeded, the algorithm will stop just after the local search is
        done. Default value is 1e7.
    seed : {int, `~numpy.random.RandomState`, `~numpy.random.Generator`}, optional
        If `seed` is not specified the `~numpy.random.RandomState` singleton is
        used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with `seed`.
        If `seed` is already a ``RandomState`` or ``Generator`` instance, then
        that instance is used.
        Specify `seed` for repeatable minimizations. The random numbers
        generated with this seed only affect the visiting distribution function
        and new coordinates generation.
    no_local_search : bool, optional
        If `no_local_search` is set to True, a traditional Generalized
        Simulated Annealing will be performed with no local search
        strategy applied.
    callback : callable, optional
        A callback function with signature ``callback(x, f, context)``,
        which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and ``context`` has value in [0, 1, 2], with the
        following meaning:
            - 0: minimum detected in the annealing process.
            - 1: detection occurred in the local search process.
            - 2: detection done in the dual annealing process.
        If the callback implementation returns True, the algorithm will stop.
    x0 : ndarray, shape(n,), optional
        Coordinates of a single N-D starting point.
    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See `OptimizeResult` for a description of other attributes.
    Notes
    -----
    This function implements the Dual Annealing optimization. This stochastic
    approach derived from [3]_ combines the generalization of CSA (Classical
    Simulated Annealing) and FSA (Fast Simulated Annealing) [1]_ [2]_ coupled
    to a strategy for applying a local search on accepted locations [4]_.
    An alternative implementation of this same algorithm is described in [5]_
    and benchmarks are presented in [6]_. This approach introduces an advanced
    method to refine the solution found by the generalized annealing
    process. This algorithm uses a distorted Cauchy-Lorentz visiting
    distribution, with its shape controlled by the parameter :math:`q_{v}`
    .. math::
        g_{q_{v}}(\\Delta x(t)) \\propto \\frac{ \\
        \\left[T_{q_{v}}(t) \\right]^{-\\frac{D}{3-q_{v}}}}{ \\
        \\left[{1+(q_{v}-1)\\frac{(\\Delta x(t))^{2}} { \\
        \\left[T_{q_{v}}(t)\\right]^{\\frac{2}{3-q_{v}}}}}\\right]^{ \\
        \\frac{1}{q_{v}-1}+\\frac{D-1}{2}}}
    Where :math:`t` is the artificial time. This visiting distribution is used
    to generate a trial jump distance :math:`\\Delta x(t)` of variable
    :math:`x(t)` under artificial temperature :math:`T_{q_{v}}(t)`.
    From the starting point, after calling the visiting distribution
    function, the acceptance probability is computed as follows:
    .. math::
        p_{q_{a}} = \\min{\\{1,\\left[1-(1-q_{a}) \\beta \\Delta E \\right]^{ \\
        \\frac{1}{1-q_{a}}}\\}}
    Where :math:`q_{a}` is a acceptance parameter. For :math:`q_{a}<1`, zero
    acceptance probability is assigned to the cases where
    .. math::
        [1-(1-q_{a}) \\beta \\Delta E] < 0
    The artificial temperature :math:`T_{q_{v}}(t)` is decreased according to
    .. math::
        T_{q_{v}}(t) = T_{q_{v}}(1) \\frac{2^{q_{v}-1}-1}{\\left( \\
        1 + t\\right)^{q_{v}-1}-1}
    Where :math:`q_{v}` is the visiting parameter.
    .. versionadded:: 1.2.0
    References
    ----------
    .. [1] Tsallis C. Possible generalization of Boltzmann-Gibbs
        statistics. Journal of Statistical Physics, 52, 479-487 (1998).
    .. [2] Tsallis C, Stariolo DA. Generalized Simulated Annealing.
        Physica A, 233, 395-406 (1996).
    .. [3] Xiang Y, Sun DY, Fan W, Gong XG. Generalized Simulated
        Annealing Algorithm and Its Application to the Thomson Model.
        Physics Letters A, 233, 216-220 (1997).
    .. [4] Xiang Y, Gong XG. Efficiency of Generalized Simulated
        Annealing. Physical Review E, 62, 4473 (2000).
    .. [5] Xiang Y, Gubian S, Suomela B, Hoeng J. Generalized
        Simulated Annealing for Efficient Global Optimization: the GenSA
        Package for R. The R Journal, Volume 5/1 (2013).
    .. [6] Mullen, K. Continuous Global Optimization in R. Journal of
        Statistical Software, 60(6), 1 - 45, (2014).
        :doi:`10.18637/jss.v060.i06`
    Examples
    --------
    The following example is a 10-D problem, with many local minima.
    The function involved is called Rastrigin
    (https://en.wikipedia.org/wiki/Rastrigin_function)
    >>> from scipy.optimize import dual_annealing
    >>> func = lambda x: np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    >>> lw = [-5.12] * 10
    >>> up = [5.12] * 10
    >>> ret = dual_annealing(func, bounds=list(zip(lw, up)), seed=1234)
    >>> ret.x
    array([-4.26437714e-09, -3.91699361e-09, -1.86149218e-09, -3.97165720e-09,
           -6.29151648e-09, -6.53145322e-09, -3.93616815e-09, -6.55623025e-09,
           -6.05775280e-09, -5.00668935e-09]) # may vary
    >>> ret.fun
    0.000000
    """  # noqa: E501
    if x0 is not None and not len(x0) == len(bounds):
        raise ValueError('Bounds size does not match x0')

    lu = list(zip(*bounds))
    lower = np.array(lu[0])
    upper = np.array(lu[1])
    # Check that restart temperature ratio is correct
    if restart_temp_ratio <= 0. or restart_temp_ratio >= 1.:
        raise ValueError('Restart temperature ratio has to be in range (0, 1)')
    # Checking bounds are valid
    if (np.any(np.isinf(lower)) or np.any(np.isinf(upper)) or np.any(
            np.isnan(lower)) or np.any(np.isnan(upper))):
        raise ValueError('Some bounds values are inf values or nan values')
    # Checking that bounds are consistent
    if not np.all(lower < upper):
        raise ValueError('Bounds are not consistent min < max')
    # Checking that bounds are the same length
    if not len(lower) == len(upper):
        raise ValueError('Bounds do not have the same dimensions')

    # Wrapper for the objective function
    func_wrapper = ObjectiveFunWrapper(func, maxfun, *args)
    # Wrapper fot the minimizer
    minimizer_wrapper = LocalSearchWrapper(
        bounds, func_wrapper, **local_search_options)
    # Initialization of RandomState for reproducible runs if seed provided
    rand_state = check_random_state(seed)
    # Initialization of the energy state
    energy_state = EnergyState(lower, upper, callback)
    energy_state.reset(func_wrapper, rand_state, x0)
    # Minimum value of annealing temperature reached to perform
    # re-annealing
    temperature_restart = initial_temp * restart_temp_ratio
    print(("temperature_restart: {0}").format(temperature_restart))
    # VisitingDistribution instance
    visit_dist = VisitingDistribution(lower, upper, visit, rand_state)
    # Strategy chain instance
    strategy_chain = StrategyChain(accept, visit_dist, func_wrapper,
                                   minimizer_wrapper, rand_state, energy_state)
    need_to_stop = False
    iteration = 0
    message = []
    # OptimizeResult object to be returned
    optimize_res = OptimizeResult()
    optimize_res.success = True
    optimize_res.status = 0

    t1 = np.exp((visit - 1) * np.log(2.0)) - 1.0
    # Run the search loop
    while(not need_to_stop):
        print("--------------------------")
        for i in range(maxiter):
            # Compute temperature for this step
            s = float(i) + 2.0
            t2 = np.exp((visit - 1) * np.log(s)) - 1.0
            temperature = initial_temp * t1 / t2
            # print("temp: {0}".format(temperature))
            if iteration >= maxiter:
                message.append("Maximum number of iteration reached")
                need_to_stop = True
                break
            # Need a re-annealing process?
            if temperature < temperature_restart:
                energy_state.reset(func_wrapper, rand_state)
                break
            # starting strategy chain
            val = strategy_chain.run(i, temperature)
            if val is not None:
                message.append(val)
                need_to_stop = True
                optimize_res.success = False
                break
            # Possible local search at the end of the strategy chain
            if not no_local_search:
                val = strategy_chain.local_search()
                if val is not None:
                    message.append(val)
                    need_to_stop = True
                    optimize_res.success = False
                    break
            iteration += 1

    # Setting the OptimizeResult values
    optimize_res.x = energy_state.xbest
    optimize_res.fun = energy_state.ebest
    optimize_res.nit = iteration
    optimize_res.nfev = func_wrapper.nfev
    optimize_res.njev = func_wrapper.ngev
    optimize_res.nhev = func_wrapper.nhev
    optimize_res.message = message
    return optimize_res

def _ata(
    input_series,
    input_series_length,
    w,
    h,
    epsilon,
    gradient
):

    zfit = np.array([None] * input_series_length)
    xfit = np.array([0.0] * input_series_length)

    if len(w) < 2:
        raise Exception('This is the exception you expect to handle')
    else:
        p = w[0]
        q = w[1]

        p_gradient = gradient[0]
        q_gradient = gradient[1]

    # if q > p:
    #     return

    # fit model
    cc = []
    nt = []
    j = 1
    k = 1
    p_gradientlist = []
    q_gradientlist = []
    for i in range(0, input_series_length):
        nt.append(k)
        if input_series[i] == 0:
            k += 1
            if i == 0:
                zfit[i] = 0.0
                xfit[i] = 0.0
            else:
                zfit[i] = zfit[i-1]
                xfit[i] = xfit[i-1]
        else:
            a_demand = p / j
            a_interval = q / j

            if i <= p:
                zfit[i] = input_series[i]
            else:
                zfit[i] = a_demand * input_series[i] + \
                    (1 - a_demand) * zfit[i-1]

            if i == 0:
                xfit[i] = 0.0
            elif i <= q:
                xfit[i] = k
            else:
                xfit[i] = a_interval * k + (1-a_interval) * xfit[i-1]

            k = 1

        if xfit[i] == 0:
            cc.append(zfit[i])
        else:
            cc.append(zfit[i] / (xfit[i]+1e-7))
        j += 1

        # if i > 0 :
        #     aa = 0
        #     bb = 0
        #     if ((k - xfit[i-1]) * q + j * xfit[i-1]) != 0.0 and input_series[i] != 0 :
        #         p_gradientlist.append((input_series[i] - zfit[i-1]) / ((k - xfit[i-1]) * q + j * xfit[i-1]))
        #         aa = (input_series[i] - zfit[i-1]) / ((k - xfit[i-1]) * q + j * xfit[i-1])
        #     if ((k - xfit[i-1]) * q + j * xfit[i-1]) != 0.0 and input_series[i] != 0 :
        #         q_gradientlist.append(((input_series[i] - zfit[i-1]) * p + j * zfit[i-1]) * (xfit[i-1] - k) / (((k - xfit[i-1]) * q + j * xfit[i-1]) ** 2))
        #         bb = ((input_series[i] - zfit[i-1]) * p + j * zfit[i-1]) * (xfit[i-1] - k) / (((k - xfit[i-1]) * q + j * xfit[i-1]) ** 2)

        # print(("----p_gradient: {0}   q_gradient: {1}").format(aa, bb))

    # p_gradientlist.pop(0)
    # q_gradientlist.pop(0)
    # p_gradient = sum(p_gradientlist)
    # q_gradient = sum(q_gradientlist)
    # print(("p: {0} q: {1} last interval: {2}").format(p, q, xfit[-1]))

    ata_model = {
        'a_demand':             p,
        'a_interval':           q,
        'demand_series':        pd.Series(zfit),
        'interval_series':      pd.Series(xfit),
        'demand_process':       pd.Series(cc),
        'in_sample_nt':         pd.Series(nt),
        'last_interval':        xfit[-1],
        'gredient':             (p_gradient, q_gradient)
    }
    # print(zfit)
    # print(xfit[-1])
    # calculate in-sample demand rate
    frc_in = cc

    # forecast out_of_sample demand rate
    if h > 0:
        frc_out = []
        a_demand = p / input_series_length
        a_interval = q / input_series_length
        zfit_frcout = a_demand * input_series[-1] + (1-a_demand)*zfit[-1]
        xfit_frcout = a_interval * nt[-1] + (1-a_interval)*xfit[-1]

        for i in range(1, h+1):
            result = zfit_frcout / xfit_frcout
            frc_out.append(result)

        # print(('frc_out: {0}').format(frc_out))
        # f.write(str(zfit_frcout) +'\t' + str(xfit_frcout) + '\t' + str(result) + '\n')
    else:
        frc_out = None

    # f.close()
    return_dictionary = {
        'model':                    ata_model,
        'in_sample_forecast':       frc_in,
        'out_of_sample_forecast':   frc_out,
        'fit_output':               cc,
        'gredient':                 (p_gradient, q_gradient)
    }

    return return_dictionary

def _ata_cost(
                p0,
                input_series,
                input_series_length,
                epsilon
                ):

    p = p0[0]
    q = p0[1]
    if q > p:
        return 99999999999999.999
    
    # -------------------------------------------------------------
    ata_result = _ata(
                    input_series = input_series,
                    input_series_length = input_series_length,
                    w=p0,
                    h=0,
                    epsilon = epsilon,
                    gradient= (0, 0)
                    )
    frc_in = ata_result['in_sample_forecast']
    # print(frc_in)
    # (g_p_gradient, g_q_gradient) = ata_result['gredient']

    E = input_series - frc_in

    # standard RMSE
    E = E[E != np.array(None)]
    # E = np.sqrt(np.mean(E ** 2))
    E = np.mean(E ** 2)

    # print(("p_gradient: {0}   q_gradient: {1}").format(g_p_gradient, g_q_gradient))
    if len(p0) < 2:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], 0.0, E))
    else:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], p0[1], E))
    return E

# ts = [
#     -11617.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,-11617.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-11617.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6]

# ts = [362.35, 0.0, 0.0, 0.0, 0.0, 361.63, 0.0, 0.0, 0.0, 0.0, 362.77,
#       0.0, 0.0, 0.0, 0.0, 356.11, 0.0, 0.0, 0.0, 0.0, 0.0, 331.26,
#       0.0, 0.0, 0.0, 0.0, 304.87, 0.0, 0.0, 0.0, 0.0, 0.0, 311.06,
#       0.0, 0.0, 0.0, 0.0, 323.62, 0.0, 0.0, 0.0, 0.0, 336.57, 0.0,
#       0.0, 0.0, 0.0, 342.40, 0.0, 0.0, 0.0, 0.0, 343.57, 0.0, 0.0,
#       0.0, 0.0, 349.56, 0.0, 0.0, 0.0, 0.0, 356.45, 0.0, 0.0, 0.0,
#       0.0, 362.56, 0.0, 0.0, 0.0, 0.0, 360.64, 0.0, 0.0, 0.0, 0.0,
#       312.74, 0.0, 0.0, 0.0, 0.0]

ts = [1056,0,0,0,1092,0,0,1088,1443,0,0,1406,1361,0,1019,0,1296,1022,1388,0,0,0,1235,1236,1427,0,0,1295,1120,1424,0,0,0,0,1486,0,0,0,0,1138,0,1426,1243,1220,0,1373,0,0,0,0,1195,0,0,1289,1117,1159,0,1338,0,0,0,0,0,0,1267,1081,1157,1394,0,0,1462,1099,0,0,0,0,0,0,1278,1079,0,0,0,1319,1270,0,1386,1255,0,1252,0,1410,1004,1473,1220,0,1366,0,1324,0,1143,0,0,0,1086,1313,0,0,0,1002,1406,0,1341,0,0,1456,0,0,1249,1306,1496,1343,0,1068,0,1011,0,0,1429,0,0,1096,0,0,1361,0,0,1089,0,1241,1108,0,0,0,1319,0,1323,0,1012,0,1310,0,1250,0,1465,0,0,0,1133,1034,0,1342,1167,1453,0,0,1037,1095,0,0,0,1273,0,0,0,0,0,1081,0,1326,0,0,0,0,1357,1297,1263,0,0,1430,1033,0,1115,0,1359,1354,0,0,0,1277,1278,0,0,1173,1049,1194,1241,1128,1267,0,0,0,0,0,0,0,0,1034,0,0,0,1231,0,0,1093,1234,0,1049,1081,0,1403,1173,1093,1282,1373,1057,0,1185,1368,0,0,1263,0,1375,1095,0,0,1407,0,0,0,0,1172,1356,0,1028,0,1369,1007,0,1225,1230,0,0,1157,1262,0,1105,0,0,1131,1443,0,1123,1321,1440,1215,0,0,1416,1009,1370,1482,0,0,1101,1428,1299,0,1151,1180,0,1298,1115,0,0,0,1083,1253,0,0,1420,1331,0,1466,0,0,0,1249,0,1498,1370,1017,0,0,1363,0,1345,0,1119,0,1391,0,0,1369,0,1031,0,1465,0,1219,0,1357,1103,1476,1415,1466,1467,1491,1494,1471,1296,0,1088,0,0,1409,1370,0,1337,1207,0,0,1169,0,1477,0,1250,1361,0,1336,0,1233,0,0,1306,1170,0,1116,1423,0,1376,0,0,0,1141,0,1495,0,1096,1392,0,0,0,1400,0,0,0,1104,0,1221,0,1357,1306,0,1442,1369,1177,0,1146,1471,1488,1127,1223,1191,0,1016,0,0,0,1147,0,0,1392,0,1092,1486,0,0,1187,0,0,1184,1233,1050,0,1031,0,1088,1498,0,1158,1244,0,0,1254,1229,0,1231,1359,0,0,0,0,1436,1291,1262,1424,1109,1451,0,1490,0,1099,1212,0,1400,1260,1284,1336,0,0,0,0,1030,1453,1413,1145,1141,1216,1499,0,0,0,1037,1462,0,1027,0,1245,0,0,0,0,0,0,1371,1299,0,1176,1487,0,0,0,1313,1259,0,0,0,1166,1475,0,0,1108,1093,1210,0,0,0,1047,0,1084,1492,1346,0,0,1406,0,0,0,1290,0,1467,1109,0,0,1489,0,1003,1110,1384,1460,1035,0,1327,1216,1365,0,1335,0,1366,0,0,1163,0,0,0,1247,1372,1394,1264,1080,1306,0,0,1331,1329,0,0,0,1305,0,1074,0,0,0,0,1101,0,0,0,0,1472,0,0,1059,1288,0,1014,1177,0,1005,1416,1173,1269,1387,1153,0,1230,1396,0,1182,0,0,0,0,0,1291,0,0,0,1275,1251,1108,0,0,0,1167,0,0,0,0,1056,1498,0,1230,0,0,0,0,0,0,0,1312,0,1443,1370,0,0,0,0,0,0,0,1406,0,0,0,0,0,0,1107,0,0,0,1068,0,1022,1160,0,1176,1263,0,0,0,0,0,0,1388,0,1344,1462,0,0,0,0,0,1322,1195,1115,1188,0,1314,0,0,1134,1162,0,0,0,0,1242,1261,1309,1454,0,1028,0,1397,1110,1478,0,0,1156,0,1094,1144,1173,1088,0,0,1426,0,1424,0,0,1384,1010,0,1119,1442,1381,0,1253,0,0,1406,1211,0,1478,1043,0,0,0,0,1169,1140,1489,1476,1274,0,0,0,0,0,0,0,0,0,1491,1448,1064,1154,0,1341,0,1207,0,0,1451,0,1403,0,0,0,1048,1462,1149,0,1164,1386,0,0,0,1287,0,1381,0,0,1358,0,0,1029,1206,0,1074,1454,1446,1312,0,1108,1343,1328,1492,1289,1425,1359,0,1127,1414,0,1083,1474,0,1041,1211,1272,0,0,1115,0,1115,0,0,0,1003,1110,0,1023,0,0,1118,0,0,0,1293,1054,0,1355,0,1081,0,1398,0,0,1056,1173,1129,1129,1116,0,1072,0,0,0,0,0,1155,0,0,1372,1098,0,1143,1270,0,1481,1146,0,0,1217,1490,1163,1256,1259,1212,0,0,1009,0,0,0,0,1198,0,1418,0,1258,0,0,1308,1027,0,0,1401,0,1114,1318,1226,1177,0,0,1087,1253,1156,1453,0,0,1427,1008,0,0,1235,0,0,0,1233,1327,0,0,0,0,0,1431,0,1006,1315,1465,0,1270,1262,1472,1150,0,0,0,0,1036,1294,1251,1159,0,0,0,0,1286,1233,0,0,1315,1316,1142,1500,1005,1092,1093,0,1135,1240,1013,1087,1043,1165,1187,0,0,0,1328,1114,0,0,0,0,0,0,1024,1387,1367,0,1286,1185,1414,0,1008,1103,0,0,0,0,1248,1249,1145,1382,1316,0,0,0,0,0,1086,0,0,1151,0,1004,1338,1479,0,0,1095,1268,1241,0,0,0,0,0,0,1067,1329,1270,0,0,1302,1380,0,0,1241,1293,0,1070,1443,1447,0,0,0,0,0,0,1112,1485,0,1325,0,0,0,1273,0,0,1213,0,1340,0,1218,0,1278,0,1474,1059,1263,1373,1452,1121,1251,1128,1020,0,0,0,0,0,0,0,0,0,1067,1079,1013,1247,1282,1468,0,0,1043,1033,1042,0,0,0,0,1022,1285,0,0,0,1044,0,0,0,1003,1247,1293,1352,1326,0,0,0,1434,1396,1144,1149,0,1089,1311,0,0,1126,1092,1408,1033,1460,1444,0,1048,0,1382,1223,0,1357,1113,0,0,1453,1087,0,1205,0,0,1220,0,1086,1388,0,1280,0,1037,0,1204,0,1280,1217,1015,0,0,0,1500,1339,1377,0,0,0,1183,0,1499,1492,0,1479,0,1150,0,0,0,1347,0,1310,1182,0,0,0,0,1254,0,0,0,0,0,0,0,1151,0,0,1181,1245,1118,1080,0,1242,0,1309,0,1103,0,1223,1366,0,0,0,0,0,0,1407,0,0,1186,1448,1456,1011,1493,1054,0,1492,0,1335,1281,0,1032,1463,1153,1352,0,1341,0,1239,0,0,0,0,1061,0,1019,0,1056,1370,1178,0,0,1126,0,0,0,0,1123,1351,0,0,1258,1033,1274,1409,1426,0,1366,1193,1012,1022,1269,0,0,1298,0,0,0,1193,1292,0,0,0,0,1161,1378,1170,1204,0,0,1248,1319,0,1226,1436,1269,1082,0,0,1028,1205,0,1199,0,0,0,1287,1032,1252,1298,0,1244,1336,0,1141,0,1361,1451,1236,0,0,0,0,0,1248,0,1483,0,0,0,1123,1066,1476,0,0,0,0,1096,0,1054,0,0,1303,1277,0,0,1498,0,0,1334,1386,0,0,1404,0,1394,0,1496,0,1240,0,1362,0,0,0,1455,0,1048,0,1132,0,0,1011,1343,0,1113,0,1494,1293,0,1446,0,1101,1393,0,0,0,1427,1474,1304,1324,1474,1224,0,0,0,1154,0,1163,0,0,1354,0,0,1367,1436,0,1005,0,1247,0,0,0,1310,0,1411,1148,1215,1113,0,0,1496,1139,1393,0,0,0,0,0,0,1036,1041,1235,0,1297,1300,0,1247,0,1161,1485,1079,0,0,1310,0,1203,1092,1223,1335,1492,0,1168,1045,0,0,0,1141,1149,0,1462,1335,0,1434,1006,0,0,0,0,1469,1216,1375,0,0,0,0,1122,0,0,1342,1029,1128,1145,0,1449,1328,1315,1484,1239,1050,1247,1365,1299,0,0,1011,1497,1454,0,0,1401,1378,1121,1260,0,0,1093,1314,0,1390,0,1188,0,0,1185,1195,0,0,1303,0,0,1140,1289,1256,0,0,1486,1412,0,0,0,0,1410,0,0,0,1042,0,0,0,1125,0,0,1208,1025,0,1371,0,0,1005,1129,0,1288,1104,1149,1230,1099,0,1204,0,1086,1347,0,1050,0,1259,0,1409,0,0,1469,0,0,0,1397,0,0,0,0,0,0,1469,1253,1475,1095,0,0,1324,1078,0,0,1218,1269,1154,0,1044,1396,1323,0,1403,0,0,0,1231,1383,1044,1016,
1436,0,1415,1435,1263,0,1085,0,1321,0,1098,0,1179,0,0,1107,0,1161,0,0,1448,0,1409,1356,0,1418,0,0,0,0,1341,0,0,1160,1270,0,1204,0,0,1104,0,0,0,0,1154,0,0,0,0,1259,1498,1080,0,0,0,0,0,0,0,0,1196,1008,0,0,1321,0,0,0,0,0,0,0,0,1102,1459,1245,1312,1349,0,1220,1213,0,1428,1104,1354,0,0,0,0,0,0,1300,0,1176,0,0,1046,1050,1449,1291,1298,1246,0,0,0,0,0,0,0,0,1333,1275,1157,0,0,0,1325,1109,1269,1429,0,0,0,0,0,1371,0,0,0,0,1137,0,1108,1382,0,1340,0,0,1300,1109,1083,0,0,0,0,0,1060,0,0,1000,0,0,1344,1257,1444,1016,0,0,0,0,1026,0,1338,0,0,1140,1407,0,1490,1415,1122,1382,1469,0,1025,0,0,1167,1328,1126,1252,1011,1391,1080,1334,0,0,0,0,0,1057,0,0,1183,0,1108,1300,0,0,0,0,1379,1293,1078,1174,0,1385,0,0,1404,1440,1484,0,1214,0,0,1100,1224,0,0,1328,0,1197,0,1248,0,1343,0,1171,1283,0,0,1028,1347,1266,1282,0,0,1488,0,1303,0,1265,0,0,1493,1196,0,0,1117,0,1424,1282,1033,1207,1237,1348,0,0,0,0,1222,0,1033,0,1256,0,0,1073,0,0,0,0,1231,0,1001,0,0,0,1327,1075,1437,0,1133,1472,0,1267,0,1048,1165,1091,0,0,1291,0,1493,0,1408,1360,1153,0,0,1393,0,1017,1412,0,1010,0,0,1445,0,1208,0,0,1487,1042,1126,1008,1257,1019,1358,1337,0,1165,0,0,1262,1061,0,0,0,0,1340,1399,1445,0,0,0,0,1242,1295,1294,0,0,1174,1302,0,1080,1382,1141,0,0,0,1090,1127,1375,1478,1483,1271,0,0,0,0,1310,1261,1253,1241,0,1163,1261,1064,0,1074,0,0,0,1257,0,1102,0,1357,1098,1201,0,0,0,0,0,0,0,0,1206,1068,0,1217,0,0,1225,1263,0,0,0,0,1058,0,1077,0,0,0,1102,1163,0,0,1088,1060,0,1115,1214,1453,0,1189,1071,0,1342,0,0,1325,1201,1082,0,0,1133,1356,0,1142,0,0,0,0,0,1151,0,1415,1412,1490,0,1200,1145,0,1159,0,1406,1098,0,0,1187,1404,1001,1225,1318,0,0,1440,0,0,0,1409,1100,1199,1095,1201,0,1422,1174,1256,1442,1473,0,0,1020,1213,1054,1163,0,0,0,1340,0,1364,0,1057,0,0,0,1223,1316,0,1056,0,0,1126,1279,0,1421,1271,1449,1331,1192,0,1436,1098,1389,0,0,1247,0,1188,1169,0,1101,1404,1200,1224,1225,0,1078,1270,0,1057,1189,1299,0,0,0,1409,1249,1076,0,1472,0,1380,1118,1324,1373,1457,0,0,0,0,1313,0,1024,0,0,1172,1488,1458,1402,1171,0,0,0,1039,1046,0,1360,0,1292,0,1302,1420,0,1243,1312,1313,0,0,0,1126,1222,0,1045,0,0,0,1309,1276,1419,0,0,0,0,1187,0,0,0,1246,0,1324,0,1482,1439,1252,0,0,1313,1079,1072,0,0,1348,1238,0,1470,1124,1429,1228,0,0,1286,0,0,0,0,0,1499,0,1204,1226,1254,0,1392,0,0,0,0,1341,0,0,1345,0,1382,0,0,0,0,1395,1160,0,1497,1340,1490,1380,0,1178,0,0,0,0,1361,1141,0,0,0,0,1375,0,1420,0,1025,1028,0,1086,0,0,0,1188,0,1379,0,0,1268,1223,0,0,1097,0,1343,1344,0,1012,0,1203,0,1025,1400,1393,1025,0,1120,0,1009,1135,1060,1190,0,1340,0,1000,0,1260,1413,1079,1162,0,1274,0,1061,0,1324,0,0,1006,1328,0,1002,0,0,1156,0,0,1282,1032,1249,0,1082,1214,0,1081,1290,1335,1082,0,0,0,1044,1339,0,1178,0,1367,1024,1359,1157,1291,0,1215,1419,1015,0,1368,1343,1395,0,0,1139,0,1161,1033,1032,1083,0,1051,0,1309,1474,0,1182,1341,1095,1338,0,1322,1154,0,1179,0,0,0,0,0,0,0,0,1500,1279,0,0,0,1182,0,1132,0,1489,1304,1478,1190,1077,0,1202,1355,1121,1235,0,1448,1256,0,1265,1231,0,0,1396,0,0,0,0,0,0,1410,1281,0,0,0,1078,0,0,0,0,1139,0,0,1250,1272,0,1099,0,1377,0,0,0,0,0,0,1094,0,1042,1087,1125,1309,0,0,0,0,0,1223,0,0,0,0,1406,0,0,0,0,0,0,1364,1015,1487,0,1339,1218,0,0,1031,1026,0,0,0,0,0,0,1445,1267,1187,1448,1210,0,0,1445,1364,0,1174,1046,1376,0,0,0,1187,0,1215,0,1117,0,1175,1473,1288,1450,0,0,1281,1139,0,1233,1164,0,0,0,0,1083,1436,1173,0,0,0,1497,1225,0,1146,1466,1095,0,1410,0,0,0,0,1279,0,1180,0,1246,0,0,0,0,1212,0,1006,1115,1413,0,1283,0,1213,0,0,0,0,0,1280,1052,1081,1097,1162,1146,0,0,0,0,0,1356,1393,0,1204,0,1311,0,1314,0,1029,0,0,0,1383,1500,0,0,1150,0,1301,1057,0,1447,0,0,1137,0,0,0,0,0,0,0,1378,1220,1210,1223,1015,1080,0,0,0,1467,0,1447,1176,1292,0,0,0,1018,0,1010,1250,0,0,1424,0,0,1370,1337,1309,0,1405,1212,1248,1164,0,0,1332,0,0,1274,0,1424,1421,0,0,1086,0,1297,1052,1082,0,0,1358,0,1191,0,0,1411,1050,1107,0,0,1413,1126,0,0,0,1247,1350,0,1427,1495,1369,1291,1115,0,0,0,0,1395,1115,0,0,0,1268,1467,0,0,0,0,0,1402,1310,0,1472,1282,0,0,0,0,1209,1284,1056,0,1411,0,0,0,1464,0,1354,0,0,1071,1162,1180,1425,0,0,0,0,0,1089,0,0,1280,0,0,0,1099,0,0,1022,0,1090,0,1432,0,1175,1163,1443,0,1165,0,1290,1145,1423,1175,1158,1310,1320,1084,1423,1205,0,1105,0,0,0,0,1473,0,1205,1497,1180,1331,1046,0,1113,0,0,1237,0,0,1173,0,0,1326,0,1435,0,1101,1027,1100,0,1194,0,1103,1038,0,1429,1146,0,0,1048,1083,1208,1264,0,1410,0,0,0,0,1429,1287,1423,0,
1223,1488,0,1224,0,1496,1043,1270,1141,1320,1323,0,1143,1040,1357,1412,1247,0,1151,1339,1199,0,0,0,0,1499,1329,0,1322,0,0,0,1452,1081,0,0,1353,0,1005,0,1232,1368,1136,1269,1163,0,0,1466,1178,1285,0,0,1420,1205,1080,0,1406,0,1435,0,1362,1404,0,0,0,0,1138,1305,1208,1246,1487,1434,1385,0,1242,1309,0,0,1209,0,1352,0,0,1090,1111,0,1495,0,0,0,1417,1252,0,0,1458,0,0,1280,0,0,0,0,0,1201,0,0,0,0,1197,0,1140,1444,1350,1199,0,1447,0,0,1282,1017,0,1465,0,1214,0,0,1452,0,0,0,1416,0,1446,1244,0,1310,0,1115,0,1266,1459,1007,1201,1260,0,1221,1012,1382,0,1114,1011,0,0,1154,1016,1316,1354,0,1040,0,0,0,1055,0,0,0,1372,0,0,1098,1282,0,1140,1442,1106,0,0,0,1331,1165,1451,0,0,0,0,1269,1466,1492,0,1156,0,1021,1129,0,0,1414,1067,1321,0,1138,0,1434,0,0,0,0,1337,1493,0,1094,0,1305,1067,1177,0,0,1196,1354,0,1035,1111,0,1453,1252,1050,1446,0,1035,1074,1204,1260,0,1278,1261,0,0,1028,1453,0,1318,0,0,1165,0,1005,0,0,1240,0,1397,1437,0,1259,1018,0,0,0,1193,0,1051,0,1485,1167,0,0,1052,1481,0,0,0,0,0,0,1366,0,0,1258,1483,1343,0,0,0,0,1426,0,1052,1006,0,1439,0,1462,0,0,0,0,0,1439,1024,1379,1041,0,0,0,0,0,1321,1236,0,0,0,1371,1452,0,0,0,1166,0,0,0,1017,1278,0,0,1138,1337,0,1446,1180,1130,1495,0,1286,0,1358,0,0,0,0,1042,0,0,1085,0,1370,0,1381,0,0,0,0,0,0,0,0,1338,0,1043,1188,1426,0,0,1425,0,1410,0,0,0,0,1482,1211,0,0,1332,0,0,1272,1358,0,1353,0,0,0,1127,0,0,1473,0,1039,0,1092,1234,0,0,0,1213,0,0,0,1250,1295,0,0,1029,1316,1396,0,1489,1303,0,1295,1302,0,1189,1268,0,1367,1118,0,0,0,0,1173,1400,1474,0,1355,0,1034,0,1007,0,1069,1142,1283,0,0,1472,0,1190,0,1334,1349,0,0,0,0,1379,1418,1449,1187,0,0,1433,1426,0,1321,1249,0,0,0,0,0,1314,1460,0,1098,0,1149,1425,1169,0,0,0,1002,1296,1339,1481,0,0,1005,1013,1090,0,0,1338,1339,1049,1250,0,0,1337,1247,1067,0,1017,1151,1162,1210,1353,1224,1454,1026,1179,1282,1174,0,1013,0,1405,1118,0,0,1425,1403,1490,0,1411,0,1395,0,1346,1275,0,0,1399,0,0,0,1228,1426,1188,1303,1259,1060,0,1331,1423,1103,0,0,1426,0,1473,0,1410,1411,1449,0,0,0,0,1160,0,0,0,1369,0,0,1140,1229,0,0,1184,1380,0,0,1310,0,1205,0,1112,0,1252,1423,0,1141,1210,0,1481,1007,0,1313,0,1329,0,0,1096,1462,1146,0,0,1252,0,0,1369,1019,1367,0,0,1268,1181,1223,1132,0,1102,0,0,1277,0,0,1361,0,1360,1038,0,0,1234,1213,1286,1446,1112,0,1139,0,1379,0,0,0,1012,0,0,1013,0,0,1314,1000,0,1213,0,1265,0,0,0,1290,1204,1379,1286,1257,1069,1320,0,0,0,1100,1205,1246,0,1472,0,0,0,0,1326,1226,1363,0,0,1132,1305,1424,0,0,0,1118,0,0,0,1237,1417,0,0,1038,1064,1251,0,1007,0,1430,1279,0,0,0,0,0,1216,0,1112,1394,0,1469,1416,0,1191,1194,1093,0,1180,0,1120,1129,0,0,0,0,1275,0,1288,1395,0,1207,1125,1192,1088,1133,0,1358,1215,1419,0,1260,1100,0,0,1265,0,1388,0,0,1429,1078,0,1381,1016,1105,0,0,0,1000,0,1456,0,0,1129,0,0,1058,0,1016,0,1263,1285,1343,1058,1141,1316,0,1229,1321,0,1445,0,1327,0,1016,0,0,1406,0,1498,0,1378,0,1199,0,1272,1046,1166,0,0,1391,0,1091,1205,0,1199,1369,1242,1342,1103,0,0,0,0,1363,0,0,0,1249,0,1010,1140,1176,0,1251,1423,0,0,0,1114,0,0,1091,0,1278,1354,1276,1213,0,1303,1153,1386,0,1462,0,1267,1101,1456,1112,1152,0,1264,0,1018,1415,0,1338,1342,1018,0,1339,0,0,0,0,1277,1330,0,0,0,1403,0,1011,1045,1460,1468,1420,1297,0,0,1483,1409,0,0,1489,1153,1478,0,1259,0,0,0,1189,0,0,1479,1148,1080,0,0,0,1058,1298,1301,1255,1475,1171,1131,1001,0,1256,0,1462,1331,1181,1192,0,1340,0,1081,1294,1255,0,1239,1459,0,1475,1071,1331,0,0,0,0,0,0,1396,0,1408,1080,1332,1133,0,1134,1076,1411,1313,1454,0,0,1024,0,1190,1025,0,0,1147,1450,1376,1107,1385,1358,1134,0,1063,1067,1478,0,1361,1024,0,0,1270,0,0,1224,0,1488,0,1172,1468,1304,0,1392,1428,0,1338,1271,1146,0,0,1280,0,0,1036,0,0,1205,1365,0,0,1076,1468,1304,0,1377,1449,0,0,1383,1475,1451,0,1119,0,1400,0,1014,1473,1046,0,1197,0,1130,1130,1294,1380,0,0,1059,0,0,1187,1466,1249,1027,1316,1397,1409,1456,0,1446,0,1191,1118,1301,0,0,1492,1500,0,0,1480,0,0,0,0,0,0,1281,1099,1001,0,0,0,1479,1386,1234,0,1106,1128,0,0,1019,1011,0,1346,1349,0,0,1093,0,1202,0,0,0,1191,0,0,0,0,0,0,0,0,0,0,0,1008,0,0,1018,1089,0,1474,0,0,0,1388,0,0,0,0,0,1382,1343,0,1067,0,1404,0,1350,1116,0,0,0,0,1381,1298,1066,1361,1041,1140,1499,0,1386,0,1272,0,1189,0,0,1402,0,1128,1471,0,1181,1356,1432,0,1400,0,1068,0,1129,0,1283,0,1385,1130,1116,1416,1348,1497,0,0,1358,1497,1197,0,1397,0,0,0,0,1466,1483,1388,0,1099,1032,1148,0,0,1295,1442,1146,1157,1145,0,0,1122,1085,0,1106,0,1231,1247,1311,1472,1150,0,0,1412,1095,0,0,0,0,0,0,1241,0,1465,0,0,0,0,0,0,1290,1386,0,1351,1464,1230,1433,1107,0,1314,1126,0,0,1098,1050,1327,0,0,0,0,1195,0,1493,1149,0,0,1299,0,1494,0,0,0,0,0,0,1154,1394,0,1060,1204,1487,1123,0,1075,0,1361,1413,1139,0,0,1194,0,0,0,0,1266,0,1432,1352,1131,0,1000,0,0,0,0,1228,0,0,0,1302,0,0,1288,0,1058,1013,0,0,1141,0,0,0,0,0,0,1302,1328,1340,0,0,0,0,1063,1098,0,1212,1044,0,1045,0,0,1478,1480,0,1469,1050,1094,1476,1111,0,0,1255,0,1485,1315,0,0,0,0,1044,0,1294,1029,1305,0,1451,0,0,1290,1474,0,1328,1333,0,0,1248,0,1483,1480,0,1210,0,1148,1072,0,0,1127,1357,0,1332,1192,0,1447,1171,1488,0,0,1099,0,1022,1107,0,0,0,1262,0,1103,0,1118,0,1426,0,1107,1416,0,1419,0,0,0,1443,1134,0,1254,1412,0,0,1140,0,1075,1034,0,1359,1298,0,0,1117,0,0,1385,0,0,0,0,1352,1310,1192,1066,0,1174,1388,1098,0,1110,0,1313,1025,1060,0,1283,0,0,1261,0,1374,0,1160,1495,0,1320,0,1273,0,1160,1023,1373,1258,1397,1484,1500,1027,0,1135,1491,0,0,1315,1490,0,1300,1228,1491,0,0,1391,1021,0,1234,0,1500,1206,1301,0,1352,0,1155,1357,1216,0,1415,1449,0,1481,0,0,0,0,1290,0,1049,1298,0,1496,0,1272,0,1479,1034,0,0,0,0,1357,1217,0,1124,0,0,1421,1128,0,1423,1079,0,0,1186,1045,0,0,0,1104,1472,1178,1137,0,0,0,1249,1369,0,1433,1015,1259,1061,1157,1081,0,1238,1161,0,0,0,1166,0,1033,0,1451,
1214,1290,0,1105,1078,0,0,1153,0,1225,0,1398,0,0,0,0,1005,1000,0,0,0,0,1022,1475,0,0,0,0,1030,1296,0,0,0,0,0,1118,0,0,0,0,1290,1472,1188,1314,0,1260,0,0,0,0,0,1280,1499,1235,0,0,0,1363,1163,1145,0,1483,0,0,1408,0,0,0,1153,0,0,1350,1032,0,1366,1461,1442,0,1263,1239,1401,0,1192,1104,0,0,1248,1063,1366,0,1172,0,0,0,0,0,0,0,0,1445,1336,1431,0,1140,1393,1371,1290,1053,0,1118,0,0,1266,0,0,1092,0,0,1394,1025,0,1275,0,0,1014,1381,0,1443,0,0,1258,1422,1263,0,0,1093,1042,0,1315,0,1320,0,1199,0,0,0,1275,0,1367,0,1139,0,0,0,1104,0,1451,1267,1087,1256,1295,0,0,1495,1411,0,0,1116,1000,0,1011,0,1464,0,0,0,1124,0,1372,0,1257,1052,0,1360,1061,1271,1386,1075,0,0,1017,1144,1229,1020,1079,1024,0,0,1472,1071,1325,1113,0,1126,1088,0,0,0,0,1032,1211,0,0,0,1219,1418,0,0,1137,1368,0,0,0,1125,1093,1030,0,0,0,0,1146,1315,1089,1213,0,0,0,0,0,1405,0,1132,0,1275,0,0,1166,1043,0,1135,1117,0,0,1394,0,1461,1105,1129,1338,1385,0,1193,1103,1474,0,0,0,1123,0,1287,1000,0,0,1484,1042,0,0,0,0,0,0,1195,0,1075,1426,1493,0,1142,0,0,0,1085,0,0,0,1298,1438,0,0,0,0,0,1062,0,1366,0,0,0,0,0,1212,0,0,1410,0,1280,1091,0,0,0,0,1158,0,0,0,1204,1055,1292,1146,1179,1340,1423,1257,1031,0,1231,1061,0,1350,0,0,0,0,0,1043,1069,0,0,1342,0,0,1175,1367,0,1173,1177,1164,0,0,0,1464,1445,1354,0,0,0,1152,1407,0,1331,1209,1309,1032,1038,1358,0,1393,0,0,1442,0,0,1116,0,1007,0,0,1180,1121,0,1332,0,1336,0,1470,0,0,0,1498,1362,0,1190,1491,0,1483,1492,0,0,1033,1435,1289,0,0,1176,1484,1038,1458,1057,0,1236,0,0,1278,1470,1176,1471,0,0,0,0,1215,1341,1038,1227,0,0,0,0,1181,0,0,1418,0,1362,0,1277,0,0,0,0,0,0,0,0,0,1095,1500,0,1196,1315,0,0,0,1030,0,0,1240,0,0,1039,0,0,1098,1179,0,0,1141,0,1116,1472,1384,1038,0,1285,1423,0,1468,0,1055,0,0,1370,1089,1312,0,0,1490,1456,0,1455,0,0,0,0,1303,1461,1247,1398,0,0,1219,1121,1234,0,1458,1469,0,0,0,1184,0,1479,0,0,0,0,0,1321,1259,0,0,1418,0,1164,0,0,0,1241,0,0,0,0,0,1423,0,0,0,0,1378,1076,0,0,0,1318,1280,1169,0,0,1109,1174,0,1339,1178,1289,0,0,0,1017,0,1115,0,0,1061,1362,1324,0,0,1405,0,0,0,0,0,1249,0,1312,0,0,1134,1463,1071,0,0,0,1240,1351,1394,1171,1057,0,0,1341,0,0,0,1457,1388,0,1112,1191,1120,1031,1364,1462,0,1147,1044,1394,0,0,0,0,1055,1492,0,0,0,1077,1091,0,1295,1310,1193,1432,0,0,0,1481,0,0,1093,0,0,0,0,0,1003,0,0,1065,0,1485,1102,0,0,1035,0,0,0,1246,1460,1145,0,1212,0,1479,1055,1229,1410,1469,1341,0,1297,0,1345,0,1444,0,0,1487,1152,1107,1100,0,1156,1291,1006,0,0,1388,1452,0,1335,0,0,1266,1488,1133,0,0,0,0,1218,1004,0,0,1385,0,1322,1218,1230,0,0,0,1225,0,1491,0,0,1171,1112,1156,0,1335,0,0,1001,1221,1421,1450,0,0,0,0,0,1377,1087,1121,0,0,1158,0,0,1362,1128,0,1082,0,0,0,0,0,0,1101,1365,0,1168,0,1059,1439,1419,1373,1035,1487,0,0,1125,0,0,0,0,0,1006,0,0,1071,1482,0,1127,1133,0,0,1393,1357,0,1291,1231,1294,0,1202,0,0,1063,1443,0,1250,1261,0,0,1314,1101,1131,1316,1336,0,1052,0,1158,0,1224,1498,0,0,0,0,0,0,1482,0,1061,0,1120,0,1322,0,1478,0,0,1447,1330,0,1442,0,1286,0,1259,0,1307,1470,0,1300,0,1332,0,1103,0,1432,1244,1142,0,0,0,0,1142,1020,1120,1431,1003,1215,1394,0,0,1098,1352,1224,0,1450,1469,0,1297,1446,1258,0,0,1353,0,1316,1309,0,1225,1320,1122,0,1357,0,0,1438,1119,0,0,0,1209,0,1204,1326,1134,1479,1370,0,0,1261,1229,1082,0,0,0,1279,1095,0,0,1306,1156,0,1016,1179,0,0,0,0,0,0,0,0,1161,1279,1480,1193,1134,0,1107,0,1431,1173,0,1254,0,1244,1024,1268,1454,1028,1439,0,0,1403,0,0,1424,0,1288,0,1462,0,1360,1303,1325,1211,0,1187,0,1392,1111,0,1294,0,1367,1047,1133,1029,0,1080,1240,1306,1180,0,0,0,0,0,1074,1338,1490,1398,0,1226,1039,0,0,0,0,1326,1483,0,0,0,0,0,0,1000,1491,1013,0,0,0,1327,1297,0,1243,0,1385,1022,0,1020,1034,0,0,0,1332,0,1200,1253,1351,1175,1084,1311,0,0,0,0,0,0,1476,0,0,1045,0,0,1437,0,0,0,0,1002,0,1319,1469,0,1336,0,0,0,0,0,1180,1219,0,1486,1395,1286,0,0,1020,1246,0,1048,0,1133,1347,1029,1381,1055,0,0,0,0,0,1254,1428,0,0,0,0,0,1077,0,0,0,1355,1341,1372,1033,1315,1087,0,0,0,0,0,1292,1467,0,1023,0,0,1225,0,1109,1043,1373,1453,1124,1380,0,1176,1391,0,0,0,0,1076,1216,0,1211,0,1284,1309,0,1498,1403,0,0,0,1118,0,1231,1089,1145,1465,0,0,0,1112,0,1086,0,1167,1101,0,0,0,0,1320,1140,0,0,0,0,1029,1348,0,1394,1309,1131,0,1094,0,1037,0,1101,1059,1169,0,1043,1496,0,1452,1378,1117,1226,0,1206,0,0,0,1398,0,1480,0,0,0,1178,1297,1138,0,1424,0,0,1490,0,0,1045,1479,0,1104,0,0,0,0,1166,0,1440,1103,0,1174,1284,1128,1277,1261,1006,0,0,0,0,0,1158,1243,1146,1109,1172,0,0,1074,1381,1048,0,1460,0,0,1025,1108,1138,0,0,0,1470,0,1354,0,0,0,0,1071,
0,0,1363,0,1274,1187,1371,0,1417,1490,1291,0,1140,1456,1252,0,0,0,1159,0,1387,0,0,1311,0,1254,0,1318,1314,0,0,1204,0,0,1499,0,1248,0,0,1106,1413,1440,1392,1371,1334,0,0,1356,1135,1446,1440,0,0,1072,0,0,0,1156,1027,0,0,1311,1146,0,0,1278,1316,0,1070,0,0,1372,0,0,0,0,0,0,0,1377,1342,1085,1122,1212,1130,0,0,1302,0,0,1399,0,0,1099,0,0,1241,1347,0,0,1091,1062,0,0,0,0,0,0,1187,1278,0,0,0,1231,0,1472,0,1019,0,1234,1463,1450,1326,1488,1073,1467,0,1376,1279,1420,1244,1157,0,0,1034,0,1054,0,1500,0,1361,0,0,0,0,1000,1194,1042,1483,1416,0,1015,0,1035,0,0,1049,1100,1407,1392,1362,0,1127,0,0,0,1156,1121,0,1303,1086,0,0,1186,0,1389,0,1196,1445,1066,0,0,1176,0,1488,0,1032,1470,1464,0,0,1479,0,1269,1304,0,0,0,0,0,0,0,1230,1338,0,1490,0,0,0,1204,0,1125,0,1226,1494,0,0,0,1498,1491,0,1367,0,0,1328,1168,1380,1166,1198,0,0,1280,1156,1261,0,1161,0,0,1184,1268,0,0,0,0,1168,1182,0,0,0,0,1415,1094,1437,0,0,1448,1278,0,1422,0,1015,0,0,1005,0,1288,1153,1391,1293,1168,0,0,1123,0,1197,0,0,0,1138,1183,1017,0,1471,1064,0,1442,0,1364,0,0,0,1458,1333,0,1067,1390,1484,1225,0,1342,0,0,1055,0,1485,0,1421,0,0,1500,0,0,0,0,1219,0,0,1211,0,1176,0,0,0,0,1047,0,0,1015,0,1397,0,1310,1127,0,0,0,1413,1207,0,1025,1316,0,1005,1324,1031,1280,0,0,1116,0,0,0,0,0,1072,1376,1452,0,0,1309,1421,1181,0,0,1366,0,1102,1180,0,1384,0,1373,1329,0,0,0,0,0,0,0,0,1423,0,0,1240,1070,0,0,0,0,1280,1102,0,1487,1254,1170,0,1147,1268,0,0,1423,0,0,0,0,0,1498,0,0,1460,1305,1500,0,0,1058,0,1131,1376,0,0,1232,0,1055,1053,0,0,1236,0,1238,1205,1233,0,1494,1191,0,1335,1372,1383,0,1300,0,1245,1435,0,0,0,1243,1174,1427,1031,1367,0,0,0,1197,1103,0,1446,0,1284,1324,1247,0,0,0,1334,1431,1258,1097,1393,0,0,0,1030,0,1470,1021,1159,1140,1249,1005,1328,0,1387,1180,1261,0,1365,0,0,0,1113,1202,0,0,1458,1121,0,1380,0,0,0,1320,0,1180,1261,1309,0,0,0,1241,0,0,0,1273,0,0,0,1142,1180,1182,1183,0,1138,1253,0,1279,0,0,0,1346,1098,0,1360,1433,0,0,1278,0,1094,1399,0,1001,1040,0,1447,1048,1066,1183,1427,0,0,1448,1375,1491,1444,0,0,1009,0,0,0,1032,1168,1205,1154,1072,1310,0,0,1205,0,0,1121,0,1398,0,1011,1225,1435,0,1109,1078,0,1046,1004,0,1412,0,1371,1003,0,0,0,0,0,0,0,1421,0,0,1118,1070,1061,0,1464,0,1306,0,0,0,1016,0,0,1314,1367,0,1248,0,1389,1069,1266,0,1435,1226,0,1291,1483,0,1032,0,1380,1305,0,0,1068,1243,0,0,0,1065,1040,1055,1199,0,0,1027,0,1425,0,1336,1038,1187,1387,0,0,0,1163,0,0,0,0,0,1306,1461,0,0,0,1091,0,0,1076,1052,0,1274,1427,0,1145,1379,1164,1433,0,0,0,0,0,0,0,1147,1019,1125,1495,1112,1201,1192,0,0,0,1175,1351,0,0,0,1344,1165,0,1375,0,1462,0,1436,0,0,1116,0,0,0,1099,0,0,0,0,1333,0,0,0,0,0,1231,1261,0,0,1002,0,0,0,0,0,0,0,1433,1134,0,1485,1300,0,1143,1243,0,0,0,0,0,0,0,0,1275,1404,0,0,1459,1109,0,0,1016,1081,0,1470,1282,0,1309,1404,0,1115,1282,1342,1332,1144,0,1232,1447,0,1500,1436,0,1189,1085,1002,1229,1449,1226,0,1347,1121,0,0,0,0,1141,0,1158,0,1372,1095,1242,1283,1361,0,1043,0,1314,1126,1061,1133,1331,0,0,1462,1116,0,1048,0,1027,1133,0,0,0,0,0,0,0,0,0,1248,1193,1137,1305,1264,1031,1209,0,0,1407,0,0,0,0,1431,1437,1336,1440,1013,0,0,0,1399,1129,1135,0,0,0,1400,0,1082,0,1231,1275,1414,0,0,1268,0,1024,1232,1273,1233,1499,0,1054,1330,0,0,0,0,0,1372,0,1304,1305,1471,0,1074,1359,0,0,0,0,1120,0,0,1342,1254,1424,0,0,0,1306,0,0,1363,0,0,1407,0,1266,0,1116,0,1383,1375,1472,0,0,1425,1379,1039,0,0,1226,0,1065,0,1401,0,0,1269,0,1299,1462,0,1371,0,1253,1328,0,0,1023,0,0,0,1442,0,0,0,1004,0,1049,0,0,0,1446,0,0,1000,1005,0,1071,1028,0,1218,0,1435,1489,1295,1135,1285,1406,1119,1442,1404,0,0,0,1469,0,1148,1448,0,1473,0,0,0,0,0,1018,0,1362,1350,1175,1460,1021,0,1387,1186,0,1333,0,1221,0,0,1121,1391,0,0,0,0,0,1103,0,0,1329,1387,1194,1207,1427,1274,0,0,1000,0,0,0,1204,1227,0,1365,1178,1320,1470,1445,0,0,0,0,1245,0,0,1295,1460,1036,1482,1498,1087,1281,0,1058,0,1119,1219,0,1045,1215,0,1146,0,1049,1410,0,1092,1340,0,0,1263,0,0,1333,1249,1164,1350,0,1420,1045,1154,1161,1386,1389,1474,0,1046,1100,1405,1466,1110,1120,1391,1213,0,0,1282,0,1098,1069,0,1468,1441,1474,0,1130,0,1302,0,0,0,0,1472,0,1024,0,0,1067,0,1499,1149,0,1183,0,1121,1494,1306,1130,1123,1462,1256,0,1232,1085,1034,1032,1221,0,0,0,0,1341,1366,1376,1076,1359,0,0,1039,0,0,0,1465,0,0,1391,1166,1179,0,0,0,0,1393,0,1017,0,1007,0,0,0,0,1375,1293,1437,0,1388,1245,0,1106,1236,0,1139,1459,1055,1212,1004,1323,0,0,1302,0,0,0,0,1419,0,0,1067,1481,1114,1225,0,1111,0,1197,0,0,1120,0,1325,1008,0,1139,1063,1396,0,1315,1116,0,1275,0,1330,1271,1396,0,0,1331,1393,0,1444,0,1083,0,1315,0,0,1432,1176,1422,0,1037,1081,0,0,1188,0,1446,1065,1219,0,0,1344,0,0,1173,1453,0,1345,0,0,0,0,1002,0,0,0,1383,1319,0,1132,0,1105,1141,1096,0,0,0,0,0,1274,0,0,1391,0,0,0,1298,1423,0,1018,1355,0,0,0,0,1127,0,0,0,1408,1485,0,1206,1075,1149,1346,0,0,1189,1049,0,1410,0,0,0,0,1437,0,1136,1205,1097,0,1297,0,1425,0,1320,1110,1331,0,1115,1262,0,1078,0,0,0,0,1121,0,0,0,0,0,0,0,0,0,1214,0,1385,0,0,0,0,0,0,0,1302,1457,1314,1122,0,0,1274,0,1184,1170,1132,1341,1234,0,1043,0,0,0,1319,0,0,1028,0,0,1064,0,0,0,0,1032,0,1484,0,0,0,0,1267,0,1050,1145,0,1306,0,0,1141,0,0,1483,0,0,1496,1028,0,0,1273,0,0,0,1236,0,0,1295,0,1128,1333,0,1447,1341,0,0,1315,0,1051,0,0,1389,0,0,0,1261,1065,1132,0,1236,1319,0,1059,0,0,0,0,1205,0,0,0,0,0,0,0,0,1315,1193,0,1039,0,0,0,0,0,0,1340,1391,1327,1045,0,1013,0,0,1004,1204,1120,0,1347,1445,0,1414,1283,1327,0,0,1135,1293,1213,1091,0,0,1187,0,0,0,1299,1431,1330,0,1126,0,1268,1376,1398,1293,1399,0,1065,0,0,0,0,0,1241,1139,1103,1261,0,1363,1480,1037,0,1162,1195,1491,1232,1324,1074,1442,1306,1388,1207,1315,0,1486,0,1195,1042,0,0,0,1194,0,1237,0,0,1168,1215,1297,0,1173,0,1091,1072,1289,1330,1282,0,1381,0,1194,1116,1033,0,1374,0,0,0,1329,
0,0,0,1227,1250,0,0,0,0,0,0,1475,1493,1310,0,0,0,1310,0,0,0,1421,0,0,0,1021,1180,0,0,0,1454,1391,0,0,1099,0,0,1460,1213,0,1396,0,0,0,0,1028,1445,0,0,1261,1009,0,0,0,1422,0,0,1134,1433,0,1081,1159,1029,1009,0,0,1335,0,1410,0,0,0,1178,1428,1022,0,1051,1264,1313,1483,0,1480,0,1272,0,1116,1495,0,0,1143,0,1491,0,1228,1271,0,0,1428,1146,0,0,1311,0,1401,1144,0,1431,0,1409,0,0,1213,1075,0,1392,0,0,0,1187,0,0,0,1455,1102,1198,0,1149,0,1291,0,1496,1478,0,1041,1470,0,1232,0,1031,1087,0,1173,0,1161,0,0,0,0,0,1188,0,1359,1473,1426,0,0,0,0,1402,0,1425,0,0,0,0,0,0,1174,0,1189,0,0,0,1072,1170,0,1126,1129,0,0,1464,0,1158,0,1402,0,0,0,0,0,0,0,1471,1181,0,0,1099,0,0,1349,1342,0,1463,0,1384,1343,0,0,0,0,1250,0,0,1425,1048,1035,1132,1465,0,1378,1059,1169,1162,0,0,0,0,1457,0,1341,1467,1380,1312,0,0,1022,0,0,0,0,0,1194,1363,1195,1115,0,0,1103,1402,0,1171,1267,1273,0,0,0,0,1285,0,0,1195,0,0,0,1475,0,1008,0,0,0,1211,1493,1033,0,0,1058,0,1299,1092,0,1049,1183,0,1450,0,1206,1067,1330,1197,0,1463,0,0,0,1434,0,1375,0,0,0,1114,0,1406,1243,0,0,0,0,1027,0,1434,1234,1033,1100,0,0,0,1118,0,1166,1122,1082,1460,1191,1147,0,1284,1209,1244,0,1159,1090,1180,0,1171,1077,1033,1045,0,1075,1129,1312,1410,0,1470,0,0,0,0,1399,0,0,1297,1426,1209,1072,1011,0,1151,0,0,0,0,0,0,1337,0,0,1052,0,0,0,0,1396,1407,1188,0,1000,0,0,0,1107,1408,0,1290,1494,1077,0,1291,0,1088,0,1427,0,0,1136,0,1380,1421,1390,1476,1000,1408,0,1111,1437,0,1162,1091,1436,0,0,0,0,0,1466,0,1072,0,1370,1181,1255,0,1154,1401,1498,0,0,1445,1133,1288,1384,0,0,1408,0,0,0,1249,1051,0,0,0,1140,0,0,0,1173,1473,1162,0,0,0,1309,0,1235,0,1277,0,0,1027,1417,1333,1331,0,0,1021,1098,1182,1068,1288,1255,0,0,0,0,0,0,1021,1379,1414,1259,1309,1207,0,1325,0,1229,0,0,1008,1124,1111,1414,0,1179,0,1479,1068,0,0,0,0,1010,1376,0,1468,1265,1372,1132,0,1363,0,0,1264,1183,1343,0,0,0,1384,0,1328,1258,0,0,0,1172,0,1318,1426,0,0,0,1277,1059,0,1145,1125,0,0,0,0,0,1430,0,1209,1196,1332,0,1300,1007,1017,1164,0,0,1499,0,0,1389,1223,1496,0,0,1178,1389,1440,0,0,1430,0,1393,0,1149,0,0,1019,0,1152,0,0,1418,0,0,1196,0,1108,1298,1159,1119,1322,0,1293,0,1238,1210,1204,0,1078,1145,1401,0,0,0,1274,1136,0,1220,1150,1333,0,1465,0,0,1231,1114,1285,0,0,0,0,1224,1220,1281,0,0,1445,0,1137,1106,0,1022,1031,0,1025,1292,0,1450,0,0,0,0,1000,0,1023,0,1464,1082,0,1373,1036,1407,0,1059,1189,1374,0,1149,0,0,0,0,1313,0,0,0,1248,0,1124,1224,1257,1094,1223,1447,0,0,0,1202,1249,1436,0,0,0,1163,1187,0,0,0,0,1401,1354,0,1339,0,0,0,0,0,1477,1007,0,1303,1053,1178,1453,0,0,1452,1108,1287,1477,1407,0,0,1140,1084,1044,0,0,1021,1421,0,1229,0,1346,1363,0,1061,0,0,0,0,1050,1080,1159,1484,0,0,0,1217,1309,0,0,1498,1430,0,1105,1004,0,1025,0,1079,0,0,1127,1205,0,0,0,1342,0,0,1090,0,1056,0,1035,1066,0,0,1107,1355,1448,0,0,0,1098,1387,0,1064,1330,1156,0,0,1043,0,0,0,1330,0,0,0,0,0,1395,1231,0,1138,1158,0,0,0,0,1223,0,0,0,0,0,0,1386,1491,1328,0,0,1168,1351,1347,1233,0,0,1096,1335,0,0,0,1457,1022,0,1239,0,1190,0,0,1341,1380,1315,1413,1021,1130,1101,1166,1388,0,0,0,0,0,1067,0,1287,0,1096,0,1398,1239,1207,0,1170,0,1403,0,1406,0,0,0,1080,1119,1477,0,0,0,0,1422,1420,0,1067,1368,0,1359,0,1324,1414,0,0,0,1096,1169,1157,1170,0,1296,1151,0,0,1291,0,0,1140,0,0,1072,0,0,0,0,0,1108,0,0,1461,0,0,1189,0,0,0,0,0,1221,1314,1154,1015,1022,0,0,0,1147,0,0,1236,1021,0,0,0,0,1186,0,0,0,0,1475,0,1199,0,1335,1462,0,1198,1425,1173,1200,1127,1008,1234,1278,1107,0,1299,0,0,0,0,0,1333,0,0,1135,1495,1233,1453,1040,1201,1462,1255,0,0,0,1300,0,1455,0,1172,0,0,0,1384,1302,0,1293,1200,1468,0,0,0,0,0,1186,0,0,0,1217,1428,1127,1070,0,1142,0,1208,0,1240,1175,0,1082,1401,0,0,1127,0,1233,0,0,1282,1126,1034,0,1288,0,1435,0,0,0,0,0,1258,1122,1440,1307,1371,0,1280,1050,0,1090,1302,1060,1260,0,0,1017,0,1238,0,1340,0,0,1368,0,0,1124,0,1015,1406,1439,1306,0,1267,0,1359,0,1363,0,0,0,0,1375,0,0,0,1104,1128,0,0,0,1231,0,0,1341,1169,1069,1310,0,1299,1061,1085,0,1394,1098,1235,1262,0,0,1132,0,1326,0,1022,1201,1373,1050,1211,1178,1161,1426,1217,1086,1011,1378,1378,1355,1101,1017,1105,1214,1120,0,0,1095,0,1069,1349,0,1104,1166,1323,1202,0,0,1117,1443,1331,0,0,1013,1246,0,1344,1343,0,0,0,0,1214,0,0,0,0,0,0,0,0,0,1432,1198,0,0,0,0,1193,0,1384,1217,1023,1487,1433,0,0,0,1224,0,0,1486,0,1121,1241,1003,0,1083,1314,0,0,1064,0,0,0,1427,1133,1404,0,1163,1164,0,1196,0,0,1136,0,1121,1028,1061,1311,0,0,1284,1145,1122,1405,0,0,1289,1445,1363,1319,0,1334,1453,1116,1365,1285,1118,1243,0,1256,1479,0,0,0,1167,
0,1491,0,0,1023,1052,0,0,0,1311,0,0,1436,0,1240,1096,0,1354,1322,0,0,1059,0,0,0,0,0,0,1408,1410,0,1143,1067,1402,1199,0,0,0,0,0,0,1104,0,0,0,1479,1422,0,1015,1248,1250,0,1142,1227,1179,0,0,1386,1078,0,0,1396,1110,0,0,1186,0,0,1025,1049,0,1318,1430,0,1208,1019,1303,0,1295,1414,1281,1030,0,0,1334,0,1417,0,0,1001,0,0,1069,0,1156,1347,1262,0,1183,1413,0,0,0,1416,0,0,0,0,0,1093,1269,0,0,0,1352,1174,0,0,1179,1369,0,1321,0,0,0,1461,0,1090,0,0,0,0,0,1344,1386,1066,1194,0,0,0,0,1412,1305,1217,0,0,0,1206,1294,0,0,1492,0,1047,1281,0,0,1059,1314,0,0,1486,1153,0,1437,1001,0,1085,0,0,0,0,0,1201,1085,0,1481,0,0,1134,0,0,1332,1234,0,1436,0,1365,1275,0,0,0,0,1480,0,0,1185,1435,0,0,1290,0,1435,0,0,0,1009,1386,1026,0,1263,1098,1488,1076,1112,0,0,0,1236,1152,0,1098,0,0,1056,1146,0,0,1246,1361,1463,1274,1072,1014,1149,1347,1020,1059,0,0,0,1035,0,1491,0,0,0,1341,0,1097,1488,0,1341,1477,0,0,1312,0,0,1267,1102,1310,1372,1031,1049,0,0,1356,0,1145,0,0,0,0,0,0,1471,0,1410,0,0,0,0,0,0,0,0,1284,0,0,0,0,0,1120,0,0,0,1372,1144,1166,1130,1294,1251,0,0,0,0,0,1206,1217,1480,0,0,1490,1331,0,0,0,1478,1160,0,0,1138,1105,0,1205,0,0,0,0,0,0,0,1111,1324,1467,1477,1326,0,1064,1119,1396,0,1294,1139,0,0,1143,0,1322,1441,1245,0,0,1278,1116,0,1266,0,1180,0,0,1184,1408,1042,0,1143,1372,1486,1384,0,1119,1249,1176,0,0,0,1321,1266,0,0,1397,1298,0,0,0,0,1084,1282,0,0,0,1023,1216,0,0,1174,0,1383,0,0,0,1222,0,1132,1200,1149,0,0,0,0,0,0,1388,1382,0,0,0,0,0,1456,0,1427,0,0,0,1404,0,0,0,0,0,0,1299,0,0,0,0,0,1211,0,0,1079,0,1073,1000,1316,0,0,1430,1492,1257,0,1452,1245,1310,0,1212,0,1117,0,0,0,0,1122,0,1057,0,0,1367,0,0,1348,0,1480,1311,1075,1031,1126,1130,1147,1109,1118,0,1398,0,0,0,0,0,0,1108,0,0,1445,1439,1056,0,0,0,1250,0,1421,1021,1248,0,0,0,0,0,0,1159,1166,0,1330,1201,0,1264,1112,0,1005,0,1159,1305,1096,1450,1006,1039,1119,1301,1085,1155,1086,0,0,1216,1258,0,0,1244,1199,0,0,1020,1074,0,0,1174,1213,1095,0,0,1342,0,0,1337,1055,1267,0,1251,1393,1483,1021,0,0,0,0,1241,0,1364,1106,1414,0,1267,1236,1048,1173,1045,1269,0,1436,1423,1071,0,1175,0,1073,1168,1162,0,1015,0,1264,1420,0,0,1099,0,0,0,0,1402,1123,0,1354,1031,0,0,1141,0,1228,1054,0,0,1294,1251,1267,1140,0,1494,1203,1181,0,1070,0,1413,0,0,0,1382,1458,0,1385,1058,0,1327,1393,1300,1171,0,1049,1016,1326,0,1371,0,1475,1309,1130,1308,1287,1379,1454,1260,0,0,1023,0,0,0,0,0,0,0,1351,1347,0,1458,1279,1028,1367,0,0,1146,1483,1152,1310,0,1214,1489,1039,1478,0,1017,1159,0,1088,1045,0,0,0,1154,0,0,1067,1214,1054,0,0,0,0,0,0,0,0,0,1174,1096,1019,0,0,1015,1415,1199,0,1170,1351,0,0,1381,1474,1192,1110,1219,1115,1240,1033,0,0,0,0,0,0,1129,1184,1255,1090,0,1240,0,0,0,1098,1233,0,0,0,0,1096,0,0,0,1425,1436,0,1078,1387,0,0,1137,1164,0,0,0,0,1064,1492,1044,1325,1056,1310,1263,0,0,1084,0,1330,1343,0,1402,0,0,1320,0,1277,1483,1488,0,1345,0,0,1032,0,0,1170,0,1381,1298,0,0,0,1156,0,0,0,1439,1482,1113,1111,1210,1002,1486,0,1489,1020,0,1120,1123,1038,0,1399,1008,1406,0,1094,1424,1444,1242,0,0,0,1383,1227,1141,1140,1041,0,1379,1143,1270,0,0,1191,0,1136,1424,0,1304,1035,1441,0,1235,1044,1283,0,0,0,1322,1178,0,1162,0,0,0,1407,0,1196,1307,0,0,1164,1199,1121,1126,0,1073,1057,1107,0,1247,0,1322,1167,0,1365,0,1085,0,1325,1079,0,0,0,1252,1067,1284,1071,0,0,1134,1007,1240,1104,0,1120,0,1147,1270,1308,0,1304,1007,0,1398,1423,1135,0,1161,0,0,0,0,0,1117,0,0,0,1360,0,1262,0,0,1435,1238,1419,1216,0,1351,0,1102,1178,1395,0,1458,0,1487,1188,1146,1152,0,0,0,1320,0,0,1325,0,1464,0,1211,1357,0,0,0,1163,0,0,1263,1371,1332,0,0,1378,1190,0,0,0,0,0,0,0,0,1181,1432,0,0,0,0,1056,1276,0,0,0,1302,1462,1133,1117,0,1394,1074,0,0,1257,0,1076,1420,0,0,0,1022,0,0,1248,1144,1130,1056,1020,1281,0,0,0,1417,0,0,0,1301,1387,1389,0,1208,0,0,0,1327,0,1293,1445,1160,0,1416,1100,0,1418,1436,1229,1357,0,0,1173,1417,0,0,1131,0,1158,1297,0,0,0,0,1175,0,0,1406,1031,1028,0,1218,0,0,0,0,1426,0,1174,0,1342,0,0,1480,1289,1188,1119,1040,0,1154,1098,0,0,1271,0,1124,0,0,1137,0,1285,0,0,0,0,0,1348,0,0,0,0,1133,0,1096,0,0,0,1066,0,0,1202,1044,1231,0,0,1373,1202,0,0,0,0,0,1091,1378,0,0,1325,1389,1057,0,1013,0,0,1090,1332,1441,0,1071,1092,0,0,1192,1297,1019,1148,1331,0,0,0,0,1428,1175,0,1243,0,0,1397,0,0,1035,1248,1148,1442,0,0,1179,0,1137,1467,0,0,1499,0,0,0,1468,1078,0,1172,1423,1123,0,1058,1391,0,0,0,1312,1079,0,1259,1432,0,1179,0,1242,0,1365,1458,1354,1010,1419,1323,1155,0,0,1333,1132,0,0,0,1362,1059,1151,0,0,0,1016,0,1440,0,1022,1012,0,1068,0,0,0,0,1338,1313,0,1256,0,1315,1138,0,0,1293,1397,1377,0,1115,0,0,1041,1119,1229,1032,1435,0,1110,0,0,0,0,1227,1377,0,1318,0,0,0,0,1022,1095,0,0,1346,1053,0,1026,0,1125,0,0,0,1031,1114,1023,1001,0,1077,0,0,1366,1410,1159,1153,1408,1252,1222,0,0,0,0,1431,1025,0,0,1392,1079,0,0,1190,0,0,0,0,0,0,1190,1229,1187,0,1268,1006,1066,1025,0,0,0,1367,0,1419,0,0,1181,0,0,1422,1117,0,1328,1205,0,0,1114,0,1458,1157,0,1312,0,1452,1069,1020,1021,1215,0,1363,0,1010,0,0,0,0,1108,0,1411,0,1074,0,0,1373,1451,0,0,0,1463,1481,0,1308,1211,0,1262,1080,1305,0,0,0,0,1046,0,0,0,0,1138,0,1496,1480,1354,0,1221,1170,1381,0,0,1457,1191,1408,0,1189,1187,1288,0,0,0,0,0,1166,0,0,1339,1429,1318,0,1438,0,0,0,0,1098,0,0,0,1444,0,1288,0,1193,1491,1060,1238,0,1348,0,1376,1132,1091,0,1091,0,1129,1086,1126,1374,1438,1298,0,1320,0,0,0,1479,0,0,1404,0,0,0,0,0,1441,0,1427,0,1299,1471,1136,0,1204,0,0,1182,1078,1208,1266,0,1490,1337,0,1062,0,0,1104,0,0,0,0,0,0,0,0,1241,1055,0,0,1222,1496,1203,0,1296,0,0,0,0,1187,0,0,1160,0,0,0,1247,1452,0,0,0,0,0,1130,0,0,0,1045,1477,0,1462,0,1254,0,0,1133,1057,1248,1472,1175,1424,0,0,1440,0,0,0,0,1288,1346,1324,0,0,0,1097,0,1236,1295,0,0,0,1405,0,1408,1264,1348,1023,0,0,1197,0,1243,0,0,0,0,0,1430,0]

# _dataset = pd.read_csv("data/M4DataSet/Monthly-train.csv")
# ts = _dataset['V560'].fillna(0).values[:1000]

time_start = time.perf_counter()
pbounds = ((1, len(ts)), (0, len(ts)))
wopt = dual_annealing(_ata_cost, pbounds,
                                args=(np.asarray(ts), len(ts), 1e-7), no_local_search = True) # restart_temp_ratio= 2e-3
print (("{0} s").format(time.perf_counter() - time_start))
constrained_wopt = wopt.x
fun = wopt.fun
print(wopt.message)
print("opt P: {0}   Q: {1} mse: {2}".format(constrained_wopt[0],constrained_wopt[1], fun))
