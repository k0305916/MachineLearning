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

ts = [362.35, 0.0, 0.0, 0.0, 0.0, 361.63, 0.0, 0.0, 0.0, 0.0, 362.77,
      0.0, 0.0, 0.0, 0.0, 356.11, 0.0, 0.0, 0.0, 0.0, 0.0, 331.26,
      0.0, 0.0, 0.0, 0.0, 304.87, 0.0, 0.0, 0.0, 0.0, 0.0, 311.06,
      0.0, 0.0, 0.0, 0.0, 323.62, 0.0, 0.0, 0.0, 0.0, 336.57, 0.0,
      0.0, 0.0, 0.0, 342.40, 0.0, 0.0, 0.0, 0.0, 343.57, 0.0, 0.0,
      0.0, 0.0, 349.56, 0.0, 0.0, 0.0, 0.0, 356.45, 0.0, 0.0, 0.0,
      0.0, 362.56, 0.0, 0.0, 0.0, 0.0, 360.64, 0.0, 0.0, 0.0, 0.0,
      312.74, 0.0, 0.0, 0.0, 0.0]

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
