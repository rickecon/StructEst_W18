'''
------------------------------------------------------------------------
Functions module for simple cake eating problem value function iteration
------------------------------------------------------------------------
This module is called by dynprog_cake1.py and defines the following
function(s):
    get_neg_V_cur()
    get_V_cur_W_pr()
    print_time()
------------------------------------------------------------------------
'''
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def get_neg_V_cur(W_pr, *args):
    '''
    --------------------------------------------------------------------
    Given values for W', W, and beta and the corresponding interpolated
    V(W'), solve for the negative of V(W), which is the object of the
    minimizer
    --------------------------------------------------------------------
    INPUTS:
    W_pr = scalar > 0, cake size next period (W prime)
    args = length 4 tuple, (beta, W_cur, W_vec, V_next_vec)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    beta       = scalar in (0, 1), discount factor
    W_cur      = scalar > 0, current size of the cake W
    W_vec      = (W_size,) vector, discretized support of W
    V_next_vec = (W_size,) vector, discretized next period value
                 function V(W')
    V_next     = scalar, value function tomorrow given W_pr, V(W')
    neg_V_cur  = scalar, negative current period value function value
                 for given W, W', and V'

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: neg_V_cur
    --------------------------------------------------------------------
    '''
    beta, W_cur, W_vec, V_next_vec = args
    V_next_intp = interp1d(W_vec, V_next_vec, kind='linear',
                           fill_value='extrapolate')
    V_next = V_next_intp(W_pr)
    neg_V_cur = -(np.log(W_cur - W_pr) + beta * V_next)

    return neg_V_cur


def get_V_cur_W_pr(V_next, beta, W_vec):
    '''
    --------------------------------------------------------------------
    Compute the current period value function given a guess for the next
    period value function. This function is the contraction operator
    that is iteratively applied to guesses of the value function in
    order to find the solution
    --------------------------------------------------------------------
    INPUTS:
    V_next = (W_size,) vector, next-period value function V(W')
    beta   = scalar in (0, 1), discount factor
    W_vec  = (W_size,) vector, discretized support of cake size W

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    W_size       = integer >= 2, number of elements in support of W
    W_prime      = (W_size,) vector, policy function for cake size next
                   period: W'=psi(W)
    V_cur        = (W_size,) vector, current period value function V(W)
    ind          = integer > 0, index of element of W_prime for which
                   solving
    W_cur        = scalar > 0, current cake size
    W_pr_init    = scalar in (W_lb, W_ub), initial guess for W_pr
    W_ub         = scalar > 0, upper bound on W_pr (W_cur - epsilon)
    W_lb         = scalar > 0, lower bound on W_pr (0 + epsilon)
    W_pr_args    = length 4 tuple, arguments to pass into
                   get_neg_V_cur()
    results_W_pr = results object, minimize() results


    FILES CREATED BY THIS FUNCTION: None

    RETURNS: V_cur, W_prime
    --------------------------------------------------------------------
    '''
    W_size = len(W_vec)
    W_prime = np.zeros(W_size)
    V_cur = np.zeros(W_size)
    for ind in range(W_size):
        W_cur = W_vec[ind]
        W_pr_init = 0.5 * W_cur
        W_ub = W_cur - 1e-10
        W_lb = 1e-10
        W_pr_args = (beta, W_cur, W_vec, V_next)
        results_W_pr = \
            minimize(get_neg_V_cur, W_pr_init, args=(W_pr_args),
                     method='L-BFGS-B', bounds=((W_lb, W_ub),))
        W_prime[ind] = results_W_pr.x
        V_cur[ind] = -results_W_pr.fun

    return V_cur, W_prime


def print_time(seconds, type):
    '''
    --------------------------------------------------------------------
    Takes a total amount of time in seconds and prints it in terms of
    more readable units (days, hours, minutes, seconds)
    --------------------------------------------------------------------
    INPUTS:
    seconds = scalar > 0, total amount of seconds
    type    = string, either "VFI"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    secs = scalar > 0, remainder number of seconds
    mins = integer >= 1, remainder number of minutes
    hrs  = integer >= 1, remainder number of hours
    days = integer >= 1, number of days

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Nothing
    --------------------------------------------------------------------
    '''
    if seconds < 60:  # seconds
        secs = round(seconds, 4)
        print(type + ' computation time: ' + str(secs) + ' sec')
    elif seconds >= 60 and seconds < 3600:  # minutes
        mins = int(seconds / 60)
        secs = round(((seconds / 60) - mins) * 60, 1)
        print(type + ' computation time: ' + str(mins) + ' min, ' +
              str(secs) + ' sec')
    elif seconds >= 3600 and seconds < 86400:  # hours
        hrs = int(seconds / 3600)
        mins = int(((seconds / 3600) - hrs) * 60)
        secs = round(((seconds / 60) - hrs * 60 - mins) * 60, 1)
        print(type + ' computation time: ' + str(hrs) + ' hrs, ' +
              str(mins) + ' min, ' + str(secs) + ' sec')
    elif seconds >= 86400:  # days
        days = int(seconds / 86400)
        hrs = int(((seconds / 86400) - days) * 24)
        mins = int(((seconds / 3600) - days * 24 - hrs) * 60)
        secs = round(
            ((seconds / 60) - days * 24 * 60 - hrs * 60 - mins) * 60, 1)
        print(type + ' computation time: ' + str(days) + ' days, ' +
              str(hrs) + ' hrs, ' + str(mins) + ' min, ' +
              str(secs) + ' sec')
