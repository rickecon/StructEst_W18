'''
------------------------------------------------------------------------
Dynamic programming cake eating
------------------------------------------------------------------------
This code solves the simple deterministic cake eating problem by value
function iteration

This script calls the following function(s)
    vf.get_V_cur_W_pr()
    vf.print_time()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vfi_funcs as vf

'''
------------------------------------------------------------------------
Set parameters, initial conditions
------------------------------------------------------------------------
beta      = scalar in (0, 1), discount factor
W_0       = scalar > 0, initial cake size and maximum cake size
W_min     = scalar > 0, minimum cake size in discrete support of W
W_max     = scalar > 0, maximum cake size in discrete support of W
W_size    = integer >= 2, number of points in the support of W
W_vec     = (W_size,) vector, discrete support of W
maxiter   = int >= 1, maximum iterations for value function iteration
tol_VFI   = scalar > 0, convergence criterion tolerance distance for VFI
graph_VFI =
------------------------------------------------------------------------
'''
# set up parameters, w matrix, etc.
beta = 0.9
W_0 = 1.0
W_min = 0.01
W_max = W_0
W_size = 100
W_vec = np.linspace(W_min, W_max, W_size)

maxiter = 200
tol_VFI = 1e-14
graph_VFI = True


'''
------------------------------------------------------------------------
Perform value function iteration (VFI)
------------------------------------------------------------------------
start_time = scalar > 0, start time according to computer clock
V_next     = (W_size,) vector, initial guess for next-period's value
             function of W
iter_VFI   = integer >= 0, current iteration number of VFI
dist       = scalar > 0, current iteration distance measure of V_cur and
             V_next
V_cur      = (W_size,) vector, current period value function
W_prime    = (W_size,) vector, policy function for W'=psi(W)
vfi_time   = scalar > 0, time (in seconds) elapsed for VFI computation
V          = (W_size,) vector, solution for value function
------------------------------------------------------------------------
'''
start_time = time.clock()
V_next = np.zeros(W_size)
iter_VFI = 0
dist = 10.0
while (iter_VFI < maxiter) and (dist >= tol_VFI):
    iter_VFI += 1
    V_cur, W_prime = vf.get_V_cur_W_pr(V_next, beta, W_vec)
    dist = ((V_cur - V_next) ** 2).sum()
    print('Iter=', iter_VFI, ', distance=', "%10.4e" % dist)
    V_next = V_cur

vfi_time = time.clock() - start_time
vf.print_time(vfi_time, 'VFI')
V = V_cur.copy()

if graph_VFI:
    '''
    --------------------------------------------------------------------
    Plot the value function and policy function solutions
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    image_fldr = 'images'
    image_dir = os.path.join(cur_path, image_fldr)
    if not os.access(image_dir, os.F_OK):
        os.makedirs(image_dir)

    # Plot value function
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(W_vec, V)
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Value function: V(W)')
    plt.xlabel(r'Cake size $W$')
    plt.ylabel(r'Value function $V(W)$')
    image_path = os.path.join(image_dir, 'V')
    plt.savefig(image_path)
    # plt.show()
    plt.close()

    # Plot policy function
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(W_vec, W_prime)
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Policy function')
    plt.xlabel(r'Current cake size $W$')
    plt.ylabel(r'Next period cake size $W_pr$')
    image_path = os.path.join(image_dir, 'W_pr')
    plt.savefig(image_path)
    # plt.show()
    plt.close()
