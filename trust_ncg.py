'''
Trust-Region Newton-Conjugate-Gradient Algorithm (method='trust-ncg')

The Newton-CG method is a line search method: finds direction of search minimizing
a quadratic approximation of the function and uses a line search algorithm to find
the (nearly) optimal step size in that direction.

An alternative approach is to, first, fix the step size limit (lambda sign)
and then find the optimal step (sign 'P') inside the given trust-radius 
by solving the following quadratic subproblem.
'''
import numpy as np
from scipy.optimize import minimize
from nelder_mead import rosen
from BFGS import rosen_der
from ncg import rosen_hess, rosen_hess_p
if __name__ == '__main__':
    # full hessian
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='trust-ncg', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': True})
    print(res.x)

    # hessian product
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='trust-ncg', jac=rosen_der, hessp=rosen_hess_p, options={'gtol': 1e-8, 'disp': True})
    print(res.x)