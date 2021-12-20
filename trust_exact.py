'''
Trust-Region Nearly Exact Algorithm (method='trust-exact')

for medium-size problem
because storage and factorization cost of the Hessian are not critical,
it is possible to obtain a solution within fewer iteration by solving the trust-region subproblems.
The Hessian product option is not supported by this algorithm.
'''

from optimization import rosen, rosen_der, rosen_hess, rosen_hess_p
import numpy as np
from scipy.optimize import minimize

if __name__=='__main__':
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='trust-exact', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': True})
    print(res.x)