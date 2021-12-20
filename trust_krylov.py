'''
Trust-Region Truncated Generalized Lanczos / Conjugate Gradient Algorithm (method='trust-krylov')

Similar to trust-ncg, it's suitable for large-scale problems as
uses hessian only as linear operator by means of matrix-products.
Solved quadratic more accurately than trust-ncg

Reduces the number of nonlinear iterations at the expense of few more matrix-vector products per subproblem solve.
'''
from optimization import rosen, rosen_der, rosen_hess, rosen_hess_p
import numpy as np
from scipy.optimize import minimize

if __name__=="__main__":
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    # full hessian
    res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': True})
    print(res.x)

    # hessian product
    res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hessp=rosen_hess_p, options={'gtol': 1e-8, 'disp': True})
    print(res.x)