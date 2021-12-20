'''
Newton-Conjugate-Gradient algorithm (method='Newton-CG')
'''
import numpy as np
from scipy.optimize import minimize
from nelder_mead import rosen
from BFGS import rosen_der
'''
i -> [1:-1]
i - 1 -> [:-2]
i + 1 -> [2:]
supply code to compute this product rather than the full Hessian matrix by giving a hess function 
if possible, using Newton-CG with the Hessian product option is probably the fastest way to minimize the function.
'''
def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1], 1) - np.diag(400*x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H
def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2] + (202 + 1200*x[1:-1]**2 - 400*x[2:])*p[1:-1] - 400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp
if __name__=='__main__':

    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess, options={'xtol':1e-8, 'disp':True})
    print(res.x)

    res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hessp=rosen_hess_p, options={'xtol':1e-8, 'disp':True})
    print(res.x)