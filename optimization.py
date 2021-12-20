import numpy as np
from scipy.optimize import minimize

# Unconstrained minimization of multivariate scalar function

# Nelder-Mead Simplex algorithm (method='Nelder-Mead')
def rosen(x):
    return sum(100*(x[1:] - x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

# Broyden-Fletcher-Goldfarb-Shanno algorithm (method='BFGS')
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1-xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

# Newton-Conjugate-Gradient algorithm (method='Newton-CG')
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
    # Use product matrix rather than hole hessian matrix.
    # Use Newton-CG with hessian product option is the fastest way to minimize the function.
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2] + (202 + 1200*x[1:-1]**2 - 400*x[2:])*p[1:-1] - 400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp
