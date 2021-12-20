from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, BFGS, minimize, SR1
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from optimization import rosen, rosen_der, rosen_hess, rosen_hess_p
import numpy as np
# Bound Constraints :
# 0 <= x0 <= 1, -0.5 <= x1 <= 2.0 -> defined using Bound object
bounds = Bounds([0, -0.5], [1.0, 2.0])

# Linear Constraints :
# x0 + 2x1 <= 1
# 2x0 + x1 = 1 -> defined by LinearConstraint
linear_constraints = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

# Nonliner Constraints :
def cons_f(x):
    return [x[0]**2 + x[1], x[0]**2 - x[1]]

def cons_J(x):
    return [[2*x[0]**2, 1], [2*x[0]**2, -1]]

def cons_H(x, v):
    return v[0]*np.array([[2, 0], [0, 0]]) + v[1]*np.array([[2, 0], [0, 0]])

nonlinear_constraints = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)

# sparse matrix
def cons_H_sparse(x, v):
    return v[0]*csc_matrix([[2, 0], [0, 0]]) + v[1]*csc_matrix([[2, 0], [0, 0]])

nonlinear_constraints = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H_sparse)

# LinearOperator
def cons_H_linear_operator(x, v):
    def matvec(p):
        return np.array([p[0]*2*(v[0]+v[1]), 0])
    return LinearOperator((2, 2), matvec=matvec)
nonlinear_constraints = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H_linear_operator)

# HessianUpdateStrategy
# The evaluation of Hessian is difficult to implements, can use BFGS and SR1
nonlinear_constraints = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=BFGS())

nonlinear_constraints = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess='2-point')

# Jacobian constrains could be approximated by finite differances, the Hessian can't be computed.
# and needs to be provide by user or defined using HessianUpdateStrategy
nonlinear_constraints = NonlinearConstraint(cons_f, -np.inf, 1, jac='2-point', hess=BFGS())


# with LinearOperator
def rosen_hess_linop(x):
    def matvec(p):
        return rosen_hess_p(x, p)
    return LinearOperator((2, 2), matvec=matvec)
if __name__ == "__main__":

    x0 = np.array([0.5, 0])
    res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess, constraints=[linear_constraints, nonlinear_constraints], options={'verbose':1}, bounds=bounds)
    print(res.x)

    res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess_linop, constraints=[linear_constraints, nonlinear_constraints], options={'verbose':1}, bounds=bounds)
    print(res.x)

    res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hessp=rosen_hess_p,
                   constraints=[linear_constraints, nonlinear_constraints], options={'verbose': 1}, bounds=bounds)
    print(res.x)

    res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=SR1(),
                   constraints=[linear_constraints, nonlinear_constraints], options={'verbose': 1}, bounds=bounds)
    print(res.x)