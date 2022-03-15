import numpy as np
import matplotlib.pyplot as plt
import cvxopt as otp
from cvxopt import blas, solvers
# import panda as pd
from plotly.matplotlylib.mplexporter._py3k_compat import xrange

np.random.seed(123)
solvers.options['show_progress'] = False

# print(plotly.__version__)

# from chart_studio.grid_objs import *

## NUMBERS OF ASSERT
n_asset = 4
## NUMBER OF OBSERVATIONS
n_obs = 1000
return_vec = np.random.randn(n_asset, n_obs)
print(return_vec)
fig = plt.figure()
plt.plot(return_vec.T, alpha=.4)
plt.xlabel('time')
plt.ylabel('returns')
# iplot_mpl(fig, filename='123')
# plt.show()

def rand_weights(n):
    # create a random weight vector to plot those portfolio
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)
# print(rand_weights(n_asset))
# print(rand_weights(n_asset))

def random_portfolio(returns):
    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    c = np.asmatrix(np.cov(returns))
    # p.T -> transpose of the vector for the mean each time
    # w -> weight vector of the portfolio
    mu = w*p.T

    # standard deviation
    sigma = np.sqrt(w*c*w.T)

    # only allow standard deviation < 2
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

n_portfolios = 500
means, stds = np.column_stack([random_portfolio(return_vec) for i in xrange(n_portfolios)])
fig = plt.figure()
plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
# py.iplot_mpl(fig, filename='mean_std', strip_style=True)
# plt.show()

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    # produces a series of expected return values mu
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    S = otp.matrix(np.cov(returns))
    pbar = otp.matrix(np.mean(returns, axis=1))

    # constraint matrices
    G = -otp.matrix(np.eye(n))
    h = otp.matrix(0.0, (n, 1))
    A = otp.matrix(1.0, (1, n))
    b = otp.matrix(1.0)

    # compute the portfolio with excepted return value bounding constrain
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]

    # calculate return and risk for frontier by portfolios
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])

    wt = solvers.qp(otp.matrix(x1*S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks



weights, returns, risks = optimal_portfolio(return_vec)
print(weights)
fig = plt.figure()
plt.plot(stds, means, 'o')
plt.ylabel('means')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')
plt.show()
