import numpy as np
import cvxopt as otp
from cvxopt import blas, solvers
from openpyxl import load_workbook

# output save into database
def fn(file='/'):
    def optimal_portfolio(returns):
        n = len(returns)
        returns = np.asmatrix(returns)

        N = 100

        mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]
        S = otp.matrix(np.cov(returns))
        pbar = otp.matrix(np.mean(returns, axis=1))

        G = -otp.matrix(np.eye(n))
        h = otp.matrix(0.0, (n, 1))
        A = otp.matrix(1.0, (1, n))
        b = otp.matrix(1.0)

        portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x'] for mu in mus]

        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])

        wt = solvers.qp(otp.matrix(x1 * S), -pbar, G, h, A, b)['x']
        return np.asarray(wt), returns, risks

    wb = load_workbook(file, data_only=True)
    sheets = wb.get_sheet_names()
    bs = wb.get_sheet_by_name(sheets[2])
    columns = bs.columns
    li = []
    for i in columns:
        li.append(list([x.value for x in i]))

    return_vec = []
    # save company name
    # comp = []
    # for i in li[2:]:
    #     comp.append(i[0])
    #     return_vec.append(i[1:])
    weights, returns, risks = optimal_portfolio(return_vec)

    return weights, returns, risks

weights, returns, risks = fn(file='/Users/apple/Downloads/S_P500.xlsx')

print(weights)

