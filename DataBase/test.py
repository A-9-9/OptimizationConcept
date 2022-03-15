import psycopg2
from DB import DB
from models import Weight, Company
import numpy as np
import cvxopt as otp
from cvxopt import blas, solvers
from openpyxl import load_workbook
def get_Companys():
    conn = DB.getConn()
    cursor = conn.cursor()

    # 執行SQL命令
    cursor.execute("SELECT * FROM polls_company")
    data = cursor.fetchall()

    result = []
    for i in data:
        result.append(Company(*i))

    cursor.close()
    conn.close()
    return result
def get_company_by_id(id):
    conn = DB.getConn()
    cursor = conn.cursor()

    # 執行SQL命令
    cursor.execute("SELECT * FROM polls_company WHERE id = %s" % id)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return Company(*data[0])
def get_company_by_name(name):
    conn = DB.getConn()
    cursor = conn.cursor()

    # 執行SQL命令
    cursor.execute("SELECT * FROM polls_company WHERE comp_name = '%s'" % name.upper())
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    if not data:
        raise ValueError('There is not name match in comp_name column.')
    return Company(*data[0])
def get_Wieghts():
    conn = DB.getConn()
    cursor = conn.cursor()

    # 執行SQL命令
    cursor.execute("SELECT * FROM polls_weight")
    data = cursor.fetchall()

    result = []
    for d in data:
        result.append(Weight(*d))

        # 關閉cursor及連線
    cursor.close()
    conn.close()

    # 回傳資料
    return result
def optimization(file='/'):
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
    comp = []
    for i in li[2:]:
        comp.append(i[0])
        return_vec.append(i[1:])
    weights, returns, risks = optimal_portfolio(return_vec)

    return weights, returns, risks, comp
def save_weight(weight, company):
    conn = DB.getConn()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO polls_weight (weight, company_id) VALUES (%s, %s)" % (weight, company.id))

    conn.commit()

    cursor.close()
    conn.close()
def save_weights(weights, comp):
    weights = list(map(list, weights))
    if len(weights) != len(comp):
        raise AttributeError('The length between Company and Weights are not same.')

    for i in range(len(comp)):
        try:
            company = get_company_by_name(comp[i])
        except ValueError:
            # could not found related company in database, create a new company in database.
            pass
        finally:
            save_weight(weights[i][0], company)
            print("Company %s Weight: %s saved." % (company, weights[i][0]))
            print('='*20)


# weights, returns, risks, comp = optimization(file='/Users/apple/Downloads/S_P500.xlsx'
# save_weights(weights, comp)

for i in get_Wieghts():
    print(i)
