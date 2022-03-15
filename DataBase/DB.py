import psycopg2
class DB():
    __host = 'ec2-3-215-57-87.compute-1.amazonaws.com'
    __user = 'jxmysjfosljroi'
    __dbname = 'dcqn8t2vn9g29l'
    __password = '3cfd7b500b3adc86b1b48d3ed9278817b9dc8b54ed2568c26a602b7687fa607e'
    __sslmode = 'require'

    @staticmethod
    def getConn():
        conn_string = f'host={DB.__host} user={DB.__user} dbname={DB.__dbname} password={DB.__password} sslmode={DB.__sslmode}'
        return psycopg2.connect(conn_string)

