import psycopg2


def create_table(db_connect):
    # db생성 쿼리
    create_table_query = """
    CREATE TABLE IF NOT EXISTS iris_data (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        sepal_length float8,
        sepal_width float8,
        petal_length float8,
        petal_width float8,
        target int
    );"""

    # 쿼리 실행 
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()
        
    print(create_table_query)

if __name__ == "__main__":
    # db연결
    db_connect = psycopg2.connect(
        user="myuser",
        password="1234",
        host="localhost",
        port=5432,
        database="mydatabase",
    )
    
create_table(db_connect)



    
