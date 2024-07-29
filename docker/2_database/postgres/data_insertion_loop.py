import time
import data_insertion
import psycopg2

def generate_data(db_connect, df):
    while True:
        data_insertion.insert_data(db_connect, df.sample(1).squeeze())
        time.sleep(1)


if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host="localhost",
        port=5432,
        database="mydatabase",
    )
    df = data_insertion.get_data()
    generate_data(db_connect, df)