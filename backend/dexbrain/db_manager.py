import psycopg2
import pandas as pd
from psycopg2.extras import Json

DB_CONFIG = {
    "database": "dexbrain",
    "user": "postgres",
    "password": "password123",
    "host": "localhost",
    "port": 5432
}

class DexBrainDB:
    def __init__(self):
        self.connection = psycopg2.connect(**DB_CONFIG)

    def query_strategies(self, token_pair):
        query = """
        SELECT strategy_details FROM strategies
        WHERE token_pair = %s
        ORDER BY timestamp DESC LIMIT 1
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (token_pair,))
            result = cursor.fetchone()
            return result[0] if result else None

    def add_strategy(self, token_pair, strategy_details, performance_metrics):
        query = """
        INSERT INTO strategies (token_pair, strategy_details, performance_metrics)
        VALUES (%s, %s, %s)
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (token_pair, Json(strategy_details), Json(performance_metrics)))
        self.connection.commit()

    def list_datasets(self):
        query = "SELECT id, name, description FROM datasets"
        return pd.read_sql(query, self.connection)
