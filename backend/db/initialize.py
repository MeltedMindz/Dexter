import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DB_CONFIG = {
    "database": os.getenv("DB_NAME", "dexbrain"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432"))
}

if not DB_CONFIG["password"]:
    raise ValueError("DB_PASSWORD environment variable is required")

def initialize_database():
    connection = psycopg2.connect(
        database="postgres",
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = connection.cursor()

    try:
        cursor.execute("CREATE DATABASE dexbrain;")
        print("Database created successfully.")
    except psycopg2.errors.DuplicateDatabase:
        print("Database already exists.")

    cursor.close()
    connection.close()

    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()

    with open("db/schema.sql", "r") as schema_file:
        cursor.execute(schema_file.read())

    connection.commit()
    cursor.close()
    connection.close()
    print("DexBrain database initialized.")

if __name__ == "__main__":
    initialize_database()
