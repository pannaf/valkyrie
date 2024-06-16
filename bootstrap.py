import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

print(f"{DATABASE_URL=}")

# SQL statements to create tables
CREATE_TABLES_SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    first_name TEXT,
    last_name TEXT,
    email TEXT UNIQUE,
    date_of_birth DATE,
    height REAL,
    join_date DATE DEFAULT CURRENT_DATE,
    onboarded BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_profile_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    last_updated DATE,
    activity_preferences TEXT,
    workout_location TEXT,
    workout_frequency TEXT,
    workout_duration TEXT,
    workout_constraints TEXT,
    fitness_level TEXT,
    weight DOUBLE PRECISION,
    goal_weight DOUBLE PRECISION,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS goals (
    goal_id TEXT PRIMARY KEY,
    user_id UUID,
    goal_type TEXT,
    description TEXT,
    target_value TEXT,
    current_value TEXT,
    unit TEXT,
    start_date DATE,
    end_date DATE,
    goal_status TEXT,
    notes TEXT,
    last_updated DATE,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""


def create_tables():
    """Create tables in the PostgreSQL database."""
    conn = None
    try:
        # Connect to the PostgreSQL server
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        # Create tables
        cur.execute(CREATE_TABLES_SQL)
        # Commit the changes
        conn.commit()
        # Close communication with the PostgreSQL database server
        cur.close()
        print("Tables created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    create_tables()
