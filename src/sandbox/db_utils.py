import os
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


@contextmanager
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def fetch_user(user_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
            return user


def fetch_user_profile(user_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
            return user


def update_user_profile(user_id, field, value):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"UPDATE user_profiles SET {field} = %s, last_updated = NOW() WHERE user_id = %s"
            cur.execute(query, (value, user_id))
            conn.commit()
    return f"User profile updated successfully {field} = {value}"


def fetch_goals_db(user_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM goals WHERE user_id = %s", (user_id,))
            goals = cur.fetchall()
            return goals


def update_goal_db(user_id, goal_id, field, value):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"UPDATE goals SET {field} = %s, last_updated = NOW() WHERE user_id = %s AND goal_id = %s"
            cur.execute(query, (value, user_id, goal_id))
            conn.commit()
    return f"Goal updated successfully {field} = {value}"


def create_empty_goal_db(user_id, goal_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO goals (goal_id, user_id, goal_type, description, target_value, current_value, unit, start_date, end_date, goal_status, notes, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """,
                (goal_id, user_id, None, None, None, None, None, None, None, "Pending", None),
            )
            conn.commit()
    return goal_id
