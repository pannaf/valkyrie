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


def set_user_field_db(user_id, field, value):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"UPDATE users SET {field} = %s WHERE user_id = %s"
            cur.execute(query, (value, user_id))
            conn.commit()
    return f"User profile updated successfully {field=} : {value=}"


def set_user_fitness_level_db(user_id, value):
    field = "fitness_level"
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"UPDATE users SET {field} = %s, last_updated = NOW() WHERE user_id = %s"
            cur.execute(query, (value, user_id))
            conn.commit()
    return f"User profile updated successfully {field} = {value}"


def fetch_user_activities_db(user_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM user_activities WHERE user_id = %s", (user_id,))
            activities = cur.fetchall()
            return activities


def update_user_activities_db(user_id, activity_id, field, value):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"UPDATE user_activities SET {field} = %s WHERE user_id = %s AND user_activity_id = %s"
            cur.execute(query, (value, user_id, activity_id))
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


def create_empty_activity_db(user_id, user_activity_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_activities (user_activity_id, user_id, activity_name, activity_location, activity_duration, activity_frequency)
                VALUES (%s, %s, %s, %s, %s, %s)
            """,
                (user_activity_id, user_id, None, None, None, None),
            )
            conn.commit()
    return user_activity_id


def insert_user(first_name, last_name, email, date_of_birth, height_cm):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if the email already exists
            cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
            existing_user = cur.fetchone()
            if existing_user:
                return existing_user["user_id"]

            # Insert the new user
            cur.execute(
                """
                INSERT INTO users (first_name, last_name, email, date_of_birth, height)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING user_id
                """,
                (first_name, last_name, email, date_of_birth, height_cm),
            )
            user_id = cur.fetchone()["user_id"]

            # Insert into user_profiles
            cur.execute(
                """
                INSERT INTO user_profiles (user_id, last_updated)
                VALUES (%s, NOW())
                """,
                (user_id,),
            )

            conn.commit()
    return user_id
