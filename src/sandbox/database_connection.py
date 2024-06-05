import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables from the .env file
load_dotenv()

# Get the PostgreSQL connection URL from the environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Establish the connection
conn = psycopg2.connect(DATABASE_URL)

try:
    # Create a cursor
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Execute a query
    cursor.execute("SELECT * FROM users")

    # Fetch all results
    users = cursor.fetchall()

    # Print results
    for user in users:
        print(user)
finally:
    # Close the cursor and connection
    cursor.close()
    conn.close()
