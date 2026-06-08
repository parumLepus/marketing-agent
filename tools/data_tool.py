import os
from langchain.tools import tool
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

engine = sqlalchemy.create_engine(DATABASE_URL)

@tool
def list_database_tables(placeholder: str = "") -> str:
    """
        Returns all available tables and their columns in the marketing database.
        Call this first if you're unsure what data is available.
    """
    query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """
    try:
        df = pd.read_sql_query(query, engine)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_marketing_data(query: str) -> str:
    """
    Fetch marketing data from the Postgres database using SQL.

    Available tables:
    - campaigns(id, name, channel, spend, impressions, clicks, conversions, month)
    - traffic(id, source, sessions, bounce_rate, avg_duration_seconds, month)
    - content(id, title, type, views, shares, leads_generated, published_date)

    Use this when you need real numbers: campaign performance, traffic sources,
    conversion rates, content engagement, spend vs results, etc.
    Write a valid PostgreSQL SELECT query.
    """
    try:
        df = pd.read_sql_query(query, engine)
        if df.empty:
            return "No data found for that query."
        return df.to_string(index=False)
    except Exception as e:
        return f"Error: {e}"
