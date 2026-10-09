import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent  # repository root (a local .env here is loaded automatically)


def load_project_env():
    load_dotenv(PROJECT_ROOT / ".env")


def get_db_params():
    load_project_env()
    return {
        "dbname": os.getenv("DB_NAME", "dectwin"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5433"),
    }