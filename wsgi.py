"""WSGI entrypoint for Gunicorn."""
from app import app  # noqa: F401
from modules.storage import init_db

init_db()

if __name__ == "__main__":
    app.run()
