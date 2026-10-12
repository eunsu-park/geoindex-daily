"""PostgreSQL access to the shared `space_weather` / `solar_images` databases.

Credentials come from the SOLARIS_DB_* env vars in ~/.solaris_env (the same ones
solaris-data and geoindex-data use). Nothing here writes.
"""
import os

import psycopg2


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name) or default
    if v is None:
        raise KeyError(f"{name} is not set (source ~/.solaris_env)")
    return v


def connect(dbname: str = "space_weather", timeout: int = 10):
    """Read-only-by-convention connection to one of the shared databases."""
    return psycopg2.connect(
        host=_env("SOLARIS_DB_HOST"),
        port=int(_env("SOLARIS_DB_PORT", default="5432")),
        user=_env("SOLARIS_DB_USER"),
        password=_env("SOLARIS_DB_PASSWORD"),
        dbname=dbname,
        connect_timeout=timeout,
    )
