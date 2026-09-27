import pytest

from tvmdbhex.config import Settings, parse_database_url

URL = "postgresql://user:p%40ss@ep-cool-123-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"


@pytest.mark.parametrize(
    "raw",
    [
        URL,
        f"  {URL}\n",
        f'"{URL}"',
        f"DATABASE_URL={URL}",
        f"DATABASE_URL='{URL}'",
        # The whole .env.local snippet copied from the Vercel/Neon dashboard.
        f"# Recommended for most uses\nDATABASE_URL={URL}\n\n"
        "# For uses requiring a connection without pgbouncer\n"
        "DATABASE_URL_UNPOOLED=postgresql://user:x@ep-cool-123.us-east-1.aws.neon.tech/neondb\n"
        "PGHOST=ep-cool-123-pooler.us-east-1.aws.neon.tech\n",
    ],
)
def test_parse_database_url_accepts_pasted_values(raw):
    assert parse_database_url(raw) == URL


def test_unset_means_sqlite():
    assert parse_database_url(None) == ""
    assert parse_database_url("  ") == ""


def test_garbage_fails_loudly_instead_of_becoming_a_sqlite_path():
    with pytest.raises(ValueError, match="postgres"):
        parse_database_url("neondb")


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"DATABASE_URL={URL}")
    assert Settings.from_env().db_target == URL
