# tvmdbhex

Dominant poster colours for **every movie and TV series on TMDB**: primary,
secondary and tertiary, as hex codes, served over a small HTTP API.

This service stands on its own. It's the only thing that talks to TMDB.
Clients such as **Hoozat** call this API and never call TMDB themselves.

```
TMDB ──(ingester)──> database ──(read-only API + website)──> Hoozat / you
```

The database is SQLite locally, or Postgres when `DATABASE_URL` is set, as it
is on Vercel.

## How it works

1. **Seed**: downloads TMDB's [daily ID exports](https://developer.themoviedb.org/docs/daily-id-exports)
   (`movie_ids_*.json.gz`, `tv_series_ids_*.json.gz`), which list every title,
   and inserts each one as `pending`.
2. **Process**: for each pending title (most popular first) it calls
   `/movie/{id}` or `/tv/{id}` to get `poster_path` and downloads the poster
   (`w185` by default). It then extracts the three dominant colours. Titles that
   share a poster file are only downloaded once.
3. **Sync**: TMDB's `/movie/changes` and `/tv/changes` endpoints re-queue titles
   edited since the last run, so new posters get new colours.

`tvmdbhex run` does all three steps. Run it daily. It is resumable: stop it at
any time and the next run carries on from where it stopped.

### Colour extraction

The poster is downscaled to 64×96, converted to CIELAB (perceptual colour
space) and clustered with k-means (k=8, fixed seed, so results are
reproducible). Clusters are ranked by the share of pixels they cover:

- **primary**: the largest cluster
- **secondary / tertiary**: the next largest clusters that are *visibly
  different* (ΔE ≥ 15) from the colours already picked. Near-identical shades
  of the same black don't take up all three slots.
- Clusters covering less than 2% of the poster are ignored.
- If a poster has fewer than three distinct colours, the slots fall back to
  the next-largest cluster, then repeat the last colour.

Each colour also carries a `ratio` (share of pixels, 0–1). This lets clients
tell a colour that dominates the poster from one that only just made the cut.

## Setup

```bash
cp .env.example .env         # add TMDB_READ_TOKEN (or TMDB_API_KEY) and TVMDBHEX_API_KEYS
pip install -e '.[dev]'

tvmdbhex run                 # seed + sync + process everything (long first run)
tvmdbhex process --limit 500 # or process a batch (most popular first)
tvmdbhex stats               # counts by status
tvmdbhex serve --port 8000   # start the API
```

Or with Docker: `docker compose up -d` starts the API plus an ingester that
runs `tvmdbhex run` once a day, both sharing the same database volume.

| Variable | Default | |
|---|---|---|
| `TMDB_READ_TOKEN` / `TMDB_API_KEY` | – | TMDB credentials (ingester only) |
| `DATABASE_URL` / `POSTGRES_URL` | – | Postgres URL. Takes precedence over the SQLite file |
| `TVMDBHEX_DB_PATH` | `data/tvmdbhex.db` | SQLite file |
| `TVMDBHEX_API_KEYS` | *(empty = no auth)* | Comma-separated keys accepted in `X-API-Key` |
| `TVMDBHEX_CONCURRENCY` | `16` | Parallel workers |
| `TVMDBHEX_REQUESTS_PER_SECOND` | `40` | TMDB API rate cap (TMDB allows ~50/s) |
| `TVMDBHEX_POSTER_SIZE` | `w185` | Poster size downloaded for analysis |
| `TVMDBHEX_INCLUDE_ADULT` | `false` | Also process titles flagged adult |
| `TVMDBHEX_MAX_TITLES` | *(no cap)* | Only ever process the N most popular titles (the Vercel ingest workflow uses 500000). Titles below the cap stay `pending` |

**First run:** TMDB lists roughly 1M+ movies and 200k+ series. At 40 req/s
that takes about 8–9 hours. Popular titles are processed first, so the useful
part of the catalogue is ready early on.

## Deploying on Vercel

Vercel runs the website and API as a serverless function (`app.py`,
`vercel.json`). Vercel functions can't keep a SQLite file or run for hours,
so two pieces live elsewhere:

- **Database:** Postgres. The easiest route is Vercel → Storage → Create
  Database → **Neon**, connected to this project. That sets `DATABASE_URL`
  automatically.
- **Ingester:** a GitHub Actions workflow (`.github/workflows/ingest.yml`)
  that runs `tvmdbhex run` every 6 hours against the same database. The first
  backfill (~9h) spreads over a couple of runs. After that each run takes
  minutes.

One-time setup:

1. **Vercel → Storage:** create a Neon Postgres database and connect it to
   the project. Redeploy once. Until a database is connected, every URL
   returns a 503 explaining this.
2. **Vercel → Settings → Environment Variables** (optional): add
   `TVMDBHEX_API_KEYS` to require an `X-API-Key`.
3. **GitHub → Settings → Secrets and variables → Actions:** add
   `DATABASE_URL` (the same URL Vercel shows for the database) and
   `TMDB_READ_TOKEN` (or `TMDB_API_KEY`).
4. **GitHub → Actions → Ingest posters → Run workflow** to start the first
   backfill now instead of waiting for the schedule. The site fills in as it
   runs, most popular titles first.

Every push to `main` redeploys the site.

**Storage:** the full catalogue (~1.2M rows plus indexes) is roughly
400–500 MB of Postgres. That's around the limit of Neon's free tier, so you
may need a paid plan once the backfill completes.

## Website

`tvmdbhex serve` also serves a browser UI at **`/`** (e.g. http://localhost:8000/)
for looking up any title and its colours:

- With an empty search it shows the **Top 1,000** titles, ranked, with movies
  and TV alternating. Search by title or TMDB ID to find anything else.
  Filter by All / Movies / TV.
- Each card shows the poster, a strip of the three colours sized by pixel
  share, and the hex codes.
- Click a card for the detail view: large swatches with hex codes (click to
  copy), the share of the poster each colour covers, a preview of the colours
  used together, and "Copy JSON".
- Every title has a shareable link, e.g. `/#movie/550` or `/?q=matrix`.
- If `TVMDBHEX_API_KEYS` is set, the page asks for a key once and keeps it in
  that browser.
- The page loads the poster images straight from TMDB's image CDN, so you
  can check the colours against the poster. Turn this off with the
  "Posters" toggle and it shows a gradient of the three colours instead.
  Only the website does this. The API itself never contacts TMDB.

## API

Interactive docs are at `/docs`. Every `/v1` route needs an `X-API-Key` header
when `TVMDBHEX_API_KEYS` is set.

### `GET /v1/{movie|tv}/{tmdb_id}`

```json
{
  "media_type": "movie",
  "tmdb_id": 550,
  "title": "Fight Club",
  "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
  "status": "done",
  "colors": {
    "primary":   { "hex": "#1C1B1D", "ratio": 0.4812 },
    "secondary": { "hex": "#D8C6B4", "ratio": 0.2127 },
    "tertiary":  { "hex": "#C0364A", "ratio": 0.0831 }
  },
  "updated_at": "2026-09-27T10:00:00+00:00"
}
```

(Illustrative values.) `status` is one of `done`, `pending`, `no_poster`,
`not_found` or `error`. `colors` is `null` unless the status is `done`. An ID
that isn't in the database returns `404`.

### `POST /v1/lookup`: batch (up to 500)

```json
{ "items": [ { "media_type": "movie", "tmdb_id": 550 }, { "media_type": "tv", "tmdb_id": 1399 } ] }
```

Returns `{ "results": [...], "missing": [...] }`, with results in request order.

### `GET /v1/top?media_type=movie&limit=60&offset=0`

The most popular titles that have colours, as a ranked list (`rank` field).
Movies and TV alternate: #1 is the most popular movie, #2 the most popular
series, and so on. This is the website's default "Top 1,000" view.

### `GET /v1/search?q=…&media_type=movie&status=done&limit=24&offset=0`

Searches titles by name (substring) or TMDB ID. Exact matches come first,
then the rest by popularity. With an empty `q` it browses by popularity.

### `GET /v1/{movie|tv}?after_id=0&limit=1000&updated_since=…`

Pages through every processed title, ordered by ID. Use it to bulk-sync or
cache the whole dataset: pass the last `tmdb_id` you received as `after_id`.

### `GET /v1/stats`, `GET /health`

Counts by status, and a liveness check.

## Tests

```bash
pytest
# also run the storage tests against a throwaway Postgres database:
TVMDBHEX_TEST_DATABASE_URL=postgresql://postgres:pg@localhost/tvmdbhex_test pytest
```

TMDB is mocked in the tests, so they run offline.
