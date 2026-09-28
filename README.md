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
   (`w342` by default) and the title's TMDB logo. It then builds the six-role palette. Titles that
   share a poster file are only downloaded once.
3. **Sync**: TMDB's `/movie/changes` and `/tv/changes` endpoints re-queue titles
   edited since the last run, so new posters get new colours.

`tvmdbhex run` does all three steps. Run it daily. It is resumable: stop it at
any time and the next run carries on from where it stopped.

### Colour selection: six-role semantic palette (v3)

Every title gets six colours, each with a job:

```json
{
  "base": "#2CC1FB",
  "identity1": "#EE0DA4",
  "identity2": "#FBBBD8",
  "highlight1": "#D4F0F5",
  "highlight2": "#F49C70",
  "accent": "#8E4C37"
}
```

| Role | Prominence in generated artwork | Comes from |
|---|---|---|
| base | 40–50% | Environment meeting the two upper corners (top ~18% band) |
| identity1 | 25–30% | Main title/logo colour, else the strongest (non-neutral) artwork colour |
| identity2 | 10–20% | Second logo colour, else the next strongest artwork colour |
| highlight1 | 10–20% | Strongest artwork colour not used yet |
| highlight2 | 10–20% | Next unused artwork colour |
| accent | 3–8% | Next unused artwork colour, leaning a little towards vivid and distinctive |

How it works (`tvmdbhex/semantic.py`):

1. **Title treatment.** TMDB's official logo for the title (a transparent PNG,
   English or language-neutral) is fetched with the details request. It is
   *located* on the poster by multi-scale edge matching, with a size-aware
   threshold so small templates can't match random texture. On the 44-poster
   test set this finds 35 logos with no false locations.
   - **Located:** logo colours are the TMDB logo's colours confirmed under the
     letters on the poster. They are topped up with the poster's own letter
     pixels, minus the surrounding background.
   - **Not located** (the poster restacks or restyles the title): the TMDB
     logo's colours are used only if the poster really contains them.
   - **No TMDB logo:** a text-block detector finds the most title-like block of
     letters (31/35 correct, 0 wrong on the test set). The letter colours are
     the ones concentrated inside the block compared with just outside it.

   White, black and neutral grey logo colours don't count. Cream, gold, beige,
   deep navy, dark red and metallic silver do. Near-identical logo colours
   merge into one.
2. **Artwork.** With the title masked out, the poster is clustered in OKLab
   (16 clusters, near-duplicates merged, all near-blacks one black). The
   clusters are ranked into a coverage-driven *dominant ladder*: pixel
   coverage, spatial coverage, coherence and region size first, with a little
   vividness in later slots.
3. **Roles** are filled from two queues: logo colours first for identity,
   otherwise the artwork ladder. Highlights and accent then take the next
   *unused* ladder colours, so the ladder shifts depending on how many
   identity colours the logo supplied. Every assignment must be perceptually
   distinct (OKLab distance) from all earlier roles. The threshold only
   relaxes for genuinely monochrome art; a last-resort tint of base is used
   only for a completely flat image.

The legacy `colors` field (primary/secondary/tertiary) is still returned,
derived from the new roles: identity1, base and accent. Tunables live in
`SemanticConfig`. It is deterministic and takes about 120 ms per poster.

The previous 3-colour algorithm (v2, `tvmdbhex/colors.py`) and the original
area-ranking (v1, `colors_legacy.py`) remain for comparison in the debug tools.

#### Tuning tools

- **Debug view:** set `TVMDBHEX_DEBUG=1` and open `/debug`. Drop in poster
  images, paste TMDB poster paths, or load the top N titles. Each poster
  shows every candidate with its features and role scores, the new picks next
  to the legacy ones, and automatic review flags: palettes that are too
  similar, a neutral beating a distinctive accent, a vivid speck chosen as
  primary, and a lightness drift from the poster.
- **Offline report:** run `python tools/palette_report.py <folder> report.html`
  on any folder of posters.
- **Test set:** `tools/palette_testset.json` lists 44 posters across the
  categories that matter (bright, horror, monochrome, skies, skin tones,
  black backgrounds, single strong accents). The *Palette fixtures* workflow
  downloads them to the `palette-fixtures` branch.

The legacy algorithm (area ranking) stays in `tvmdbhex/colors_legacy.py`,
for comparison only.

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
| `TVMDBHEX_POSTER_SIZE` | `w342` | Poster size downloaded for analysis (logos are fetched at w300) |
| `TVMDBHEX_INCLUDE_ADULT` | `false` | Also process titles flagged adult |
| `TVMDBHEX_DEBUG` | `false` | Enables `/debug` and `/v1/debug/*` palette tuning tools |
| `TVMDBHEX_CATALOGUE_SIZE` | *(unlimited)* | Store only the N most popular titles; prune uncoloured ones below it (workflow: 500000) |
| `TVMDBHEX_MAX_TITLES` | *(no cap)* | A run only processes titles ranked in the top N (`--top`; never shrinks the catalogue) |

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
