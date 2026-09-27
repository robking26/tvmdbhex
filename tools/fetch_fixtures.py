"""Download the palette test-set posters from TMDB (run in CI, where TMDB is reachable).

    python tools/fetch_fixtures.py tools/palette_testset.json out_dir

Writes out_dir/<media>_<id>.jpg (w342) and out_dir/manifest.json.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from tvmdbhex.config import Settings
from tvmdbhex.tmdb import TMDBClient


async def main(testset: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env()
    entries = json.loads(testset.read_text())["titles"]
    manifest = []
    async with TMDBClient(settings.tmdb_api_key, settings.tmdb_read_token, 20) as client:
        for e in entries:
            try:
                details = await client.details(e["media_type"], e["id"])
                path = details.get("poster_path")
                if not path:
                    print(f"no poster: {e['name']}")
                    continue
                image = await client.poster(path, "w342")
            except Exception as exc:  # keep going; report what failed
                print(f"failed {e['name']}: {exc}")
                continue
            file = f"{e['media_type']}_{e['id']}.jpg"
            (out / file).write_bytes(image)
            manifest.append({**e, "tmdb_title": details.get("title") or details.get("name"),
                             "poster_path": path, "file": file})
            print(f"ok {e['name']} -> {manifest[-1]['tmdb_title']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)}/{len(entries)} posters saved")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
