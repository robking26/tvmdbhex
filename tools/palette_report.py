"""Static HTML report comparing the legacy and current palette algorithms.

    python tools/palette_report.py <fixtures_dir> report.html

<fixtures_dir> holds poster images plus an optional manifest.json (as written
by tools/fetch_fixtures.py, including TMDB logo files); without a manifest every
image in the folder is used (no logos, so titles are found by text detection).
For each poster the report shows the poster, every candidate cluster with its
features and role scores, legacy vs new picks, and automatic review flags.
"""
from __future__ import annotations

import base64
import html
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvmdbhex.palette_debug import compare  # noqa: E402

ROLES = ("primary", "secondary", "tertiary")
SEMANTIC = ("base", "identity1", "identity2", "highlight1", "highlight2", "accent")


def _swatch(hex_: str, label: str = "", big: bool = False) -> str:
    size = "56px" if big else "26px"
    return (
        f'<div class="sw"><i style="background:{hex_};width:{size};height:{size}"></i>'
        f'<span>{html.escape(label)}<br><code>{hex_}</code></span></div>'
    )


def render(entries: list[dict], out: Path) -> None:
    rows, summary = [], {"flagged": 0, "changed_primary": 0, "ms": []}
    for e in entries:
        img = Image.open(e["path"]).convert("RGB")
        logo = Image.open(e["logo"]) if e.get("logo") else None
        t = time.perf_counter()
        r = compare(img, logo)
        summary["ms"].append((time.perf_counter() - t) * 1000)
        cur, leg, sem = r["current"], r["legacy"], r["semantic"]
        if r["flags"]:
            summary["flagged"] += 1
        if cur["palette"]["primary"]["hex"] != leg["primary"]["hex"]:
            summary["changed_primary"] += 1
        thumb = img.copy()
        thumb.thumbnail((160, 240))
        buf = __import__("io").BytesIO()
        thumb.save(buf, "JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        cand_rows = "".join(
            f"<tr class='{c['role'] or ''}'><td><i class='chip' style='background:{c['hex']}'></i><code>{c['hex']}</code></td>"
            f"<td>{c['coverage']:.1%}</td><td>{c['lightness']:.2f}</td><td>{c['chroma']:.3f}</td>"
            f"<td>{c['saturation']:.2f}</td><td>{c['distinctiveness']:.2f}</td><td>{c['contrast']:.2f}</td>"
            f"<td>{c['neutrality']:.2f}</td><td>{c['neutral_factor']:.2f}</td>"
            + "".join(f"<td>{c['scores'].get(role, float('nan')):.3f}</td>" for role in ROLES)
            + f"<td>{c['role'] or ''}</td></tr>"
            for c in cur["candidates"]
        )
        strip = "".join(
            f"<span style='background:{c['hex']};flex:{max(c['coverage'], 0.01)}' title='{c['hex']} {c['coverage']:.1%}'></span>"
            for c in cur["candidates"]
        )
        flags = "".join(f"<li>{html.escape(f)}</li>" for f in r["flags"]) or "<li class='ok'>no flags</li>"
        rows.append(
            f"""<section>
  <img src="data:image/jpeg;base64,{b64}" alt="">
  <div class="main">
    <h2>{html.escape(e.get('name', Path(e['path']).stem))} <small>{html.escape(e.get('category', ''))}</small></h2>
    {f"<p class='expect'>Expected: {html.escape(e['expect'])}</p>" if e.get('expect') else ''}
    <div class="pal"><b>v3</b>{''.join(_swatch(sem['palette'][role], f"{role} ({sem['sources'][role]})", True) for role in SEMANTIC)}</div>
    <p class="meta">title treatment: {sem['logo']['source']} · logo colours: {', '.join(c['hex'] for c in sem['logo']['colours']) or 'none usable'} · ladder: {' '.join(c['hex'] for c in sem['ladder'])}</p>
    <div class="pal old"><b>v2</b>{''.join(_swatch(cur['palette'][role]['hex'], role) for role in ROLES)}</div>
    <div class="pal old"><b>v1</b>{''.join(_swatch(leg[role]['hex'], role) for role in ROLES)}</div>
    <div class="strip">{strip}</div>
    <ul class="flags">{flags}</ul>
    <p class="meta">colourfulness {cur["colourfulness"]:.3f} · colour strength {cur["colour_strength"]:.2f}</p>
    <details><summary>{len(cur['candidates'])} candidates</summary>
    <table><tr><th>colour</th><th>cov</th><th>L</th><th>C</th><th>sat</th><th>distinct</th><th>contrast</th><th>neutral</th><th>factor</th><th>P</th><th>S</th><th>T</th><th>role</th></tr>{cand_rows}</table>
    </details>
  </div>
</section>"""
        )
    ms = summary["ms"]
    head = (
        f"{len(entries)} posters · primary changed on {summary['changed_primary']} · "
        f"{summary['flagged']} flagged · {sum(ms) / len(ms):.0f} ms avg (incl. legacy)"
    )
    out.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Palette report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font:14px/1.4 system-ui,sans-serif;margin:0;padding:16px;background:#f4f4f5;color:#18181b}}
h1{{font-size:20px;margin:0 0 4px}} .head{{color:#555;margin-bottom:16px}}
section{{display:flex;gap:16px;background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px;margin-bottom:12px}}
section>img{{width:160px;height:240px;object-fit:cover;border-radius:6px;flex:none}}
.main{{flex:1;min-width:0}} h2{{font-size:16px;margin:0 0 4px}} h2 small{{color:#777;font-weight:400}}
.expect{{margin:0 0 6px;color:#555;font-style:italic}}
.pal{{display:flex;gap:12px;align-items:center;margin:4px 0}} .pal b{{width:34px}}
.sw{{display:flex;gap:6px;align-items:center;font-size:12px}} .sw i{{border-radius:6px;border:1px solid #0002;display:block}}
.strip{{display:flex;height:14px;border-radius:4px;overflow:hidden;margin:8px 0}} .strip span{{display:block}}
.flags{{margin:4px 0;padding-left:18px;color:#b42318}} .flags .ok{{color:#067647}}
.meta{{color:#777;font-size:12px;margin:2px 0}}
table{{border-collapse:collapse;font-size:12px;margin-top:6px}} td,th{{padding:2px 6px;border-bottom:1px solid #eee;text-align:right}}
td:first-child{{text-align:left;white-space:nowrap}} .chip{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:4px;vertical-align:-2px;border:1px solid #0002}}
tr.primary{{background:#fef3c7}} tr.secondary{{background:#e0f2fe}} tr.tertiary{{background:#ede9fe}}
</style></head><body><h1>Palette report</h1><div class="head">{head}</div>{''.join(rows)}</body></html>"""
    )
    print(head)


def main(src: Path, out: Path) -> None:
    manifest = src / "manifest.json"
    if manifest.exists():
        entries = [
            {**e, "path": src / e["file"], "logo": src / e["logo_file"] if e.get("logo_file") else None}
            for e in json.loads(manifest.read_text())
        ]
    else:
        entries = [{"path": p} for p in sorted(src.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    render(entries, out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
