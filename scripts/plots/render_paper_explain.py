"""Render explanation in the paper's dark-terminal A-E layout.

Order is FIXED and must not change:
  A. Station Info  +  Forecast Summary (7-day, hourly)  + Largest line
  B. Cross-KPI Causal Influences (PAX-TS + GraphRAG)  +  3GPP Causal Chains
     +  Cross-Channel Sensitivity Matrix (7x7)
  C. Anomaly Root-Cause Analysis (1/2)  +  (2/2)
  D. Actionable Recommendations
  E. Numerical Fidelity Check (KPI metrics + sensitivity comparison)

Stage 1 (this file): mock data drives the template so layout is verifiable
without pipeline changes. Stage 2 will swap mock -> result.json + pipeline CSV.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from html import escape
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Mock payload (matches the screenshot's J_9 figure: DL_rBler -37.59%, etc.)
# ---------------------------------------------------------------------------
MOCK = {
    "station": {
        "id": "station_J_9",
        "location": "Johnson County, TX",
        "environment": "RMa (Rural Macro)",
        "nearest_road": "County Road 1203",
        "area_type": "aeroway/aerodrome",
        "explainer_model": "GPT-4o (ReAct, 24 tool-call budget)",
        "forecast_backend": "Chronos-2 (amazon/chronos-2)",
        "elapsed": "40.65 s  (PAX-TS + ReAct report)",
    },
    "forecast": [
        {
            "kpi": "RRC_Conn",
            "fc_mean": "2.0952",
            "slope": "+0.0610",
            "base": "2.2738",
            "chg": "-7.85%",
            "arrow": "down",
        },
        {
            "kpi": "DL_CQI",
            "fc_mean": "7.9940",
            "slope": "+0.0476",
            "base": "8.3393",
            "chg": "-4.14%",
            "arrow": "up",
        },
        {
            "kpi": "DL_iBler",
            "fc_mean": "6.3014",
            "slope": "-0.2289",
            "base": "6.2849",
            "chg": "+0.26%",
            "arrow": "up",
        },
        {
            "kpi": "DL_rBler",
            "fc_mean": "0.1421",
            "slope": "+0.0076",
            "base": "0.2277",
            "chg": "-37.59%",
            "arrow": "up",
        },
        {
            "kpi": "MAC_DL_Eff",
            "fc_mean": "12837.4524",
            "slope": "-835.7887",
            "base": "11609.7500",
            "chg": "+10.57%",
            "arrow": "up",
        },
        {
            "kpi": "PRB_Util",
            "fc_mean": "14.0833",
            "slope": "-1.0134",
            "base": "12.5536",
            "chg": "+12.19%",
            "arrow": "down",
        },
        {
            "kpi": "Throughput",
            "fc_mean": "13723.9345",
            "slope": "-908.5789",
            "base": "12955.1548",
            "chg": "+5.93%",
            "arrow": "down",
        },
    ],
    "largest": "Largest: DL_rBler -37.59%, DL_iBler near-stable +0.26%",
    "causal_top": [
        ("DL_rBler -> Throughput", "97.0816", "red"),
        ("DL_rBler -> MAC_DL_Eff", "80.7582", "red"),
        ("RRC_Conn -> Throughput", "29.0735", "yellow"),
        ("RRC_Conn -> MAC_DL_Eff", "21.9984", "yellow"),
        ("DL_CQI   -> Throughput", "13.4353", "yellow"),
    ],
    "graphrag_chains": [
        ("rBler->Thpt", "iBler↑->HARQ↑->rBler↑->RLC↑->Thpt↓"),
        ("RRC->Thpt", "RRC↑->PRB↑->Latency↑->Thpt↓ (per UE)"),
        ("CQI->Thpt", "CQI↓->MAC_Eff↓->iBler↑->rBler↑->Thpt↓"),
    ],
    "sensitivity_matrix": {
        "labels": ["RRC", "CQI", "iBler", "rBler", "MAC_Eff", "PRB", "Thpt"],
        "rows": [
            ["", "0.0078", "0.0040", "0.0085", "0.0002", "21.9984", "0.0142", "29.0735"],
            ["", "0.0010", "0.0089", "0.0047", "0.0001", "11.0128", "0.0074", "13.4353"],
            ["", "0.0003", "0.0089", "0.0071", "0.0000", "4.3823", "0.0030", "5.3038"],
            ["", "0.0050", "0.0125", "0.0271", "0.0056", "80.7582", "0.0541", "97.0816"],
            ["", "0.0001", "0.0001", "0.0000", "0.0000", "0.0000", "0.0000", "0.0019"],
            ["", "0.0000", "0.0002", "0.0005", "0.0000", "1.3900", "0.0063", "1.4861"],
            ["", "0.0001", "0.0000", "0.0000", "0.0000", "0.0037", "0.0000", "0.0081"],
        ],
    },
    "anomaly_left": [
        {
            "title": "DL_rBler: -37.59% Decrease",
            "lines": [
                ("Forecast: 37.59% decrease from baseline 0.2277", "[Forecast]"),
                ("Driver: DL_rBler->Thpt sens=97.08", "[Sensitivity]"),
                ("3GPP: iBler↑->HARQ↑->rBler↑->RLC↑->Thpt↓", "[GraphRAG]"),
                ("rBler decrease -> reduced retx overhead -> improved thpt", None),
                ("Rural macro, low POI density supports trend", "[Spatial]"),
            ],
        },
        {
            "title": "MAC_DL_Eff: +10.57% Increase",
            "lines": [
                ("Forecast: 10.57% increase from baseline 11609.7500", "[Forecast]"),
                ("Driver: RRC->MAC_Eff sens=21.99", "[Sensitivity]"),
                ("RRC decrease (-7.85%) -> reduced congestion -> improved MAC eff", None),
            ],
        },
    ],
    "anomaly_right": [
        {
            "title": "PRB_Util: +12.19% Increase",
            "lines": [
                ("Forecast: 12.19% increase from baseline 12.5536", "[Forecast]"),
                ("RRC_Conn->PRB_Util sens=0.014: user count is not the driver", None),
                ("Fewer users -> scheduler allocates more PRBs per UE", "[3GPP]"),
            ],
        },
        {
            "title": "Throughput: +5.93% Increase",
            "lines": [
                ("Forecast: 5.93% increase from baseline 12955.1548", "[Forecast]"),
                ("Driver: DL_rBler decrease -> reduced retx overhead", None),
                ("Consistent with rBler -37.59% & sens=97.08", None),
            ],
        },
    ],
    "actions": [
        {
            "n": "1",
            "when": "Day 2-4, 03:00-06:00 (rBler trough)",
            "action": "Tune OLLA iBler target; verify maxHARQ-Tx=4 keeps rBLER < 0.15%",
            "evidence": "rBler->Thpt sens=97.08; rBler -37.59% sustains thpt gain",
        },
        {
            "n": "2",
            "when": "Day 1, 00:00-09:00 (RRC idle window)",
            "action": "Adjust RRC inactivity timer t310/t311 to reduce idle UE overhead",
            "evidence": "RRC->MAC_Eff sens=21.99; RRC -7.85% -> MAC_Eff +10.57%",
        },
        {
            "n": "3",
            "when": "Day 5-7, 22:00-23:59 (PRB peak)",
            "action": "Review proportional fair scheduler PRB weights under low-load (PRB_Util<15%)",
            "evidence": "RRC->PRB sens=0.014 (decoupled); PRB +12.19% despite RRC -7.85%",
        },
    ],
    "fidelity_left": [
        ("RRC_Conn", "mean", "2.0952", "2.0952", "0.00", "OK"),
        ("RRC_Conn", "slope", "0.0610", "0.0610", "0.00", "OK"),
        ("RRC_Conn", "base", "2.2738", "2.2738", "0.00", "OK"),
        ("RRC_Conn", "chg%", "-7.85", "-7.8534", "0.04", "OK"),
        ("DL_CQI", "mean", "7.9940", "7.9940", "0.00", "OK"),
        ("DL_CQI", "slope", "0.0476", "0.0476", "0.00", "OK"),
        ("DL_CQI", "base", "8.3393", "8.3393", "0.00", "OK"),
        ("DL_CQI", "chg%", "-4.14", "-4.1399", "0.00", "OK"),
        ("DL_iBler", "mean", "6.3014", "6.3014", "0.00", "OK"),
        ("DL_iBler", "slope", "-0.2289", "-0.2289", "0.00", "OK"),
        ("DL_iBler", "base", "6.2849", "6.2849", "0.00", "OK"),
        ("DL_iBler", "chg%", "+0.26", "+0.2623", "0.00", "OK"),
        ("DL_rBler", "mean", "0.1421", "0.1421", "0.00", "OK"),
        ("DL_rBler", "slope", "0.0076", "0.0076", "0.00", "OK"),
    ],
    "fidelity_right": [
        ("DL_rBler", "base", "0.2277", "0.2277", "0.00", "OK"),
        ("DL_rBler", "chg%", "-37.59", "-37.5883", "0.00", "OK"),
        ("MAC_DL_Eff", "mean", "12837.4524", "12837.4524", "0.00", "OK"),
        ("MAC_DL_Eff", "slope", "-835.7887", "-835.7887", "0.00", "OK"),
        ("MAC_DL_Eff", "base", "11609.7500", "11609.7500", "0.00", "OK"),
        ("MAC_DL_Eff", "chg%", "+10.57", "+10.5748", "0.04", "OK"),
        ("PRB_Util", "mean", "14.0833", "14.0833", "0.00", "OK"),
        ("PRB_Util", "slope", "-1.0134", "-1.0134", "0.00", "OK"),
        ("PRB_Util", "base", "12.5536", "12.5536", "0.00", "OK"),
        ("PRB_Util", "chg%", "+12.19", "+12.1859", "0.03", "OK"),
        ("Throughput", "mean", "13723.9345", "13723.9345", "0.00", "OK"),
        ("Throughput", "slope", "-908.5789", "-908.5789", "0.00", "OK"),
        ("Throughput", "base", "12955.1548", "12955.1548", "0.00", "OK"),
        ("Throughput", "chg%", "+5.93", "+5.9342", "0.07", "OK"),
    ],
    "fidelity_sens_left": [
        ("rBler->Thpt", "97.0816", "97.0816", "OK"),
        ("rBler->MAC", "80.7582", "80.7582", "OK"),
        ("RRC->Thpt", "29.0735", "29.0735", "OK"),
    ],
    "fidelity_sens_right": [
        ("RRC->MAC", "21.9984", "21.9984", "OK"),
        ("CQI->Thpt", "13.4353", "13.4353", "OK"),
    ],
    "fidelity_result": "Result: 33/33 match  (28 forecast metrics + 5 sensitivity scores verified against pipeline CSV)",
}


# ---------------------------------------------------------------------------
# CSS — terminal palette
# ---------------------------------------------------------------------------
CSS = """
@page { size: 11in 17in; margin: 0.3in; }
@media print { body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
:root {
  --bg: #0d0f12;
  --panel: #14181d;
  --border: #ff3b3b;
  --header: #ff8a3a;
  --text: #d6d6d6;
  --muted: #7a8089;
  --green: #6cff6c;
  --red: #ff5050;
  --yellow: #ffd24a;
  --cyan: #6fe0d0;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', 'DejaVu Sans Mono', 'Menlo', monospace;
  font-size: 12.5px;
  line-height: 1.35;
}
.page {
  width: 1080px;
  margin: 0 auto;
  padding: 14px;
}
.page p, .page li, .page td, .page th, .page div { hyphens: none; }
.section {
  position: relative;
  border: 1.5px solid var(--border);
  background: var(--panel);
  margin-bottom: 10px;
  padding: 10px 12px 12px 12px;
  overflow: hidden;
}
.row { display: flex; gap: 14px; }
.col { flex: 1; min-width: 0; overflow: hidden; }
.h { color: var(--header); font-weight: 600; margin: 0 0 6px 0; }
table { width: 100%; border-collapse: collapse; table-layout: auto; }
th, td {
  padding: 2px 8px; text-align: right;
  word-break: break-word; overflow-wrap: anywhere;
}
th { color: var(--header); font-weight: 600; border-bottom: 1px solid #333; white-space: nowrap; }
td.l, th.l { text-align: left; }
td.kpi, td.label { color: var(--cyan); white-space: nowrap; }
td.num, th.num { white-space: nowrap; }
.green { color: var(--green); }
.red { color: var(--red); }
.yellow { color: var(--yellow); }
.muted { color: var(--muted); }
.kv { display: grid; grid-template-columns: 130px 1fr; row-gap: 2px; }
.kv .k { color: var(--muted); }
.kv .v { color: var(--text); }
.note { color: var(--text); margin-top: 6px; font-size: 12px; }
.bullets { margin: 4px 0 8px 0; padding-left: 14px; }
.bullets li { margin-bottom: 2px; }
.title { color: var(--yellow); font-weight: 600; margin-top: 6px; }
.grid7 td, .grid7 th { padding: 2px 6px; font-size: 12px; }
.matrix-key { color: var(--muted); margin-top: 4px; font-size: 11.5px; }
.actions th { text-align: left; }
.actions td { text-align: left; }
.actions td.idx { text-align: center; width: 28px; color: var(--yellow); }
.fidelity th, .fidelity td { padding: 1px 6px; font-size: 11.5px; }
.ok { color: var(--green); }
.result-line {
  margin-top: 6px;
  color: var(--green);
  font-weight: 600;
}
"""


# ---------------------------------------------------------------------------
# Section renderers (order is fixed: A -> B -> C -> D -> E)
# ---------------------------------------------------------------------------
def render_a(d: dict[str, Any]) -> str:
    s = d["station"]
    rows = "".join(
        f"<tr><td class='l kpi'>{escape(r['kpi'])}</td>"
        f"<td>{escape(r['fc_mean'])}</td>"
        f"<td>{escape(r['slope'])}</td>"
        f"<td>{escape(r['base'])}</td>"
        f"<td class='{ 'green' if r['chg'].startswith('+') else 'red' }'>"
        f"{escape(r['chg'])} {'▲' if r['arrow']=='up' else '▼'}</td></tr>"
        for r in d["forecast"]
    )
    return f"""
    <section class="section">
      <div class="row">
        <div class="col">
          <div class="h">Station Info</div>
          <div class="kv">
            <div class="k">Station</div><div class="v">{escape(s['id'])}  ({escape(s['location'])})</div>
            <div class="k green">Environment</div><div class="v">{escape(s['environment'])}</div>
            <div class="k yellow">Nearest Road</div><div class="v">{escape(s['nearest_road'])}</div>
            <div class="k">Area Type</div><div class="v">{escape(s['area_type'])}</div>
            <div class="k">Explainer Model</div><div class="v">{escape(s['explainer_model'])}</div>
            <div class="k">Forecast Backend</div><div class="v">{escape(s['forecast_backend'])}</div>
            <div class="k">Elapsed</div><div class="v">{escape(s['elapsed'])}</div>
          </div>
        </div>
        <div class="col">
          <div class="h">Forecast Summary  (7-day, hourly)</div>
          <table>
            <tr><th class="l">KPI</th><th>FcMean</th><th>Slope/d</th><th>Base</th><th>Chg%</th></tr>
            {rows}
          </table>
        </div>
      </div>
      <div class="note red">{escape(d['largest'])}</div>
    </section>
    """


def render_b(d: dict[str, Any]) -> str:
    causal_rows = "".join(
        f"<tr><td>{i+1}</td><td class='l kpi'>{escape(src)}</td>"
        f"<td class='{cls}'>{escape(score)}</td></tr>"
        for i, (src, score, cls) in enumerate(d["causal_top"])
    )
    chain_rows = "".join(
        f"<div><span class='yellow'>{escape(name)}</span> "
        f"<span class='muted'>\"{escape(body)}\"</span></div>"
        for name, body in d["graphrag_chains"]
    )
    labels = d["sensitivity_matrix"]["labels"]
    head = (
        "<tr><th class='l'>src\\tgt</th>"
        + "".join(f"<th>{escape(lbl)}</th>" for lbl in labels)
        + "</tr>"
    )
    matrix_rows = ""
    for src, row in zip(labels, d["sensitivity_matrix"]["rows"]):
        cells = ""
        for v in row[1:]:
            try:
                num = float(v)
            except ValueError:
                num = 0.0
            cls = "muted" if num < 1 else ("yellow" if num < 50 else "red")
            cells += f"<td class='{cls}'>{escape(v)}</td>"
        matrix_rows += f"<tr><td class='l kpi'>{escape(src)}</td>{cells}</tr>"

    return f"""
    <section class="section">
      <div class="row">
        <div class="col">
          <div class="h">Cross-KPI Causal Influences  (PAX-TS + GraphRAG)</div>
          <table>
            <tr><th>#</th><th class="l">Source -> Target</th><th>Sensitivity</th></tr>
            {causal_rows}
          </table>
          <div class="title">3GPP Causal Chains (GraphRAG):</div>
          {chain_rows}
        </div>
        <div class="col">
          <div class="h">Cross-Channel Sensitivity Matrix  (7x7)</div>
          <table class="grid7">
            {head}
            {matrix_rows}
          </table>
          <div class="matrix-key">Key: <span class="red">>=50</span>  <span class="yellow">>=10</span>  >=1  |  <span class="muted"><1</span></div>
        </div>
      </div>
    </section>
    """


def render_anomaly_block(blocks: list[dict[str, Any]]) -> str:
    out = ""
    for b in blocks:
        items = ""
        for text, tag in b["lines"]:
            tag_html = f" <span class='muted'>{escape(tag)}</span>" if tag else ""
            items += f"<li>{escape(text)}{tag_html}</li>"
        out += f"<div class='title red'>{escape(b['title'])}</div><ul class='bullets'>{items}</ul>"
    return out


def render_c(d: dict[str, Any]) -> str:
    right = d["anomaly_right"]
    has_right = bool(right) and not (
        len(right) == 1 and right[0].get("title") in ("—", "(no further events)")
    )
    if has_right:
        return f"""
    <section class="section">
      <div class="row">
        <div class="col">
          <div class="h">Anomaly Root-Cause Analysis  (1/2)</div>
          {render_anomaly_block(d['anomaly_left'])}
        </div>
        <div class="col">
          <div class="h">Anomaly Root-Cause Analysis  (2/2)</div>
          {render_anomaly_block(right)}
        </div>
      </div>
    </section>
    """
    return f"""
    <section class="section">
      <div class="h">Anomaly Root-Cause Analysis</div>
      {render_anomaly_block(d['anomaly_left'])}
    </section>
    """


def render_d(d: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td class='idx'>{escape(a['n'])}</td>"
        f"<td class='yellow'>{escape(a['when'])}</td>"
        f"<td>{escape(a['action'])}</td>"
        f"<td class='muted'>{escape(a['evidence'])}</td></tr>"
        for a in d["actions"]
    )
    return f"""
    <section class="section">
      <div class="h">Actionable Recommendations</div>
      <table class="actions">
        <tr><th>#</th><th>When (Anomaly Window)</th><th>Action (3GPP-grounded)</th><th>Evidence</th></tr>
        {rows}
      </table>
    </section>
    """


def render_e(d: dict[str, Any]) -> str:
    def fid_rows(rows):
        out = ""
        for kpi, metric, rep, pipe, err, ok in rows:
            out += (
                f"<tr><td class='l kpi'>{escape(kpi)}</td>"
                f"<td class='l'>{escape(metric)}</td>"
                f"<td>{escape(rep)}</td><td>{escape(pipe)}</td>"
                f"<td>{escape(err)}</td><td class='ok'>{escape(ok)}</td></tr>"
            )
        return out

    def sens_rows(rows):
        out = ""
        for metric, value, criterion, ok in rows:
            out += (
                f"<tr><td class='l kpi'>{escape(metric)}</td>"
                f"<td>{escape(value)}</td>"
                f"<td class='l muted'>{escape(criterion)}</td>"
                f"<td class='ok'>{escape(ok)}</td></tr>"
            )
        return out

    return f"""
    <section class="section">
      <div class="h">Numerical Fidelity Check</div>
      <div class="row">
        <div class="col">
          <table class="fidelity">
            <tr><th class="l">KPI</th><th class="l">Metric</th><th>Report</th><th>Pipeline</th><th>Err%</th><th>OK</th></tr>
            {fid_rows(d['fidelity_left'])}
          </table>
        </div>
        <div class="col">
          <table class="fidelity">
            <tr><th class="l">KPI</th><th class="l">Metric</th><th>Report</th><th>Pipeline</th><th>Err%</th><th>OK</th></tr>
            {fid_rows(d['fidelity_right'])}
          </table>
        </div>
      </div>
      <div class="row" style="margin-top:8px;">
        <div class="col">
          <table class="fidelity">
            <tr><th class="l">Sensitivity Metric</th><th>Value</th><th class="l">Criterion</th><th>OK</th></tr>
            {sens_rows(d['fidelity_sens_left'])}
          </table>
        </div>
        <div class="col">
          <table class="fidelity">
            <tr><th class="l">Sensitivity Metric</th><th>Value</th><th class="l">Criterion</th><th>OK</th></tr>
            {sens_rows(d['fidelity_sens_right'])}
          </table>
        </div>
      </div>
      <div class="result-line">{escape(d['fidelity_result'])}</div>
    </section>
    """


def render_html(payload: dict[str, Any]) -> str:
    body = (
        render_a(payload)
        + render_b(payload)
        + render_c(payload)
        + render_d(payload)
        + render_e(payload)
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>TelcoAgent Paper Explain</title>
<style>{CSS}</style></head>
<body><div class="page">{body}</div></body></html>
"""


def _chrome() -> str:
    chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if not chrome:
        raise RuntimeError("No Chrome/Chromium found in PATH")
    return chrome


def capture_png(html_path: Path, png_path: Path, width: int = 1100, height: int = 2400) -> None:
    subprocess.run(
        [
            _chrome(),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            f"--screenshot={png_path}",
            html_path.absolute().as_uri(),
        ],
        check=True,
    )
    _trim_png_bottom(png_path)


def _trim_png_bottom(png_path: Path, bg_threshold: int = 25, pad: int = 12) -> None:
    """Crop trailing background-only rows so PNG height = content height.

    The renderer fixes the viewport at a generous ``height`` so all
    sections fit on stations with many anomalies. For stations with
    fewer events the bottom of the PNG is solid background — distracting
    when embedded in a paper. This pass walks rows from the bottom and
    finds the last row containing a pixel brighter than the dark panel
    background, then crops everything below it (plus a small pad).
    """
    import numpy as np
    from PIL import Image

    im = Image.open(png_path).convert("RGB")
    arr = np.asarray(im)
    luma = arr.max(axis=2)  # max channel as a quick "is non-bg" proxy
    non_bg = luma > bg_threshold
    row_has_content = non_bg.any(axis=1)
    if not row_has_content.any():
        return
    last_row = int(np.argwhere(row_has_content).max())
    new_height = min(arr.shape[0], last_row + 1 + pad)
    if new_height >= arr.shape[0]:
        return
    im.crop((0, 0, arr.shape[1], new_height)).save(png_path)


def capture_pdf(html_path: Path, pdf_path: Path, content_px_height: int | None = None) -> None:
    """Render the HTML to a print-ready PDF via Chrome headless.

    If ``content_px_height`` is given (typically passed from the trimmed
    PNG), the function rewrites the HTML's ``@page`` rule to a single
    sheet exactly tall enough for the content (px → in at 96 DPI) so
    the resulting PDF has no trailing whitespace. Otherwise the static
    ``@page`` rule baked into the CSS is used.
    """
    target = html_path
    if content_px_height is not None:
        height_in = content_px_height / 96.0 + 0.4  # small bottom margin
        html = html_path.read_text()
        html = re.sub(
            r"@page\s*\{[^}]*\}",
            f"@page {{ size: 11in {height_in:.2f}in; margin: 0.3in; }}",
            html,
            count=1,
        )
        target = html_path.with_name(html_path.stem + "_pdf.html")
        target.write_text(html)
    subprocess.run(
        [
            _chrome(),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            target.absolute().as_uri(),
        ],
        check=True,
    )
    if target != html_path:
        target.unlink(missing_ok=True)


import re


def _parse_md_section(md: str, header: str) -> str:
    pat = re.compile(rf"###\s+{re.escape(header)}.*?(?=\n###|\Z)", re.DOTALL)
    m = pat.search(md)
    return m.group(0) if m else ""


def _parse_md_table(block: str) -> list[list[str]]:
    rows = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if set(stripped.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows[1:] if rows else []  # drop header


def _classify_environment(area_type: str, env_hint: str) -> str:
    a = (area_type or "").lower()
    if "aeroway" in a or "rural" in (env_hint or "").lower():
        return "RMa (Rural Macro)"
    if "highway" in a or "primary" in a or "secondary" in a:
        return "UMa (Urban Macro - Highway)"
    if "urban" in a or "residential" in a or "commercial" in a:
        return "UMa (Urban Macro)"
    return "UMa (Mixed)"


CORE_KPI_NAMES = (
    "RRC_Conn",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC_DL_Eff",
    "PRB_Util",
    "Throughput",
)
KPI_LABELS_SHORT = ("RRC", "CQI", "iBler", "rBler", "MAC_Eff", "PRB", "Thpt")


def _load_pax_ts_matrix(pax_ts_dir: Path) -> tuple[list[list[str]], list[tuple[str, str, str]]]:
    """Return (7x7 cell rows, top-5 pairs).

    Reads ``sensitivity_global.csv``, aggregates per (source,target) by max
    |S_norm| across perturbation types, scales to ``S_norm * 100`` for
    display (matches the paper figure's 0–100 magnitude convention), and
    builds:
      * a 7×7 matrix in :data:`CORE_KPI_NAMES` order (diagonal left at 0)
      * a top-5 list of (source -> target, formatted score, css class)
    """
    csv_path = pax_ts_dir / "sensitivity_global.csv"
    rows: dict[tuple[str, str], float] = {}
    with csv_path.open() as fh:
        header = fh.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        for line in fh:
            parts = line.rstrip("\n").split(",")
            src = parts[idx["source"]]
            tgt = parts[idx["target"]]
            try:
                s_norm = abs(float(parts[idx["S_norm"]]))
            except ValueError:
                continue
            key = (src, tgt)
            if s_norm > rows.get(key, 0.0):
                rows[key] = s_norm

    # Build 7×7 matrix
    matrix: list[list[str]] = []
    for src in CORE_KPI_NAMES:
        row = [""]
        for tgt in CORE_KPI_NAMES:
            v = rows.get((src, tgt), 0.0) * 100.0
            row.append(f"{v:.4f}")
        matrix.append(row)

    # Top-5 pairs by magnitude
    pairs_sorted = sorted(rows.items(), key=lambda kv: kv[1], reverse=True)
    top5: list[tuple[str, str, str]] = []
    for (src, tgt), val in pairs_sorted[:5]:
        scaled = val * 100.0
        cls = "red" if scaled >= 50 else ("yellow" if scaled >= 10 else "muted")
        top5.append((f"{src} -> {tgt}", f"{scaled:.4f}", cls))
    return matrix, top5


_RCA_KEEP_PREFIXES = ("forecast", "driver", "3gpp", "spatial")


def _parse_anomaly_rca(md: str) -> list[dict[str, Any]]:
    """Parse §4 RCA — keep only the 3-4 most informative bullets per event.

    Each event keeps the Forecast / Driver / 3GPP / Spatial bullets (skipping
    Mechanism and Confidence which are restatements) and trims the prose to
    a single clause. If multiple events share the same KPI + same event-hour
    pattern (e.g. five RRC_Conn spikes all at 08:00-09:00), the duplicates
    are folded into a single entry with a "(also Day X, Y, Z)" suffix on
    the title — preventing the C section from looking like a copy-paste.
    """
    sec = re.search(r"## §4 Anomaly Root-Cause Analysis(.*?)(?=\n## |\Z)", md, re.DOTALL)
    if not sec:
        return []
    body = sec.group(1)
    events: list[dict[str, Any]] = []
    blocks = re.split(r"\n\*\*", body)
    for blk in blocks:
        m = re.match(r"([^*]+?)\*\*", blk)
        if not m:
            continue
        title = m.group(1).strip().lstrip("*").rstrip("*").strip()
        if not title:
            continue
        if "stable" in title.lower() or "no anomaly events" in title.lower():
            continue
        lines: list[tuple[str, str | None]] = []
        for ln in blk.splitlines()[1:]:
            ln = ln.strip()
            if not ln.startswith("- "):
                continue
            content = ln[2:]
            label_match = re.match(r"\*\*([^*]+)\*\*:\s*(.*)", content)
            if not label_match:
                continue
            label = label_match.group(1).strip().lower()
            if not any(label.startswith(p) for p in _RCA_KEEP_PREFIXES):
                continue
            rest = label_match.group(2).strip()
            tag_match = re.search(
                r"\[(Forecast|Sensitivity[^\]]*|GraphRAG|Spatial|Parametric|Anomaly)\]\s*$",
                rest,
            )
            if tag_match:
                text = rest[: tag_match.start()].strip()
                tag = f"[{tag_match.group(1)}]"
            else:
                text, tag = rest, None
            # Trim verbose subordinate clauses
            text = re.split(r"\.\s+(?=[A-Z])", text)[0].rstrip(".")
            text = re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()
            lines.append((f"{label_match.group(1).strip()}: {text}", tag))
        events.append({"title": title, "lines": lines})

    # Dedup: collapse events with same KPI + same hour pattern
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for ev in events:
        m = re.match(r"(\w+):\s*([+\-]?\d+\.?\d*%)?\s*(.*?)\s*at\s*(.*?)\s*\(?\|z\|", ev["title"])
        if not m:
            grouped[(ev["title"], "")] = ev
            continue
        kpi, _, _, when = m.group(1), m.group(2), m.group(3), m.group(4)
        hour_match = re.search(r"(\d{2}:\d{2}\s*[–-]\s*\d{2}:\d{2})", when)
        hour = hour_match.group(1) if hour_match else when
        key = (kpi, hour)
        if key in grouped:
            existing = grouped[key]
            existing.setdefault("_dups", []).append(when)
        else:
            ev["_kpi"] = kpi
            ev["_hour"] = hour
            grouped[key] = ev

    out: list[dict[str, Any]] = []
    for ev in grouped.values():
        dups = ev.pop("_dups", None)
        kpi = ev.pop("_kpi", None)
        hour = ev.pop("_hour", None)
        if dups:
            # Pull day-of-week tokens (Mon, Tue, ...) from each duplicate window.
            days_seen: list[str] = []
            for w in [ev["title"]] + dups:
                m = re.search(r"Day\s*\d+\s*\(([A-Za-z]+)\)", w)
                if m and m.group(1) not in days_seen:
                    days_seen.append(m.group(1))
            day_list = ", ".join(days_seen) if days_seen else f"{len(dups)+1} days"
            ev["title"] = f"{kpi} spike — {hour}, days: {day_list}"
        out.append(ev)
    return out


def _parse_fidelity_footer(md: str) -> tuple[list, list, list, list, list, list, str]:
    """Parse §E.1 KPM table + §E.2 sensitivity table + Result line."""
    fidelity_rows: list[tuple[str, str, str, str, str, str]] = []
    e1 = re.search(r"### §E\.1.*?(?=\n###|\Z)", md, re.DOTALL)
    if e1:
        for r in _parse_md_table(e1.group(0)):
            if len(r) < 8:
                continue
            kpi = r[2] if r[2] != "—" else "(z)"
            metric = r[3]
            rep, pipe, err, ok = r[4], r[5], r[6], r[7]
            ok_clean = "OK" if "✓" in ok else "FAIL"
            fidelity_rows.append((kpi, metric, rep, pipe, err.replace("%", ""), ok_clean))

    sens_rows: list[tuple[str, str, str, str]] = []
    e2 = re.search(r"### §E\.2.*?(?=\n###|\Z|\n\*\*Result)", md, re.DOTALL)
    if e2:
        for r in _parse_md_table(e2.group(0)):
            if len(r) < 5:
                continue
            metric, criterion, value, ok = r[1], r[2], r[3], r[4]
            ok_clean = "OK" if "✓" in ok else ("—" if "—" in ok else "FAIL")
            sens_rows.append((metric.strip("`"), value, criterion, ok_clean))

    result_match = re.search(r"\*\*Result:\s*([^*]+)\*\*", md)
    result_line = (
        f"Result: {result_match.group(1).strip()}  "
        f"(forecast metrics + sensitivity scores verified against pipeline)"
        if result_match
        else "Result: n/a"
    )

    half_f = (len(fidelity_rows) + 1) // 2
    half_s = (len(sens_rows) + 1) // 2
    return (
        fidelity_rows[:half_f],
        fidelity_rows[half_f:],
        sens_rows[:half_s],
        sens_rows[half_s:],
        fidelity_rows,
        sens_rows,
        result_line,
    )


def from_explain_dir(explain_dir: Path, pax_ts_dir: Path | None = None) -> dict[str, Any]:
    """Build a renderer payload from an explanation directory + PAX-TS dir."""
    rj_path = explain_dir / "result.json"
    md_path = explain_dir / "explanation.md"
    rj = json.loads(rj_path.read_text())
    md = md_path.read_text()
    if pax_ts_dir is None:
        station_id = rj.get("station_id", "")
        pax_ts_dir = Path("output/pax_ts_full") / station_id

    # Forecast slope per KPI from retrieved_contexts
    slope_by_kpi: dict[str, str] = {}
    for line in rj.get("retrieved_contexts", []):
        m = re.match(
            r"\[Forecast\]\s+(\w+):.*trendSlope=(-?[\d.]+)/day",
            line,
        )
        if m:
            slope_by_kpi[m.group(1)] = (
                f"+{m.group(2)}" if not m.group(2).startswith("-") else m.group(2)
            )

    # §1 forecast table -> A.forecast rows
    forecast_block = _parse_md_section(md, "§1 Forecast Summary")
    forecast_rows = _parse_md_table(forecast_block)
    forecast = []
    for r in forecast_rows:
        if len(r) < 7:
            continue
        kpi = r[0]
        triple = r[2]
        try:
            mean_val = triple.split("/")[1].strip()
        except IndexError:
            mean_val = triple
        baseline = r[4]
        chg = r[6]
        chg_clean = chg.replace("%", "").strip()
        try:
            chg_f = float(chg_clean)
        except ValueError:
            chg_f = 0.0
        arrow = "up" if chg_f >= 0 else "down"
        chg_disp = f"{'+' if chg_f >= 0 else ''}{chg_f:.2f}%"
        forecast.append(
            {
                "kpi": kpi,
                "fc_mean": mean_val,
                "slope": slope_by_kpi.get(kpi, "n/a"),
                "base": baseline,
                "chg": chg_disp,
                "arrow": arrow,
            }
        )

    # §3 spatial -> A.station block
    spatial_block = _parse_md_section(md, "§3 Spatial & Traffic Context")
    spatial = {row[0]: row[1] for row in _parse_md_table(spatial_block) if len(row) >= 2}
    location = spatial.get("Location", "n/a")
    area_type = spatial.get("Area type", "n/a")
    nearest_road = spatial.get("Nearest road", "n/a")
    env_hint = spatial.get("Environment hint", "")

    # Largest line — from §1 prose paragraph after the table
    chg_pairs = [(row["kpi"], float(row["chg"].rstrip("%"))) for row in forecast]
    if chg_pairs:
        big = max(chg_pairs, key=lambda x: abs(x[1]))
        small = min(chg_pairs, key=lambda x: abs(x[1]))
        small_label = "near-stable" if abs(small[1]) <= 1.0 else "smallest"
        largest = f"Largest: {big[0]} {big[1]:+.2f}%, " f"{small[0]} {small_label} {small[1]:+.2f}%"
    else:
        largest = ""

    # PAX-TS sensitivity matrix + top-5
    if pax_ts_dir.exists():
        sens_matrix_rows, causal_top = _load_pax_ts_matrix(pax_ts_dir)
        sensitivity_matrix = {"labels": list(KPI_LABELS_SHORT), "rows": sens_matrix_rows}
    else:
        sensitivity_matrix = MOCK["sensitivity_matrix"]
        causal_top = MOCK["causal_top"]

    # Real GraphRAG chains from §2 sub-list and §4 [Parametric] tags.
    short_for = dict(zip(CORE_KPI_NAMES, KPI_LABELS_SHORT))

    def _short(name: str) -> str:
        return short_for.get(name, name)

    chain_lines: list[tuple[str, str]] = []
    chains_block = re.search(
        r"####\s*3GPP Causal Chains.*?(?=\n####|\n##|\Z)",
        md,
        re.DOTALL,
    )
    if chains_block:
        for line in chains_block.group(0).splitlines():
            m = re.match(r"\s*-\s*(\w+)\s+(\w+)\s+(\w+)\s*\(([^)]+)\)", line)
            if m:
                src, verb, tgt, strength = m.groups()
                chain_lines.append(
                    (f"{_short(src)}->{_short(tgt)}", f"{src} {verb.lower()} {tgt} ({strength})"),
                )
    for m in re.finditer(r"\*\*3GPP\*\*:\s*([^\[\n]+)\[Parametric\]", md):
        body = m.group(1).strip().rstrip(".")
        kpis_in = re.findall(r"(\w+)", body)
        if len(kpis_in) >= 2:
            tag = f"{_short(kpis_in[0])}->{_short(kpis_in[-1])}"
            chain_lines.append((tag, body))
    # Dedup
    seen_tags: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for tag, body in chain_lines:
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        deduped.append((tag, body))
    if not deduped:
        deduped = MOCK["graphrag_chains"]
    chain_lines = deduped[:4]

    # §5 recommendations -> D.actions (collapse duplicates by action template)
    rec_block = _parse_md_section(md, "§5 Actionable Recommendations")
    rec_rows = _parse_md_table(rec_block)
    actions = []
    seen_action: dict[str, dict[str, Any]] = {}
    for r in rec_rows:
        if len(r) < 8:
            continue
        action_key = re.sub(r"Day \d+ \(\w+\)\s*\d{2}:\d{2}[–-]\d{2}:\d{2}", "<window>", r[4])
        evidence = re.sub(r"Day \d+ \(\w+\)\s*\d{2}:\d{2}[–-]\d{2}:\d{2}", "<window>", r[7])
        if action_key in seen_action:
            seen_action[action_key]["_whens"].append(r[2])
        else:
            seen_action[action_key] = {
                "n": str(len(seen_action) + 1),
                "_whens": [r[2]],
                "action": action_key.replace("<window>", "the flagged window"),
                "evidence": evidence.replace("<window>", "the flagged window"),
            }
    for entry in seen_action.values():
        whens = entry.pop("_whens")
        if len(whens) == 1:
            entry["when"] = whens[0]
        else:
            hour_match = re.search(r"(\d{2}:\d{2}\s*[–-]\s*\d{2}:\d{2})", whens[0])
            hour = hour_match.group(1) if hour_match else ""
            days_seen: list[str] = []
            for w in whens:
                m = re.search(r"\(([A-Za-z]+)\)", w)
                if m and m.group(1) not in days_seen:
                    days_seen.append(m.group(1))
            day_list = ", ".join(days_seen) if days_seen else f"{len(whens)} days"
            entry["when"] = f"{hour}, days: {day_list}" if hour else day_list
        actions.append(entry)

    # §4 RCA -> C.anomaly_left/right (split events 50/50)
    rca_events = _parse_anomaly_rca(md)
    if rca_events:
        half = (len(rca_events) + 1) // 2
        anomaly_left = rca_events[:half]
        anomaly_right = rca_events[half:] or [{"title": "(no further events)", "lines": []}]
    else:
        anomaly_left = [{"title": "No anomaly events detected", "lines": []}]
        anomaly_right = [{"title": "—", "lines": []}]

    # §E footer -> E.fidelity_*
    fid_l, fid_r, sens_l, sens_r, _, _, fid_result = _parse_fidelity_footer(md)
    if not fid_l and not fid_r:
        fid_l, fid_r = MOCK["fidelity_left"], MOCK["fidelity_right"]
        sens_l, sens_r = MOCK["fidelity_sens_left"], MOCK["fidelity_sens_right"]
        fid_result = MOCK["fidelity_result"] + "  [placeholder]"

    payload = {
        "station": {
            "id": rj.get("station_id", "n/a"),
            "location": location,
            "environment": _classify_environment(area_type, env_hint),
            "nearest_road": nearest_road,
            "area_type": area_type,
            "explainer_model": f"{rj.get('explainer_model', 'n/a')} (ReAct)",
            "forecast_backend": "Chronos-2 (amazon/chronos-2)",
            "elapsed": f"{rj.get('elapsed_sec', 0):.2f} s  (PAX-TS + ReAct report)",
        },
        "forecast": forecast,
        "largest": largest,
        "causal_top": causal_top or MOCK["causal_top"],
        "graphrag_chains": chain_lines,
        "sensitivity_matrix": sensitivity_matrix,
        "anomaly_left": anomaly_left,
        "anomaly_right": anomaly_right,
        "actions": actions or MOCK["actions"],
        "fidelity_left": fid_l,
        "fidelity_right": fid_r,
        "fidelity_sens_left": sens_l,
        "fidelity_sens_right": sens_r,
        "fidelity_result": fid_result,
    }
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/paper_explain", help="output dir")
    p.add_argument(
        "--payload", default=None, help="optional JSON file with the same schema as MOCK"
    )
    p.add_argument(
        "--from-explain",
        default=None,
        help="path to a station/condition dir containing result.json + explanation.md",
    )
    p.add_argument("--png", action="store_true", help="also render a PNG via headless chrome")
    p.add_argument(
        "--pdf", action="store_true", help="also render a paper-ready PDF via headless chrome"
    )
    args = p.parse_args()

    if args.from_explain:
        payload = from_explain_dir(Path(args.from_explain))
    elif args.payload:
        payload = json.loads(Path(args.payload).read_text())
    else:
        payload = MOCK

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "paper_explain.html"
    html_path.write_text(render_html(payload))
    print(f"[OK] wrote {html_path}")

    content_height: int | None = None
    if args.png:
        png_path = out / "paper_explain.png"
        capture_png(html_path, png_path)
        try:
            from PIL import Image

            content_height = Image.open(png_path).size[1]
        except Exception:
            content_height = None
        print(f"[OK] wrote {png_path}")

    if args.pdf:
        pdf_path = out / "paper_explain.pdf"
        capture_pdf(html_path, pdf_path, content_px_height=content_height)
        print(f"[OK] wrote {pdf_path}")


if __name__ == "__main__":
    main()
