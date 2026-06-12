"""Dark-theme HTML report rendering for the Explainer Agent output.

The markdown report is the canonical artefact for evaluation and
review (it ships through RAGAS faithfulness scoring and is easy to
diff in git). This module renders the same content as a styled HTML
page so it can be screenshot-captured into a paper figure.

Visual conventions
------------------
* **Dark background** (#1a1a1a) with high-contrast text (#e8e8e8).
* **Per-section coloured left borders** so the report reads as a
  dashboard of distinct boxes (Forecast / TE / Anomaly / Action /
  Numerics). Section colour is keyed off the heading text.
* **Trend arrows** (▲ red, ▼ green, → grey) are recoloured via a
  post-processing pass so the LLM does not have to emit raw HTML.
* **Per-KPI anomaly figures** are sized to fit a 2-column grid so the
  Anomaly section reads as side-by-side narrative + plot.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import markdown

logger = logging.getLogger(__name__)


# ─── Visual constants ────────────────────────────────────────────────────

#: Page background and primary text colours. Tuned for high contrast
#: while still appearing matte (not pitch black) under PDF capture.
_BG_PAGE: str = "#1a1a1a"
_BG_BOX: str = "#252525"
_BG_TABLE_HEADER: str = "#303030"
_BG_TABLE_ROW: str = "#1f1f1f"
_TEXT_PRIMARY: str = "#e8e8e8"
_TEXT_MUTED: str = "#9aa0a6"
_BORDER_GREY: str = "#3a3a3a"

#: Per-section accent colour, applied as the left border of the
#: heading-anchored block. Keys are case-insensitive substrings of the
#: heading text.
_SECTION_ACCENT: dict = {
    "executive summary": "#7ed957",
    "forecast summary": "#3aa6ff",
    "cross-kpi": "#ffb04a",
    "spatial": "#c084fc",
    "scenario": "#f06292",
    "anomaly": "#f06292",
    "actionable": "#ffd54a",
    "numerical": "#80cbc4",
    "anomaly figure gallery": "#f06292",
}

#: Colour applied to up / down / flat trend arrows in the rendered
#: tables. ▲/↑ on increases, ▼/↓ on decreases, → on flat.
_ARROW_COLOURS: dict = {
    "▲": "#ff6b6b",
    "↑": "#ff6b6b",
    "↗": "#ffa94d",
    "▼": "#51cf66",
    "↓": "#51cf66",
    "↘": "#74c0fc",
    "→": "#9aa0a6",
}

#: Status badge colours used in §0 Executive Summary.
_BADGE_COLOURS: dict = {
    "[CRITICAL]": "#ff6b6b",
    "[WARN]": "#ffd54a",
    "[OK]": "#7ed957",
}

#: Markdown extensions enabled when converting the raw report.
_MARKDOWN_EXTENSIONS = ("tables", "fenced_code", "sane_lists", "attr_list")


# ─── Public API ──────────────────────────────────────────────────────────


def render_html_report(
    markdown_text: str,
    output_path: str,
    title: str = "TelcoAgent Explanation Report",
    plots_relative_dir: str = "plots",
) -> str:
    """Render the markdown report as a dark-theme HTML file on disk.

    Parameters
    ----------
    markdown_text
        The full report body produced by the Explainer Agent.
    output_path
        Absolute path (typically ``<station_dir>/explanation.html``).
    title
        Browser tab title and ``<h1>`` header.
    plots_relative_dir
        Directory (relative to the HTML file) where anomaly PNGs live.
        The default ``"plots"`` matches what
        :class:`telcoagent.explainer.TelcoAgentExplainer` writes.

    Returns
    -------
    str
        The path that was written.
    """
    body_html = markdown.markdown(
        markdown_text,
        extensions=list(_MARKDOWN_EXTENSIONS),
    )
    body_html = _decorate_section_borders(body_html)
    body_html = _colourise_arrows(body_html)
    body_html = _colourise_badges(body_html)
    body_html = _wrap_tables(body_html)

    html_doc = _PAGE_TEMPLATE.format(
        title=_escape_attr(title),
        css=_STYLE_SHEET,
        title_text=_escape_html(title),
        body=body_html,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    logger.info("Saved HTML report: %s", out)
    return str(out)


# ─── Decorators ──────────────────────────────────────────────────────────


_HEADING_RE = re.compile(r"<(h[1-6])>(.*?)</\1>", flags=re.IGNORECASE)
_TABLE_RE = re.compile(r"(<table>.*?</table>)", flags=re.IGNORECASE | re.DOTALL)
_ARROW_RE = re.compile(r"(▲|▼|→|↑|↓|↗|↘)")
_BADGE_RE = re.compile(r"(\[CRITICAL\]|\[WARN\]|\[OK\])")


def _decorate_section_borders(html: str) -> str:
    """Wrap each heading + following content in a coloured-border box.

    Walks all heading tags in document order and groups every sibling
    until the next heading into a ``<section class="box">`` element
    whose left border carries the accent colour for that section name.
    """
    headings = list(_HEADING_RE.finditer(html))
    if not headings:
        return html

    chunks = []
    cursor = 0
    for i, match in enumerate(headings):
        if match.start() > cursor:
            chunks.append(html[cursor : match.start()])
        next_start = headings[i + 1].start() if i + 1 < len(headings) else len(html)
        section_html = html[match.start() : next_start]
        accent = _accent_for_heading(match.group(2))
        chunks.append(
            f'<section class="box" style="border-left-color:{accent};">' f"{section_html}</section>"
        )
        cursor = next_start
    if cursor < len(html):
        chunks.append(html[cursor:])
    return "".join(chunks)


def _accent_for_heading(heading_text: str) -> str:
    """Resolve the accent colour for a heading by substring match."""
    plain = re.sub(r"<[^>]+>", "", heading_text).lower()
    for key, colour in _SECTION_ACCENT.items():
        if key in plain:
            return colour
    return _BORDER_GREY


def _colourise_arrows(html: str) -> str:
    """Wrap trend / change arrows in coloured ``<span>`` elements."""

    def repl(match: re.Match) -> str:
        symbol = match.group(1)
        colour = _ARROW_COLOURS.get(symbol, _TEXT_PRIMARY)
        return f'<span class="arrow" style="color:{colour};">{symbol}</span>'

    return _ARROW_RE.sub(repl, html)


def _colourise_badges(html: str) -> str:
    """Wrap [CRITICAL] / [WARN] / [OK] tokens in coloured pill spans."""

    def repl(match: re.Match) -> str:
        token = match.group(1)
        colour = _BADGE_COLOURS.get(token, _TEXT_MUTED)
        return f'<span class="badge" style="background:{colour};">{token}</span>'

    return _BADGE_RE.sub(repl, html)


def _wrap_tables(html: str) -> str:
    """Wrap each ``<table>`` in a scroll container so wide tables fit."""
    return _TABLE_RE.sub(
        r'<div class="table-wrap">\1</div>',
        html,
    )


# ─── HTML escaping ───────────────────────────────────────────────────────


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_attr(text: str) -> str:
    return _escape_html(text).replace('"', "&quot;")


# ─── Page template + CSS ─────────────────────────────────────────────────


_STYLE_SHEET = f"""
:root {{
  color-scheme: dark;
}}
* {{
  box-sizing: border-box;
}}
body {{
  margin: 0;
  padding: 36px 40px 80px 40px;
  background: {_BG_PAGE};
  color: {_TEXT_PRIMARY};
  font-family: "Inter", "SF Pro Text", "Segoe UI", system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.55;
  max-width: 1280px;
  margin-left: auto;
  margin-right: auto;
}}
h1.report-title {{
  font-size: 22px;
  font-weight: 600;
  letter-spacing: 0.2px;
  margin: 0 0 28px 0;
  padding-bottom: 12px;
  border-bottom: 1px solid {_BORDER_GREY};
  color: #ffffff;
}}
section.box {{
  background: {_BG_BOX};
  border-left: 4px solid {_BORDER_GREY};
  border-radius: 6px;
  padding: 18px 22px 14px 22px;
  margin: 16px 0;
}}
section.box h1, section.box h2, section.box h3 {{
  margin-top: 0;
  color: #ffffff;
  font-weight: 600;
}}
section.box h1 {{ font-size: 18px; }}
section.box h2 {{ font-size: 16px; }}
section.box h3 {{ font-size: 14px; color: #f0f0f0; }}
section.box h4 {{ font-size: 13px; color: {_TEXT_MUTED}; margin: 14px 0 6px 0; }}
p, ul, ol {{
  margin: 8px 0;
}}
ul, ol {{
  padding-left: 22px;
}}
li {{ margin: 2px 0; }}
code {{
  background: #2f2f2f;
  color: #f8c971;
  padding: 1px 5px;
  border-radius: 3px;
  font-family: "JetBrains Mono", "Menlo", monospace;
  font-size: 12.5px;
}}
.table-wrap {{
  overflow-x: auto;
  margin: 10px 0 14px 0;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}}
th, td {{
  text-align: left;
  padding: 7px 10px;
  border-bottom: 1px solid {_BORDER_GREY};
}}
thead th {{
  background: {_BG_TABLE_HEADER};
  color: #ffffff;
  font-weight: 600;
  white-space: nowrap;
}}
tbody tr:nth-child(odd) {{
  background: {_BG_TABLE_ROW};
}}
.arrow {{
  font-weight: 700;
  font-size: 13px;
}}
.badge {{
  display: inline-block;
  padding: 1px 8px;
  border-radius: 999px;
  color: #1a1a1a;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.4px;
  margin-right: 4px;
}}
img {{
  display: block;
  max-width: 100%;
  margin: 10px auto;
  border: 1px solid {_BORDER_GREY};
  border-radius: 4px;
  background: #ffffff;
}}
hr {{
  border: none;
  border-top: 1px solid {_BORDER_GREY};
  margin: 20px 0;
}}
strong {{ color: #ffffff; }}
em {{ color: {_TEXT_MUTED}; }}
"""


_PAGE_TEMPLATE = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    "<head>\n"
    '<meta charset="utf-8" />\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
    "<title>{title}</title>\n"
    "<style>{css}</style>\n"
    "</head>\n"
    "<body>\n"
    '<h1 class="report-title">{title_text}</h1>\n'
    "{body}\n"
    "</body>\n"
    "</html>\n"
)
