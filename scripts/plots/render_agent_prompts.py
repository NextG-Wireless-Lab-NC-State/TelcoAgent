#!/usr/bin/env python3
"""Render every TelcoAgent agent prompt as a tcolorbox-style prompt-box image.

Imports the prompt constants directly from the source modules (so the images
never drift from the code) and renders each as the framed "prompt box" that
NeurIPS / ICLR papers use: a rounded thin coloured frame, a light tinted
background, a title tab, monospace body, and {placeholder} variables in an
accent colour. One PDF + PNG per prompt.

Groups (20 prompts):
  * Explainer agent   — system + 3 ReAct turn templates   (llm/prompts.py)
  * KG-construction   — 6 agents x {system, user}          (rag/prompts.py)
  * RAGAS judge       — faithfulness / answer-relevancy x {reason, score}

Usage:
    python scripts/plots/render_agent_prompts.py  [--out output/ablation/prompts]
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
    }
)

# --- group frame colours --------------------------------------------------- #
C_EXPLAINER = "#2b4c8c"  # blue
C_KG = "#117a65"  # teal
C_JUDGE = "#6c3483"  # purple
C_PLACEHOLDER = "#cf6a00"  # accent for {var} / <var>
C_BODY = "#1a1f2b"
C_SRC = "#8a93a3"

# --- geometry (inches) ----------------------------------------------------- #
WRAP = 96
BODY_FS = 7.6
TITLE_FS = 9.5
SUB_FS = 7.2
CHAR_W = BODY_FS / 72.0 * 0.6018  # DejaVu Sans Mono advance width
TITLE_CW = TITLE_FS / 72.0 * 0.62  # bold sans-serif advance (approx)
LINE_H = BODY_FS / 72.0 * 1.34
FIG_W = 7.1
MARGIN = 0.28  # box inner left/right inset
BOX_INSET = 0.03  # figure edge -> box edge
TAB_H = 0.30
PAD_TOP = 0.14
GAP_TAB_SRC = 0.05
SRC_H = 0.15
GAP_SRC_BODY = 0.13
PAD_BOTTOM = 0.18

PLACEHOLDER = re.compile(r"(\{\{?[^{}\n]*\}?\}|<[^<>\n]{1,44}>)")


def lighten(hex_colour: str, amount: float) -> str:
    """Blend a hex colour toward white by `amount` in [0,1]."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def collect_prompts() -> list[dict]:
    """Ordered prompt records pulled live from the source modules."""
    from telcoagent.evaluation import explanation_metrics as M
    from telcoagent.kg_construction import prompts as R
    from telcoagent.llm import prompts as P

    recs: list[dict] = []
    recs += [
        dict(
            colour=C_EXPLAINER,
            title="Explainer  ·  System Prompt",
            src="telcoagent/llm/prompts.py · EXPLAINER_SYSTEM_PROMPT",
            text=P.EXPLAINER_SYSTEM_PROMPT,
        ),
        dict(
            colour=C_EXPLAINER,
            title="Explainer  ·  Turn 1 — Forecast / Cross-KPI / Spatial",
            src="telcoagent/llm/prompts.py · TURN1_USER_TEMPLATE",
            text=P.TURN1_USER_TEMPLATE,
        ),
        dict(
            colour=C_EXPLAINER,
            title="Explainer  ·  Turn 2 — Scenario & Temporal",
            src="telcoagent/llm/prompts.py · TURN2_USER_TEMPLATE",
            text=P.TURN2_USER_TEMPLATE,
        ),
        dict(
            colour=C_EXPLAINER,
            title="Explainer  ·  Turn 3 — Recommendations & Summary",
            src="telcoagent/llm/prompts.py · TURN3_USER_TEMPLATE",
            text=P.TURN3_USER_TEMPLATE,
        ),
    ]
    kg = [
        ("Extractor", "EXTRACTOR_SYSTEM", "EXTRACTOR_USER_TEMPLATE"),
        ("Aligner", "ALIGNER_SYSTEM", "ALIGNER_USER_TEMPLATE"),
        ("Evaluator", "EVALUATOR_SYSTEM", "EVALUATOR_USER_TEMPLATE"),
        ("Reader", "READER_SYSTEM", "READER_USER_TEMPLATE"),
        ("Summarizer", "SUMMARIZER_SYSTEM", "SUMMARIZER_USER_TEMPLATE"),
        ("Conflict Resolver", "CONFLICT_RESOLVER_SYSTEM", "CONFLICT_RESOLVER_USER_TEMPLATE"),
    ]
    for name, sys_const, usr_const in kg:
        recs.append(
            dict(
                colour=C_KG,
                title=f"KG {name}  ·  System Prompt",
                src=f"telcoagent/kg_construction/prompts.py · {sys_const}",
                text=getattr(R, sys_const),
            )
        )
        recs.append(
            dict(
                colour=C_KG,
                title=f"KG {name}  ·  User Template",
                src=f"telcoagent/kg_construction/prompts.py · {usr_const}",
                text=getattr(R, usr_const),
            )
        )
    recs += [
        dict(
            colour=C_JUDGE,
            title="RAGAS Judge  ·  Faithfulness (reason-first, canonical)",
            src="explanation_metrics.py · _FAITHFULNESS_JUDGE_SYSTEM_PROMPT_REASON_FIRST",
            text=M._FAITHFULNESS_JUDGE_SYSTEM_PROMPT_REASON_FIRST,
        ),
        dict(
            colour=C_JUDGE,
            title="RAGAS Judge  ·  Faithfulness (score-first variant)",
            src="explanation_metrics.py · _FAITHFULNESS_JUDGE_SYSTEM_PROMPT",
            text=M._FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
        ),
        dict(
            colour=C_JUDGE,
            title="RAGAS Judge  ·  Answer Relevancy (reason-first, canonical)",
            src="explanation_metrics.py · _ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT_REASON_FIRST",
            text=M._ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT_REASON_FIRST,
        ),
        dict(
            colour=C_JUDGE,
            title="RAGAS Judge  ·  Answer Relevancy (score-first variant)",
            src="explanation_metrics.py · _ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT",
            text=M._ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT,
        ),
    ]
    return recs


def wrap_body(text: str) -> list[str]:
    """Wrap to WRAP chars, preserving blank lines and indentation."""
    out: list[str] = []
    for raw in text.expandtabs(4).rstrip("\n").split("\n"):
        if not raw.strip():
            out.append("")
            continue
        indent = raw[: len(raw) - len(raw.lstrip())]
        wrapped = textwrap.wrap(
            raw,
            width=WRAP,
            subsequent_indent=indent + "  ",
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        out.extend(wrapped or [""])
    return out


def render(rec: dict, out_dir: Path, idx: int) -> None:
    lines = wrap_body(rec["text"])
    n = len(lines)
    box_h = PAD_TOP + TAB_H + GAP_TAB_SRC + SRC_H + GAP_SRC_BODY + n * LINE_H + PAD_BOTTOM
    fig_h = box_h + 2 * BOX_INSET
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    asp = fig_h / FIG_W

    light = lighten(rec["colour"], 0.93)
    # --- outer rounded frame (thin coloured edge, light fill) ------------- #
    ax.add_patch(
        FancyBboxPatch(
            (BOX_INSET, BOX_INSET),
            FIG_W - 2 * BOX_INSET,
            box_h,
            boxstyle="round,pad=0,rounding_size=0.09",
            facecolor=light,
            edgecolor=rec["colour"],
            linewidth=1.4,
            mutation_aspect=asp,
            clip_on=False,
        )
    )

    box_top = fig_h - BOX_INSET
    # --- title tab (rounded, filled with frame colour) -------------------- #
    tab_w = min(0.26 + len(rec["title"]) * TITLE_CW + 0.13, FIG_W - 2 * MARGIN)
    tab_y = box_top - PAD_TOP - TAB_H
    ax.add_patch(
        FancyBboxPatch(
            (MARGIN, tab_y),
            tab_w,
            TAB_H,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=rec["colour"],
            edgecolor="none",
            mutation_aspect=asp,
            clip_on=False,
        )
    )
    ax.text(
        MARGIN + 0.14,
        tab_y + TAB_H / 2,
        rec["title"],
        ha="left",
        va="center",
        fontsize=TITLE_FS,
        color="#ffffff",
        fontweight="bold",
        family="sans-serif",
    )
    # source path — faint, on its own line below the tab
    src_y = tab_y - GAP_TAB_SRC - SRC_H / 2
    ax.text(
        MARGIN + 0.02,
        src_y,
        rec["src"],
        ha="left",
        va="center",
        fontsize=SUB_FS,
        color=C_SRC,
        family="monospace",
        style="italic",
    )

    # --- body — per-segment so {placeholders} are accent-coloured --------- #
    y = tab_y - GAP_TAB_SRC - SRC_H - GAP_SRC_BODY
    for line in lines:
        x = MARGIN
        for seg in PLACEHOLDER.split(line):
            if not seg:
                continue
            is_ph = bool(PLACEHOLDER.fullmatch(seg))
            ax.text(
                x,
                y,
                seg,
                ha="left",
                va="top",
                fontsize=BODY_FS,
                family="monospace",
                color=C_PLACEHOLDER if is_ph else C_BODY,
                fontweight="bold" if is_ph else "normal",
            )
            x += len(seg) * CHAR_W
        y -= LINE_H

    stem = f"prompt_{idx:02d}_" + re.sub(r"[^a-z0-9]+", "_", rec["title"].lower()).strip("_")
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{idx:02d}] {stem}  ({n} lines, {fig_h:.1f}in)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/ablation/prompts")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear stale renders from a previous style
    for old in out_dir.glob("prompt_*.p??"):
        old.unlink()

    recs = collect_prompts()
    print(f"Rendering {len(recs)} tcolorbox-style prompt cards -> {out_dir}")
    for i, rec in enumerate(recs, 1):
        render(rec, out_dir, i)
    print(f"\nDone — {len(recs)} prompts (PDF + PNG each).")


if __name__ == "__main__":
    main()
