"""
agent/output_writer.py  —  Issue 10

Serializes a ContractReviewOutput to:
  - output/json/{contract_id}.json   (model_dump_json)
  - output/html/{contract_id}.html   (Jinja2 template)

Interface:
    write_output(output: ContractReviewOutput) -> None
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CLAUSE_FAMILIES,
    OUTPUT_HTML_DIR,
    OUTPUT_JSON_DIR,
    RISK_COLOR_MAP,
    TEMPLATES_DIR,
)
from agent.models import ContractReviewOutput


def _validate(output: ContractReviewOutput) -> None:
    if len(output.clause_cards) != 4:
        raise ValueError(
            f"ContractReviewOutput must have exactly 4 clause cards, "
            f"got {len(output.clause_cards)}"
        )
    card_families = [c.clause_family for c in output.clause_cards]
    for fam in card_families:
        if fam not in CLAUSE_FAMILIES:
            raise ValueError(f"Unknown clause family in output: {fam!r}")


def write_output(output: ContractReviewOutput) -> None:
    _validate(output)

    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML_DIR.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    json_path = OUTPUT_JSON_DIR / f"{output.contract_id}.json"
    json_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    print(f"  JSON  → {json_path}")

    # --- HTML ---
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        sys.exit("ERROR: jinja2 not installed. Run: uv pip install jinja2")

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    env.globals["risk_color"] = lambda level: RISK_COLOR_MAP.get(
        level.value if hasattr(level, "value") else level,
        RISK_COLOR_MAP[None],
    )

    template = env.get_template("report.html")
    html_content = template.render(output=output, risk_color_map=RISK_COLOR_MAP)

    html_path = OUTPUT_HTML_DIR / f"{output.contract_id}.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"  HTML  → {html_path}")
