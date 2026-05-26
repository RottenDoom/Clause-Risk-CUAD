"""
services/output/html_renderer.py

Renders a ContractReviewOutput to a self-contained HTML report via Jinja2.
"""

from pathlib import Path

from agent.models import ContractReviewOutput
from config import OUTPUT_HTML_DIR, RISK_COLOR_MAP, TEMPLATES_DIR


def render_html(output: ContractReviewOutput, dest_dir: Path = OUTPUT_HTML_DIR) -> Path:
    """
    Render output to dest_dir/{contract_id}.html using templates/report.html.
    Returns the path of the written file.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

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

    out_path = dest_dir / f"{output.contract_id}.html"
    out_path.write_text(html_content, encoding="utf-8")
    return out_path
