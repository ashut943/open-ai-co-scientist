"""Local persistence and static reports for app research runs."""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .plots import cycle_plots_html
from .utils import logger, redact_secrets

DEFAULT_RESULTS_DIR = Path("results")
RUNS_DIR_ENV = "CO_SCIENTIST_RUNS_DIR"
DISABLE_PDF_ENV = "CO_SCIENTIST_DISABLE_PDF"

SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s<>'\"]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s<>'\"]+"),
]


def get_results_dir() -> Path:
    return Path(os.getenv(RUNS_DIR_ENV, DEFAULT_RESULTS_DIR))


def get_runs_dir() -> Path:
    return get_results_dir() / "runs"


def get_reports_dir() -> Path:
    return get_results_dir() / "reports"


def report_file_url(report_path: Path) -> str:
    """Return a Gradio file-serving URL for a generated report."""
    return f"/gradio_api/file={quote(report_path.resolve().as_posix())}"


def generate_run_id(created_at: Optional[dt.datetime] = None) -> str:
    timestamp = (created_at or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"run-{timestamp}-{uuid.uuid4().hex[:8]}"


def redact_text(text: str) -> str:
    redacted = redact_secrets(text)
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: f"{match.group(1)}***REDACTED***" if match.groups() else "***REDACTED***", redacted
        )
    return redacted


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    return value


def research_goal_to_dict(research_goal: Any) -> Dict[str, Any]:
    if research_goal is None:
        return {}
    return sanitize(
        {
            "description": getattr(research_goal, "description", ""),
            "constraints": getattr(research_goal, "constraints", {}),
            "llm_model": getattr(research_goal, "llm_model", None),
            "num_hypotheses": getattr(research_goal, "num_hypotheses", None),
            "generation_temperature": getattr(research_goal, "generation_temperature", None),
            "reflection_temperature": getattr(research_goal, "reflection_temperature", None),
            "elo_k_factor": getattr(research_goal, "elo_k_factor", None),
            "top_k_hypotheses": getattr(research_goal, "top_k_hypotheses", None),
            "user_references": getattr(research_goal, "user_references", []),
        }
    )


def save_run(
    *,
    research_goal: Any,
    cycle_details: Dict[str, Any],
    status: str,
    references_html: str,
    results_html: str,
    log_file: Optional[str] = None,
    run_id: Optional[str] = None,
    created_at: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    created = created_at or dt.datetime.now(dt.timezone.utc)
    run = sanitize(
        {
            "run_id": run_id or generate_run_id(created),
            "created_at": created.isoformat(),
            "research_goal": research_goal_to_dict(research_goal),
            "status": status,
            "log_file": log_file,
            "cycle_details": cycle_details,
            "references_html": references_html,
            "results_html": results_html,
        }
    )
    get_runs_dir().mkdir(parents=True, exist_ok=True)
    run_path = get_run_path(run["run_id"])
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")
    return run


def get_run_path(run_id: str) -> Path:
    safe_run_id = Path(run_id).name
    return get_runs_dir() / f"{safe_run_id}.json"


def load_run(run_id: str) -> Dict[str, Any]:
    return json.loads(get_run_path(run_id).read_text(encoding="utf-8"))


def delete_run(run_id: str) -> bool:
    """Delete a saved run and its generated HTML report.

    Returns True when the persisted run JSON existed and was removed. The report
    file is best-effort because reports can be regenerated and may not exist.
    """
    if not run_id:
        return False

    safe_run_id = Path(run_id).name
    run_path = get_run_path(safe_run_id)
    existed = run_path.exists()
    if not existed:
        return False

    run_path.unlink()
    report_path = get_reports_dir() / f"{safe_run_id}.html"
    report_path.unlink(missing_ok=True)
    report_path.with_suffix(".pdf").unlink(missing_ok=True)
    return True


def list_runs(limit: Optional[int] = 20) -> List[Dict[str, Any]]:
    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return []

    summaries = []
    for path in runs_dir.glob("*.json"):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        goal = run.get("research_goal", {})
        cycle = run.get("cycle_details", {})
        summaries.append(
            {
                "run_id": run.get("run_id", path.stem),
                "created_at": run.get("created_at", ""),
                "goal": goal.get("description", ""),
                "model": goal.get("llm_model", ""),
                "iteration": cycle.get("iteration", ""),
                "status": run.get("status", ""),
            }
        )

    sorted_runs = sorted(summaries, key=lambda item: item.get("created_at", ""), reverse=True)
    if limit is None:
        return sorted_runs
    return sorted_runs[:limit]


def render_report(run: Dict[str, Any]) -> str:
    goal = run.get("research_goal", {})
    cycle = run.get("cycle_details", {})
    steps = cycle.get("steps", {})
    final_hypotheses = _final_hypotheses(steps)

    html_parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(run.get('run_id'), 'Run report')}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;line-height:1.5;margin:32px;color:#1f2933;background:#fff}",
        "main{max-width:960px;margin:0 auto}",
        "section{border-top:1px solid #d9e2ec;padding-top:18px;margin-top:24px}",
        ".meta{color:#52606d}.hypothesis{border-left:4px solid #2f80ed;padding-left:12px;margin:14px 0}",
        "pre{white-space:pre-wrap;background:#f5f7fa;padding:12px;border-radius:6px;overflow:auto}",
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #d9e2ec;padding:8px;text-align:left}",
        "svg{max-width:100%;height:auto}",
        "@media print{body{margin:0}.hypothesis,svg,tr{break-inside:avoid}h2,h3{break-after:avoid}}",
        "</style>",
        "</head>",
        "<body><main>",
        f"<h1>Research Run {_escape(run.get('run_id'))}</h1>",
        f'<p class="meta">Created: {_escape(run.get("created_at"))}</p>',
        f"<p>{_escape(run.get('status'))}</p>",
        "<section><h2>Research Goal</h2>",
        f"<p>{_escape(goal.get('description'))}</p>",
        _settings_table(goal),
        "</section>",
        "<section><h2>Final Hypotheses</h2>",
    ]

    if final_hypotheses:
        for index, hypothesis in enumerate(final_hypotheses, start=1):
            html_parts.append(_hypothesis_block(index, hypothesis))
    else:
        html_parts.append("<p>No final hypotheses were available for this run.</p>")

    charts = cycle_plots_html(cycle)
    if charts:
        html_parts.append(f"</section><section><h2>Charts</h2>{charts}")

    html_parts.append("</section><section><h2>Cycle Steps</h2>")
    for step_name, step_data in steps.items():
        hypotheses = step_data.get("hypotheses", []) if isinstance(step_data, dict) else []
        html_parts.append(f"<h3>{_escape(step_name)}</h3>")
        html_parts.append(f"<p>{len(hypotheses)} hypotheses</p>")
        if step_name == "meta_review":
            html_parts.append(f"<pre>{_escape(json.dumps(step_data, indent=2, sort_keys=True))}</pre>")

    html_parts.extend(
        [
            "</section><section><h2>References</h2>",
            "<p>Reference results are stored from the app display for this run.</p>",
            f"<pre>{_escape(run.get('references_html'))}</pre>",
            "</section>",
            "</main></body></html>",
        ]
    )
    return "\n".join(html_parts)


def write_report(run: Dict[str, Any], pdf: bool = True) -> Path:
    """Write the HTML report and, when `pdf`, a PDF copy next to it (same name, .pdf)."""
    get_reports_dir().mkdir(parents=True, exist_ok=True)
    report_path = get_reports_dir() / f"{Path(run['run_id']).name}.html"
    report_path.write_text(render_report(run), encoding="utf-8")
    if pdf:
        write_pdf_report(report_path)
    return report_path


def write_pdf_report(report_path: Path) -> Optional[Path]:
    """Print an HTML report to PDF with headless Chromium (Playwright).

    Best-effort: returns None when disabled, when Playwright or its Chromium
    is not installed (`pip install playwright && playwright install chromium`),
    or when printing fails; the HTML report is unaffected.
    """
    if os.getenv(DISABLE_PDF_ENV):
        return None
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.info("Skipping PDF report: playwright is not installed.")
        return None
    pdf_path = Path(report_path).with_suffix(".pdf")
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(Path(report_path).resolve().as_uri())
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
                )
            finally:
                browser.close()
    except Exception as e:
        logger.warning("Could not write PDF report %s: %s", pdf_path.name, redact_secrets(str(e)))
        return None
    return pdf_path


def pdf_report_path(report_path: Path | str) -> Optional[Path]:
    """The PDF written next to an HTML report, if one exists."""
    pdf_path = Path(report_path).with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def ensure_report(run_id: str) -> Path:
    return write_report(load_run(run_id), pdf=False)


def history_html(limit: int = 20) -> str:
    runs = list_runs(limit=limit)
    if not runs:
        return "<p>No saved runs yet.</p>"

    rows = []
    for run in runs:
        try:
            report_path = ensure_report(run["run_id"])
            report_link = report_file_url(report_path)
            pdf_path = pdf_report_path(report_path)
        except OSError:
            report_link, pdf_path = "#", None
        pdf_link = f' · <a href="{_escape(report_file_url(pdf_path))}" target="_blank">PDF</a>' if pdf_path else ""
        rows.append(
            "<tr>"
            f"<td>{_escape(run.get('created_at'))}</td>"
            f"<td>{_escape(run.get('goal'))}</td>"
            f"<td>{_escape(run.get('iteration'))}</td>"
            f"<td><code>{_escape(run.get('run_id'))}</code></td>"
            f'<td><a href="{_escape(report_link)}" target="_blank">Open report</a>{pdf_link}</td>'
            "</tr>"
        )

    return (
        "<table><thead><tr><th>Created</th><th>Goal</th><th>Iteration</th><th>Run ID</th><th>Report</th></tr>"
        "</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _settings_table(goal: Dict[str, Any]) -> str:
    fields = [
        "llm_model",
        "num_hypotheses",
        "elo_k_factor",
        "top_k_hypotheses",
    ]
    rows = "".join(f"<tr><th>{_escape(field)}</th><td>{_escape(goal.get(field))}</td></tr>" for field in fields)
    return f"<table><tbody>{rows}</tbody></table>"


def _final_hypotheses(steps: Dict[str, Any]) -> List[Dict[str, Any]]:
    for step_name in ("ranking_final", "ranking2", "ranking", "ranking1"):
        hypotheses = steps.get(step_name, {}).get("hypotheses", [])
        if hypotheses:
            return sorted(hypotheses, key=lambda item: item.get("elo_score", 0), reverse=True)
    for step_data in steps.values():
        hypotheses = step_data.get("hypotheses", []) if isinstance(step_data, dict) else []
        if hypotheses:
            return hypotheses
    return []


def _hypothesis_block(index: int, hypothesis: Dict[str, Any]) -> str:
    comments = hypothesis.get("review_comments") or []
    comments_html = "".join(f"<li>{_escape(comment)}</li>" for comment in comments)
    scores = hypothesis.get("review_scores") or {}
    score_bits = ", ".join(f"{_escape(k)}={_escape(v)}" for k, v in scores.items() if v)
    scores_html = f"<p><strong>Scores:</strong> {score_bits}</p>" if score_bits else ""
    detail_sections = []
    for label, key in (
        ("Critical assumptions", "critical_assumptions"),
        ("Falsification conditions", "falsification_conditions"),
        ("Safety / ethics", "safety_ethical_concerns"),
        ("Recommended improvements", "recommended_improvements"),
        ("Closest prior work", "closest_prior_work"),
        ("References", "references"),
    ):
        values = hypothesis.get(key) or []
        if values:
            items = "".join(f"<li>{_escape(v)}</li>" for v in values)
            detail_sections.append(f"<p><strong>{label}:</strong></p><ul>{items}</ul>")
    details_html = "".join(detail_sections)
    return (
        '<div class="hypothesis">'
        f"<h3>{index}. {_escape(hypothesis.get('title'), 'Untitled')}</h3>"
        f"<p><strong>ID:</strong> {_escape(hypothesis.get('id'))} | "
        f"<strong>Elo:</strong> {_escape(hypothesis.get('elo_score'))}</p>"
        f"<p>{_escape(hypothesis.get('text'))}</p>"
        f"<p><strong>Novelty:</strong> {_escape(hypothesis.get('novelty_review'))} | "
        f"<strong>Feasibility:</strong> {_escape(hypothesis.get('feasibility_review'))}</p>"
        f"{scores_html}"
        f"{details_html}"
        f"<ul>{comments_html}</ul>"
        "</div>"
    )


def _escape(value: Any, default: str = "") -> str:
    if value is None:
        value = default
    return html.escape(str(value))
