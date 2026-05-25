#!/usr/bin/env python3
"""Run recognize API sweeps and save score results to an Excel workbook.

The script calls the FastAPI /api/v1/recognize route in-process through
TestClient so it can change recognition constants between hits without editing
source files or restarting uvicorn.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
from xml.sax.saxutils import escape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

vector_db_module: Any = None


BASE_CONSTANTS: dict[str, float | int] = {
    "FULL_VECTOR_WEIGHT": 0.52,
    "REGION_VECTOR_WEIGHT": 0.33,
    "OCCLUSION_VECTOR_WEIGHT": 0.15,
    "FULL_SCORE_THRESHOLD": 0.68,
    "REGION_SCORE_THRESHOLD": 0.55,
    "AMBIGUITY_MARGIN": 0.06,
    "MIN_ACCEPTED_SCORE": 0.70,
    "MIN_ACCEPTED_MARGIN": 0.04,
    "MIN_WEAK_MARGIN_SCORE": 0.82,
    "MIN_FULL_SUPPORTING_CHANNEL_SCORE": 0.68,
    "MIN_REGION_SUPPORTING_CHANNEL_SCORE": 0.55,
    "MIN_OCCLUSION_SUPPORTING_CHANNEL_SCORE": 0.62,
    "MIN_SUPPORTING_CHANNELS": 2,
    "MIN_FULL_STAGE_SUPPORT_SCORE": 0.40,
}

PERCENT_SWEEP_CONSTANTS = (
    "FULL_SCORE_THRESHOLD",
    "REGION_SCORE_THRESHOLD",
    "AMBIGUITY_MARGIN",
    "MIN_ACCEPTED_SCORE",
    "MIN_ACCEPTED_MARGIN",
    "MIN_WEAK_MARGIN_SCORE",
    "MIN_FULL_SUPPORTING_CHANNEL_SCORE",
    "MIN_REGION_SUPPORTING_CHANNEL_SCORE",
    "MIN_OCCLUSION_SUPPORTING_CHANNEL_SCORE",
    "MIN_FULL_STAGE_SUPPORT_SCORE",
)

DEFAULT_IMAGES = {
    "sumit": "/Users/chaitanyaarora/Downloads/sumit.jpg",
    "tarun": "/Users/chaitanyaarora/Downloads/tarun.jpg",
    "rishi": "/Users/chaitanyaarora/Downloads/rishi.png",
    "shubham": "/Users/chaitanyaarora/Downloads/shubham.png",
}

SCORE_NAMES = tuple(DEFAULT_IMAGES.keys())

OUTPUT_COLUMNS = [
    "run",
    "expected_user",
    "image_path",
    "http_status",
    "matched_name",
    "matched_name_normalized",
    "matched_score",
    "expected_user_score",
    "success_score",
    "is_correct_match",
    "failure_reason",
    *(f"{name}_score" for name in SCORE_NAMES),
    "fused_score",
    "full_face_score",
    "region_score",
    "occlusion_score",
    "recognition_stages",
    "match_quality",
    *BASE_CONSTANTS.keys(),
]

SUMMARY_COLUMNS = [
    "rank",
    "run",
    "all_matches_correct",
    "min_success_score",
    "avg_success_score",
    "avg_raw_expected_user_score",
    "min_top_score",
    "avg_top_score",
    "correct_matches",
    "total_images",
    *BASE_CONSTANTS.keys(),
]


@dataclass(frozen=True)
class ImageCase:
    expected_user: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call /api/v1/recognize for Sumit/Tarun images while sweeping "
            "recognition constants, then write scores to an .xlsx workbook."
        )
    )
    parser.add_argument(
        "--image",
        action="append",
        metavar="USER=PATH",
        help=(
            "Image case to run. Can be repeated. "
            "Default: sumit=/Users/chaitanyaarora/Downloads/sumit.jpg and "
            "tarun=/Users/chaitanyaarora/Downloads/tarun.jpg and "
            "rishi=/Users/chaitanyaarora/Downloads/rishi.png and "
            "shubham=/Users/chaitanyaarora/Downloads/shubham.png"
        ),
    )
    parser.add_argument(
        "--sweep-json",
        type=Path,
        help=(
            "Optional JSON file containing a list of constant override objects. "
            "Each object is applied over the baseline constants for one run."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "recognize_score_sweep.xlsx",
        help="Workbook path to write. Default: face_recognition_app/recognize_score_sweep.xlsx",
    )
    parser.add_argument(
        "--percent-step",
        type=float,
        default=0.01,
        help=(
            "Step size for 0%% to 100%% score/threshold sweeps. "
            "Default: 0.01 (0%%, 1%%, ... 100%%)."
        ),
    )
    return parser.parse_args()


def parse_images(raw_images: list[str] | None) -> list[ImageCase]:
    image_map = DEFAULT_IMAGES if not raw_images else {}
    if raw_images:
        for item in raw_images:
            if "=" not in item:
                raise ValueError(f"--image must look like USER=PATH, got: {item}")
            user, path = item.split("=", 1)
            image_map[user.strip().lower()] = path.strip()

    cases = [ImageCase(user, Path(path).expanduser()) for user, path in image_map.items()]
    missing = [str(case.image_path) for case in cases if not case.image_path.exists()]
    if missing:
        raise FileNotFoundError("Missing image file(s): " + ", ".join(missing))
    return cases


def percent_values(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("--percent-step must be greater than 0 and less than or equal to 1")

    values: list[float] = []
    current = 0.0
    while current < 1.0:
        values.append(round(current, 4))
        current += step
    if not values or values[-1] != 1.0:
        values.append(1.0)
    return values


def default_sweep(percent_step: float) -> list[dict[str, float | int]]:
    configs: list[dict[str, float | int]] = [copy.deepcopy(BASE_CONSTANTS)]

    weight_pairs = [
        (0.52, 0.33),
        (0.57, 0.28),
        (0.47, 0.38),
        (0.62, 0.23),
        (0.42, 0.43),
        (0.52, 0.28),
        (0.52, 0.38),
    ]
    for full_weight, region_weight in weight_pairs:
        occlusion_weight = round(1.0 - full_weight - region_weight, 2)
        if occlusion_weight <= 0:
            continue
        config = copy.deepcopy(BASE_CONSTANTS)
        config["FULL_VECTOR_WEIGHT"] = full_weight
        config["REGION_VECTOR_WEIGHT"] = region_weight
        config["OCCLUSION_VECTOR_WEIGHT"] = occlusion_weight
        configs.append(config)

    one_at_a_time: dict[str, list[float | int]] = {
        name: percent_values(percent_step)
        for name in PERCENT_SWEEP_CONSTANTS
    }
    one_at_a_time["MIN_SUPPORTING_CHANNELS"] = list(range(0, 4))
    for name, values in one_at_a_time.items():
        for value in values:
            config = copy.deepcopy(BASE_CONSTANTS)
            config[name] = value
            configs.append(config)

    deduped: list[dict[str, float | int]] = []
    seen: set[tuple[tuple[str, float | int], ...]] = set()
    for config in configs:
        key = tuple((name, config[name]) for name in BASE_CONSTANTS)
        if key not in seen:
            seen.add(key)
            deduped.append(config)
    return deduped


def load_sweep(path: Path | None, percent_step: float) -> list[dict[str, float | int]]:
    if path is None:
        return default_sweep(percent_step)

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("--sweep-json must contain a JSON list")

    configs: list[dict[str, float | int]] = []
    for index, overrides in enumerate(raw, start=1):
        if not isinstance(overrides, dict):
            raise ValueError(f"Sweep entry {index} must be an object")
        unknown = sorted(set(overrides) - set(BASE_CONSTANTS))
        if unknown:
            raise ValueError(f"Sweep entry {index} has unknown constants: {unknown}")
        config = copy.deepcopy(BASE_CONSTANTS)
        config.update(overrides)
        configs.append(config)
    return configs


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def apply_constants(config: dict[str, float | int]) -> None:
    for name, value in config.items():
        setattr(vector_db_module, name, value)


def recognize(client: Any, image_path: Path) -> tuple[int, dict[str, Any]]:
    with image_path.open("rb") as image_file:
        response = client.post(
            "/api/v1/recognize",
            files={"file": (image_path.name, image_file, "image/jpeg")},
        )
    try:
        payload = response.json()
    except Exception:
        payload = {"message": response.text}
    return response.status_code, payload


def score_for_name(results: list[dict[str, Any]], name: str) -> float | None:
    expected = normalize_name(name)
    for result in results:
        metadata = result.get("metadata") or {}
        if normalize_name(metadata.get("name")) == expected:
            return result.get("score")
    return None


def row_from_response(
    run_number: int,
    image_case: ImageCase,
    status_code: int,
    payload: dict[str, Any],
    config: dict[str, float | int],
) -> dict[str, Any]:
    first_detection = (payload.get("detections") or [{}])[0]
    results = first_detection.get("results") or []
    top_result = results[0] if results else {}
    metadata = top_result.get("metadata") or {}
    stage_scores = top_result.get("stage_scores") or {}
    expected_user_score = score_for_name(results, image_case.expected_user)
    matched_name = metadata.get("name")
    matched_name_normalized = normalize_name(matched_name)
    expected_user_normalized = normalize_name(image_case.expected_user)
    is_correct_match = matched_name_normalized == expected_user_normalized
    if is_correct_match:
        failure_reason = ""
    elif not results:
        failure_reason = first_detection.get("message") or payload.get("message") or "no accepted match"
    elif not matched_name:
        failure_reason = "matched result has no metadata name"
    else:
        failure_reason = f"name mismatch: expected {image_case.expected_user}, got {matched_name}"
    name_scores = {
        f"{name}_score": score_for_name(results, name)
        for name in SCORE_NAMES
    }

    row: dict[str, Any] = {
        "run": run_number,
        "expected_user": image_case.expected_user,
        "image_path": str(image_case.image_path),
        "http_status": status_code,
        "matched_name": matched_name,
        "matched_name_normalized": matched_name_normalized,
        "matched_score": top_result.get("score"),
        "expected_user_score": expected_user_score,
        "success_score": expected_user_score if is_correct_match else 0.0,
        "is_correct_match": is_correct_match,
        "failure_reason": failure_reason,
        **name_scores,
        "fused_score": top_result.get("fused_score"),
        "full_face_score": stage_scores.get("full_face"),
        "region_score": stage_scores.get("region"),
        "occlusion_score": stage_scores.get("occlusion_model"),
        "recognition_stages": ",".join(top_result.get("recognition_stages") or []),
        "match_quality": top_result.get("match_quality"),
    }
    row.update(config)
    return row


def best_shared_configs(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["run"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for run_number, run_rows in grouped.items():
        success_scores = [
            float(row["success_score"] or 0.0)
            for row in run_rows
        ]
        raw_expected_scores = [
            float(row["expected_user_score"] or 0.0)
            for row in run_rows
        ]
        top_scores = [
            float(row["matched_score"] or 0.0)
            for row in run_rows
        ]
        if not success_scores:
            continue

        first_row = run_rows[0]
        correct_matches = sum(1 for row in run_rows if row.get("is_correct_match"))
        summary: dict[str, Any] = {
            "run": run_number,
            "all_matches_correct": all(row.get("is_correct_match") for row in run_rows),
            "min_success_score": round(min(success_scores), 4),
            "avg_success_score": round(sum(success_scores) / len(success_scores), 4),
            "avg_raw_expected_user_score": round(sum(raw_expected_scores) / len(raw_expected_scores), 4),
            "min_top_score": round(min(top_scores), 4),
            "avg_top_score": round(sum(top_scores) / len(top_scores), 4),
            "correct_matches": correct_matches,
            "total_images": len(run_rows),
        }
        for name in BASE_CONSTANTS:
            summary[name] = first_row[name]
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            row["all_matches_correct"],
            row["correct_matches"],
            row["avg_success_score"],
            row["min_success_score"],
            row["avg_raw_expected_user_score"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(summaries[:limit], start=1):
        row["rank"] = rank
    return summaries[:limit]


def best_top_score_config(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    summaries = best_shared_configs(rows, limit=1000000)
    if not summaries:
        return None
    return max(
        summaries,
        key=lambda row: (
            row["min_top_score"],
            row["avg_top_score"],
            row["correct_matches"],
        ),
    )


def xlsx_cell(value: Any) -> str:
    if value is None:
        return "<c/>"
    if isinstance(value, bool):
        return f'<c t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)):
        return f"<c><v>{value}</v></c>"
    return f'<c t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def column_letter(column_number: int) -> str:
    letters = ""
    while column_number:
        column_number, remainder = divmod(column_number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def build_sheet_xml(columns: list[str], rows: list[dict[str, Any]]) -> str:
    all_rows = [columns]
    all_rows.extend([[row.get(column) for column in columns] for row in rows])
    row_xml = []
    for row_index, values in enumerate(all_rows, start=1):
        cells = "".join(xlsx_cell(value) for value in values)
        row_xml.append(f'<row r="{row_index}">{cells}</row>')
    filter_ref = f"A1:{column_letter(len(columns))}1"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        f'<autoFilter ref="{filter_ref}"/>'
        "</worksheet>"
    )


def write_xlsx(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/worksheets/sheet2.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        workbook.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        workbook.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets>'
            '<sheet name="Scores" sheetId="1" r:id="rId1"/>'
            '<sheet name="Best Shared Configs" sheetId="2" r:id="rId2"/>'
            '</sheets>'
            "</workbook>",
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet2.xml"/>'
            "</Relationships>",
        )
        workbook.writestr("xl/worksheets/sheet1.xml", build_sheet_xml(OUTPUT_COLUMNS, rows))
        workbook.writestr("xl/worksheets/sheet2.xml", build_sheet_xml(SUMMARY_COLUMNS, summaries))


def main() -> int:
    global vector_db_module

    args = parse_args()
    image_cases = parse_images(args.image)
    sweep_configs = load_sweep(args.sweep_json, args.percent_step)

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "face-recognition-matplotlib-cache"),
    )

    from fastapi.testclient import TestClient
    from main import app
    from app.services import vector_db as imported_vector_db_module

    vector_db_module = imported_vector_db_module

    rows: list[dict[str, Any]] = []
    client = TestClient(app)
    try:
        for run_number, config in enumerate(sweep_configs, start=1):
            apply_constants(config)
            for image_case in image_cases:
                status_code, payload = recognize(client, image_case.image_path)
                rows.append(row_from_response(run_number, image_case, status_code, payload, config))
                print(
                    f"run={run_number} user={image_case.expected_user} "
                    f"status={status_code} matched={rows[-1].get('matched_name')} "
                    f"score={rows[-1].get('matched_score')} success={rows[-1].get('is_correct_match')}"
                )
    finally:
        apply_constants(BASE_CONSTANTS)

    summaries = best_shared_configs(rows)
    top_score_summary = best_top_score_config(rows)
    write_xlsx(rows, summaries, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if summaries:
        best = summaries[0]
        label = "Best correct shared config" if best["all_matches_correct"] else "Best available shared config"
        print(
            f"{label}: "
            f"run={best['run']} "
            f"min_success_score={best['min_success_score']} "
            f"avg_success_score={best['avg_success_score']} "
            f"correct={best['correct_matches']}/{best['total_images']}"
        )
        for name in BASE_CONSTANTS:
            print(f"  {name}={best[name]}")
    if top_score_summary:
        print(
            "Highest top-score config: "
            f"run={top_score_summary['run']} "
            f"min_top_score={top_score_summary['min_top_score']} "
            f"avg_top_score={top_score_summary['avg_top_score']} "
            f"correct={top_score_summary['correct_matches']}/{top_score_summary['total_images']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
