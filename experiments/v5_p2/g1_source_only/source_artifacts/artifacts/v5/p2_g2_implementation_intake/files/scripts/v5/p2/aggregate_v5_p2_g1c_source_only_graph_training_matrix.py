#!/usr/bin/env python3
"""Aggregate the complete 4-architecture × 3-seed G1C source-only matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

OPERATORS = ("conv1d", "gcnconv", "graphconv", "gatconv")
SEEDS = (107, 117, 127)
STAGE = "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION"
COMPLETE = f"{STAGE}_COMPLETE"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "standard_deviation": stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def extract_row(report: dict[str, Any]) -> dict[str, Any]:
    selected = report["post_checkpoint_validation"]["threshold_selection"]["selected"]
    metrics = selected["metrics"]
    latency = report["cost"]["inference_latency_proxy"]
    return {
        "operator": report["operator"],
        "seed": report["seed"],
        "best_epoch": report["best_checkpoint"]["epoch"],
        "selection_score": report["best_checkpoint"]["validation_metrics_at_selection"][
            "selection_score"
        ],
        "selected_threshold": selected["threshold"],
        "threshold_objective": selected["objective"],
        "source_average_precision": metrics["source"]["average_precision"],
        "source_auroc": metrics["source"]["auroc"],
        "source_f1": metrics["source"]["thresholded"]["f1"],
        "source_precision": metrics["source"]["thresholded"]["precision"],
        "source_recall": metrics["source"]["thresholded"]["recall"],
        "source_fpr": metrics["source"]["thresholded"]["fpr"],
        "exact_set_attack": metrics["exact_set"]["attack"],
        "exact_set_overall": metrics["exact_set"]["overall"],
        "derived_graph_average_precision": metrics["derived_graph"][
            "average_precision"
        ],
        "derived_graph_auroc": metrics["derived_graph"]["auroc"],
        "derived_graph_balanced_accuracy": metrics["derived_graph"][
            "thresholded"
        ]["balanced_accuracy"],
        "derived_graph_f1": metrics["derived_graph"]["thresholded"]["f1"],
        "derived_graph_fpr": metrics["derived_graph"]["thresholded"]["fpr"],
        "parameter_count": report["architecture"]["parameter_count"],
        "total_linear_macs": report["cost"]["operation_count"]["total_linear_macs"],
        "graph_message_scalar_ops": report["cost"]["operation_count"][
            "graph_message_scalar_ops"
        ],
        "peak_cuda_memory_allocated_bytes": report["cost"][
            "peak_cuda_memory_allocated_bytes"
        ],
        "inference_microseconds_per_item": latency["microseconds_per_item"],
        "total_training_seconds": report["training"]["total_elapsed_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reports_root = args.reports_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"STOP: output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    source_reports: dict[str, dict[int, dict[str, Any]]] = {
        operator: {} for operator in OPERATORS
    }
    for operator in OPERATORS:
        for seed in SEEDS:
            run_dir = reports_root / operator / f"seed_{seed}"
            marker = run_dir / "V5_P2_G1C_SOURCE_ONLY_GRAPH_SINGLE_RUN_COMPLETE"
            report_path = run_dir / "V5_P2_G1C_SOURCE_ONLY_GRAPH_SINGLE_RUN.json"
            if not marker.is_file() or not report_path.is_file():
                raise SystemExit(f"STOP: incomplete run {operator} seed {seed}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "COMPLETE":
                raise SystemExit(f"STOP: non-COMPLETE report {report_path}")
            if report.get("operator") != operator or report.get("seed") != seed:
                raise SystemExit(f"STOP: identity mismatch in {report_path}")
            security = report.get("security_boundary", {})
            if security.get("p2_test_directory_enumerated") is not False:
                raise SystemExit(f"STOP: test enumeration flag changed in {report_path}")
            if security.get("p2_test_tensors_deserialized") is not False:
                raise SystemExit(f"STOP: test tensor flag changed in {report_path}")
            source_reports[operator][seed] = report
            rows.append(extract_row(report))

    metrics_to_summarize = [
        "selection_score",
        "selected_threshold",
        "threshold_objective",
        "source_average_precision",
        "source_auroc",
        "source_f1",
        "source_precision",
        "source_recall",
        "source_fpr",
        "exact_set_attack",
        "exact_set_overall",
        "derived_graph_average_precision",
        "derived_graph_auroc",
        "derived_graph_balanced_accuracy",
        "derived_graph_f1",
        "derived_graph_fpr",
        "inference_microseconds_per_item",
        "total_training_seconds",
    ]

    architecture_summary: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for operator in OPERATORS:
        operator_rows = [row for row in rows if row["operator"] == operator]
        summary = {
            metric: summarize([float(row[metric]) for row in operator_rows])
            for metric in metrics_to_summarize
        }
        summary["parameter_count"] = int(operator_rows[0]["parameter_count"])
        summary["total_linear_macs"] = int(operator_rows[0]["total_linear_macs"])
        summary["graph_message_scalar_ops"] = int(
            operator_rows[0]["graph_message_scalar_ops"]
        )
        memory_values = [
            row["peak_cuda_memory_allocated_bytes"] for row in operator_rows
            if row["peak_cuda_memory_allocated_bytes"] is not None
        ]
        summary["peak_cuda_memory_allocated_bytes"] = (
            summarize([float(value) for value in memory_values])
            if len(memory_values) == 3
            else None
        )
        architecture_summary[operator] = summary
        summary_rows.append(
            {
                "operator": operator,
                "source_f1_mean": summary["source_f1"]["mean"],
                "source_f1_std": summary["source_f1"]["standard_deviation"],
                "source_ap_mean": summary["source_average_precision"]["mean"],
                "source_ap_std": summary["source_average_precision"][
                    "standard_deviation"
                ],
                "exact_set_attack_mean": summary["exact_set_attack"]["mean"],
                "exact_set_attack_std": summary["exact_set_attack"][
                    "standard_deviation"
                ],
                "derived_graph_balanced_accuracy_mean": summary[
                    "derived_graph_balanced_accuracy"
                ]["mean"],
                "derived_graph_balanced_accuracy_std": summary[
                    "derived_graph_balanced_accuracy"
                ]["standard_deviation"],
                "parameter_count": summary["parameter_count"],
                "total_linear_macs": summary["total_linear_macs"],
                "inference_microseconds_per_item_mean": summary[
                    "inference_microseconds_per_item"
                ]["mean"],
            }
        )

    graphconv = architecture_summary["graphconv"]
    gat = architecture_summary["gatconv"]
    source_f1_delta = gat["source_f1"]["mean"] - graphconv["source_f1"]["mean"]
    graph_bal_delta = (
        gat["derived_graph_balanced_accuracy"]["mean"]
        - graphconv["derived_graph_balanced_accuracy"]["mean"]
    )
    explicit_gate = source_f1_delta >= 0.02 or graph_bal_delta >= 0.02
    lower_source_variance = (
        gat["source_f1"]["standard_deviation"]
        < graphconv["source_f1"]["standard_deviation"]
    )
    lower_graph_variance = (
        gat["derived_graph_balanced_accuracy"]["standard_deviation"]
        < graphconv["derived_graph_balanced_accuracy"]["standard_deviation"]
    )
    if explicit_gate:
        gat_decision = "PROMOTE_GAT_BY_EXPLICIT_0P02_PERFORMANCE_GATE"
    elif lower_source_variance or lower_graph_variance:
        gat_decision = "MANUAL_VARIANCE_REVIEW_REQUIRED_NO_AUTOMATIC_PROMOTION"
    else:
        gat_decision = "DO_NOT_PROMOTE_GAT_FROM_G1"

    report = {
        "stage": STAGE,
        "status": "COMPLETE",
        "matrix": {
            "operators": list(OPERATORS),
            "seeds": list(SEEDS),
            "completed_runs": 12,
            "execution_policy": "serial one-model-at-a-time",
        },
        "per_seed": rows,
        "mean_and_standard_deviation": architecture_summary,
        "gat_screening": {
            "rule": (
                "promote only if GAT beats GraphConv by at least 0.02 absolute "
                "source F1 or derived graph balanced accuracy, or materially "
                "reduces multi-seed variance"
            ),
            "source_f1_mean_delta_vs_graphconv": source_f1_delta,
            "derived_graph_balanced_accuracy_mean_delta_vs_graphconv": graph_bal_delta,
            "explicit_performance_gate_pass": explicit_gate,
            "gat_source_f1_std_lower_than_graphconv": lower_source_variance,
            "gat_graph_balanced_accuracy_std_lower_than_graphconv": lower_graph_variance,
            "decision": gat_decision,
            "variance_materiality_not_numerically_defined_by_g0": True,
            "architecture_selected": False,
        },
        "security_boundary": {
            "p2_test_directory_enumerated": False,
            "p2_test_tensors_deserialized": False,
            "test_evaluation_performed": False,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
        },
        "next_stage": "V5_P2_G2_DIRECT_GRAPH_DETECTION",
    }

    report_path = output_dir / f"{STAGE}.json"
    per_seed_path = output_dir / "V5_P2_G1C_PER_SEED_RESULTS.csv"
    summary_path = output_dir / "V5_P2_G1C_THREE_SEED_SUMMARY.csv"
    write_json(report_path, report)
    write_csv(per_seed_path, rows)
    write_csv(summary_path, summary_rows)
    lock = {
        "status": COMPLETE,
        "report_sha256": sha256_file(report_path),
        "per_seed_csv_sha256": sha256_file(per_seed_path),
        "summary_csv_sha256": sha256_file(summary_path),
        "completed_runs": 12,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
        "next_stage": report["next_stage"],
    }
    write_json(output_dir / f"{STAGE}_LOCK.json", lock)
    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-G1C MATRIX AGGREGATION =====")
    print("status: COMPLETE")
    print("completed_runs: 12")
    for row in summary_rows:
        print(
            f"{row['operator']}: "
            f"source_f1={row['source_f1_mean']:.6f}±{row['source_f1_std']:.6f} "
            f"exact_attack={row['exact_set_attack_mean']:.6f}±"
            f"{row['exact_set_attack_std']:.6f} "
            f"graph_bal_acc={row['derived_graph_balanced_accuracy_mean']:.6f}±"
            f"{row['derived_graph_balanced_accuracy_std']:.6f}"
        )
    print("gat_screening_decision:", gat_decision)
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
