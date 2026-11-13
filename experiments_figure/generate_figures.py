#!/usr/bin/env python3
"""Dispatch existing paper plotting functions. Never evaluate an algorithm.

All visual construction remains in the author's model_test plotting modules.
Figures 5/10 have known missing original inputs; Figure 12 is forbidden.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
WORKSPACE = CODE.parent
MODEL_TEST = CODE / "model_test"
RESULTS = MODEL_TEST / "result/DiffGDM"
IMPORTS = WORKSPACE / "server_result_import"
OUTPUTS = HERE / "outputs"
ANALYSIS = RESULTS / "decycling_dismantling_analysis"
PARAMETERS = RESULTS / "dismantle_analysis_model_params"
NOISE = RESULTS / "dismantle_analysis_dataset"
RESOURCE = RESULTS / "dismantle_dataset_synth_size"
AUDIT = IMPORTS / "20261011_mvc_run/t2t4/merge/merge_audit.json"
MANIFEST = OUTPUTS / "run_manifest.json"
ALLOWED = tuple(range(2, 12))
RENDERABLE = (2, 3, 4, 6, 7, 8, 9, 11)
PROFILES = ("accepted-tenseed", "original-plus-controls")
METHODS_NOISE = ("CI", "DiffGDM", "NIRM", "Domirank")
FAMILIES = ("barabasi", "erdos", "small", "static")
SCALES = (20, 200, 2000, 20000, 200000)
RESOURCE_METHODS = ("CoreHD", "BPD", "DiffGDM", "MinSum")
RESOURCE_SIZE_KEYS = {method: "CoreDiffGDM" if method == "DiffGDM" else method
                      for method in RESOURCE_METHODS}
TITLES = {
    2: "Synthetic decycling/dismantling comparison",
    3: "Real-network FVS and two-route CFN comparison",
    4: "Structural-guidance AUC comparison",
    5: "Toy-network decycling/dismantling comparison",
    6: "Real-network structural-guidance comparison",
    7: "Hyperparameter sensitivity",
    8: "Runtime growth and dismantling performance",
    9: "Missing/spurious-link robustness",
    10: "Training and validation loss",
    11: "Rewrite-step convergence",
}
FUNCTIONS = {
    2: ("decycling_dismantling_analysis", "run_verified_figures"),
    3: ("decycling_dismantling_analysis", "run_verified_figures"),
    4: ("decycling_dismantling_analysis", "plt_graph_auc_analysis"),
    5: ("analysis_visi.network_dismantling_topology", "draw_dimantle_nodes"),
    6: ("decycling_dismantling_analysis", "plt_DiffGDM_direction_analysis"),
    7: ("dismantle_model_params_analysis", "draw_paper_parameter_sensitivity"),
    8: ("analysis_visi.LCC_size_removed_nodes", "draw_paper_resource_comparison"),
    9: ("dismantle_dataset_analysis", "plt_dataset_analysis"),
    10: ("model_train.utils.data_visualization", "model_evaluation_metrics_curves"),
    11: ("dismantle_model_params_analysis", "draw_paper_parameter_sensitivity"),
}
MISSING = {
    5: "Original toy graph, six removal sets, and original highlight/layout information are unavailable; 111 old PNGs are not source data.",
    10: "Original epoch/train_scores/val_scores trace is unavailable; do not retrain or digitize the old PDF.",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Non-finite JSON constant: " + value)))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def finite_values(table, label, bounded=False):
    if not isinstance(table, dict) or not table:
        raise ValueError(label + ": empty/non-dictionary records")
    for name, value in table.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (bounded and not 0 <= value <= 1)):
            raise ValueError(f"{label}/{name}: invalid numeric value")


def graph_domain(table):
    """Compare aliases only; retain original keys and line breaks for drawing."""
    names = [name.replace("\\n", "").replace("\n", "").replace("\r", "") for name in table]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate network aliases after ignoring display line breaks")
    return set(names)


def graph_order(table):
    graph_domain(table)
    return [name.replace("\\n", "").replace("\n", "").replace("\r", "") for name in table]


def source_paths(number, profile="accepted-tenseed"):
    if profile == "original-plus-controls":
        if number not in (2, 3, 6):
            raise ValueError("original-plus-controls only defines Figures 2, 3 and 6")
        paths = [IMPORTS / "20261010_local_preservation/before_manifest.json"]
        paths += [ANALYSIS / name for name in (
            "synth_graphs_decycle_size.json", "synth_graphs_dismantling_size.json",
            "graphs_decycle_size.json", "graphs_coredismantle_size.json", "graphs_onepassdismantle_size.json",
            "DiffGDM_graphs_decycle_size.json", "DiffGDM_graphs_decycledismantle_size_target.json")]
        if number in (2, 3):
            paths.append(RESULTS / "dismantle_dataset_real_all/single_run_controls.json")
        return paths
    if number == 2:
        folder = IMPORTS / "20261010_figures_2"
        return [folder / "Figure2_source_data.json", folder / "figure_source_data.csv", AUDIT]
    if number == 3:
        folder = IMPORTS / "20261010_figures_3_qafix"
        return [folder / "Figure3_source_data.json", folder / "figure_source_data.csv", AUDIT,
                IMPORTS / "20261011_mvc_run/supervision_control.json"]
    if number == 4:
        return [ANALYSIS / f"graphs_properties_{metric}_AUC.json"
                for metric in ("LCC", "ACC", "NC", "GE")]
    if number == 6:
        return [ANALYSIS / name for name in ("DiffGDM_graphs_decycle_size.json",
                                            "DiffGDM_graphs_decycledismantle_size_target.json")] + [
            RESULTS / "real_seed_variance/seed_graphs_dismantle_size_target_1_True.json"]
    if number in (7, 11):
        params = (["Rewrite_Steps"] if number == 11 else
                  ["Rewrite_Ratio", "Inference_Diffusion_Steps", "GNN_Dims", "GNN_Layers"])
        return [PARAMETERS / f"dismantle_perform_{name}.json" for name in params]
    if number == 8:
        return [RESOURCE / f"graphs_dismantle_{metric}_target_1_True.json"
                for metric in ("time", "size")]
    if number == 9:
        return [NOISE / f"method_dismantle_perform_{name}.json"
                for name in ("Missing_link", "Spurious_link")]
    return []


def require_original_function(number):
    module, function = FUNCTIONS[number]
    path = MODEL_TEST / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    if not any(isinstance(node, ast.FunctionDef) and node.name == function for node in tree.body):
        raise ValueError(f"Missing original plotting interface: {module}.{function}")


def analysis_module():
    if str(MODEL_TEST) not in sys.path:
        sys.path.insert(0, str(MODEL_TEST))
    return importlib.import_module("decycling_dismantling_analysis")


def resource_ensemble_overlay():
    """Load only an independently accepted sidecar; never start an experiment."""
    manifest = RESOURCE / "scalability_ensemble.json"
    if not manifest.exists():
        return None
    document = json_read(manifest)
    if document.get("schema") != "scale-ensemble-v1":
        raise ValueError("Unknown scalability ensemble sidecar schema")
    if document.get("state") != "ACCEPTED":
        raise ValueError("Scalability ensemble is not ACCEPTED; Figure 8 export is blocked")
    manifest_sha = sha(manifest)
    overlay = analysis_module().load_scale_ensemble_overlay(WORKSPACE)
    hashes = overlay.get("provenance", {}).get("input_SHA256", {})
    required = [manifest, RESOURCE / "scalability_ensemble.tar.gz", *source_paths(8)]
    if any(str(path.resolve()) not in hashes for path in required):
        raise ValueError("Accepted ensemble provenance lacks its protected input hashes")
    if hashes[str(manifest.resolve())] != manifest_sha:
        raise ValueError("Scalability ensemble changed during validation")
    hash_guard(hashes)
    return overlay


def resource_statistics(runtime, sizes, ensemble_overlay=None):
    """Preserve all original means, optionally append the accepted ensemble."""
    if set(runtime) != set(sizes) or len(runtime) != 50:
        raise ValueError("Runtime and CFN require the same 50 original graphs")
    for label, table, methods in (
            ("runtime", runtime, set(RESOURCE_METHODS)),
            ("CFN", sizes, set(RESOURCE_SIZE_KEYS.values()))):
        for graph, values in table.items():
            if set(values) != methods:
                raise ValueError(f"{label}/{graph}: four-method coverage changed")
            finite_values(values, f"{label}/{graph}", bounded=(label == "CFN"))
            if label == "runtime" and any(value <= 0 for value in values.values()):
                raise ValueError("Runtime values must be positive")
    buckets = {scale: [graph for graph in runtime if f"_{scale}-" in graph]
               for scale in SCALES}
    if any(len(graphs) != 10 for graphs in buckets.values()):
        raise ValueError("The five network scales require ten graphs each")
    if sorted(graph for graphs in buckets.values() for graph in graphs) != sorted(runtime):
        raise ValueError("Network-scale groups must partition the original graphs")
    time_means = {method: [math.fsum(runtime[graph][method] for graph in buckets[scale]) / 10
                           for scale in SCALES] for method in RESOURCE_METHODS}
    statistics = {"methods": list(RESOURCE_METHODS), "scales": list(SCALES),
            "graph_count": 50, "graphs_per_scale": 10,
            "CFN_source_keys": RESOURCE_SIZE_KEYS,
            "runtime_mean_seconds": time_means,
            "runtime_relative": {method: [value / values[0] for value in values]
                                 for method, values in time_means.items()},
            "CFN_mean_percent": {
                method: [100 * math.fsum(sizes[graph][RESOURCE_SIZE_KEYS[method]]
                                        for graph in buckets[scale]) / 10
                         for scale in SCALES] for method in RESOURCE_METHODS},
            "statistical_unit": "Ten different graph instances at each scale, not inference seeds",
            "error_bars": False,
            "runtime_definition": "Original runtime mean normalized to each method's smallest-scale mean"}
    if ensemble_overlay is not None:
        for key, bounded in (("cfn", True), ("inference_seconds", False)):
            values = ensemble_overlay[key]
            if set(values) != set(runtime):
                raise ValueError("Accepted ensemble requires the same 50 original graphs")
            finite_values(values, "ensemble/" + key, bounded=bounded)
            if not bounded and any(value <= 0 for value in values.values()):
                raise ValueError("Ensemble inference times must be positive")
        method = "DiffGDM_ens"
        statistics["methods"].append(method)
        times = [math.fsum(ensemble_overlay["inference_seconds"][graph]
                           for graph in buckets[scale]) / 10 for scale in SCALES]
        statistics["runtime_mean_seconds"][method] = times
        statistics["runtime_relative"][method] = [value / times[0] for value in times]
        statistics["CFN_mean_percent"][method] = [
            100 * math.fsum(ensemble_overlay["cfn"][graph] for graph in buckets[scale]) / 10
            for scale in SCALES]
        means = [value for values in statistics["CFN_mean_percent"].values() for value in values]
        padding = max(1., .05 * (max(means) - min(means)))
        statistics["CFN_y_limits_percent"] = [max(0., 5 * math.floor((min(means) - padding) / 5)),
                                               min(100., 5 * math.ceil((max(means) + padding) / 5))]
        statistics["ensemble_provenance"] = ensemble_overlay["provenance"]
        statistics["ensemble_runtime_definition"] = (
            "Accepted parallel ensemble inference elapsed time, normalized to its smallest-scale mean; "
            "original four-method timing values and definitions are unchanged")
    return statistics


def preflight(number, profile="accepted-tenseed"):
    """Read only. No output directory or image is created by this function."""
    result = {"figure": number, "title": TITLES[number], "profile": profile,
              "function": ".".join(FUNCTIONS[number])}
    if profile == "original-plus-controls" and number in (2, 3):
        result["function"] = ("analysis_visi.LCC_size_removed_nodes." +
                              ("draw_verified_synthetic_comparison" if number == 2 else
                               "draw_verified_route_comparison"))
    if number in MISSING:
        return dict(result, status="BLOCKED_MISSING_SOURCE_DATA", reason=MISSING[number])
    try:
        require_original_function(number)
        paths = source_paths(number, profile)
        absent = [str(path) for path in paths if not path.is_file()]
        if absent:
            return dict(result, status="BLOCKED_MISSING_SOURCE_DATA", reason="Missing: " + "; ".join(absent))
        checks = {}
        if profile == "original-plus-controls":
            payload = analysis_module().original_controls_payload(WORKSPACE, number)
            return dict(result, status="READY", checks={
                "graph_count": len(payload["graphs"]), "cell_counts": payload["cell_counts"],
                "average_rule": payload["average_rule"], "dispersion": payload["dispersion"],
                "paired_auto_selection": False, "new_synthetic_T2T": False},
                input_SHA256=payload["input_SHA256"])
        if number == 2:
            payload = json_read(paths[0])
            if payload["methods"] != ["MinSum", "BPD", "CoreHD", "CoreDQN", "DiffGDM", "Random-Decycle"]:
                raise ValueError("Figure 2 accepted method order changed")
            if set(payload["panels"]) != {"fig2a", "fig2b"}:
                raise ValueError("Figure 2 must contain the two accepted panels")
            if set(payload["family_graphs"]) != {"BA", "ER", "WS", "PL"}:
                raise ValueError("Figure 2 graph-family domain changed")
            if any(len(graphs) != 35 for graphs in payload["family_graphs"].values()):
                raise ValueError("Figure 2 requires 35 existing graphs per family")
            graphs = [name for family in payload["family_graphs"].values() for name in family]
            if len(set(graphs)) != 140:
                raise ValueError("Figure 2 requires 140 distinct original graphs")
            for panel in payload["panels"].values():
                if set(panel["graph_values"]) != set(graphs):
                    raise ValueError("Figure 2 panel graph coverage changed")
                for values in panel["graph_values"].values():
                    if set(values) != set(payload["methods"]):
                        raise ValueError("Figure 2 method coverage changed")
                    finite_values({method: cell["mean"] for method, cell in values.items()},
                                  "Figure 2 graph means", bounded=True)
                    for cell in values.values():
                        for metric in ("sd", "variance"):
                            if cell.get(metric) is not None:
                                finite_values({metric: cell[metric]}, "Figure 2 seed dispersion")
                                if cell[metric] < 0:
                                    raise ValueError("Negative Figure 2 dispersion")
                if set(panel["family_means"]) != set(payload["methods"]):
                    raise ValueError("Figure 2 family-mean method coverage changed")
                for method, means in panel["family_means"].items():
                    if set(means) != set(payload["family_graphs"]):
                        raise ValueError("Figure 2 family-mean graph-family coverage changed")
                    finite_values(means, "Figure 2 family means", bounded=True)
                    for family, names in payload["family_graphs"].items():
                        mean = math.fsum(panel["graph_values"][name][method]["mean"] for name in names) / 35
                        if not math.isclose(mean, means[family], rel_tol=0, abs_tol=1e-12):
                            raise ValueError("Figure 2 plotted mean differs from its original graph means")
            checks = {"graph_count": 140, "family_graph_count": 35, "error_bars": False}
        elif number == 3:
            module = analysis_module()
            inputs = {}
            audit = module._verified_read(AUDIT, inputs)
            payload = module.verified_figure3_payload(WORKSPACE, inputs, audit)
            module._verified_hash_guard(inputs)
            paths = [Path(path) for path in inputs]
            checks = {"graph_count": len(payload["graphs"]), "cell_counts": payload["cell_counts"],
                      "average_rule": payload["average_rule"], "ten_seed_average_cells": 9}
        elif number == 4:
            tables = [json_read(path) for path in paths]
            expected = 6
            for path, table in zip(paths, tables):
                if len(table) != expected:
                    raise ValueError(f"{path.name} requires {expected} networks")
                methods = set(next(iter(table.values())))
                method_order = list(next(iter(table.values())))
                for graph, values in table.items():
                    if set(values) != methods:
                        raise ValueError(f"{path.name}/{graph}: method coverage mismatch")
                    if list(values) != method_order:
                        raise ValueError(f"{path.name}/{graph}: stacked-bar method order mismatch")
                    finite_values(values, graph)
            if any(graph_domain(table) != graph_domain(tables[0]) for table in tables[1:]):
                raise ValueError("Panel graph domains differ")
            if any(graph_order(table) != graph_order(tables[0]) for table in tables[1:]):
                raise ValueError("Panel network order differs; shared labels would be misleading")
            checks = {"graph_count": expected, "panel_method_counts": [len(next(iter(t.values()))) for t in tables]}
        elif number == 6:
            payload = analysis_module().verified_figure6_payload(ANALYSIS, paths[-1])
            paths = [Path(path) for path in payload["input_SHA256"]]
            if len(payload["graphs"]) != 27 or [len(payload["panels"][key]["methods"]) for key in "ab"] != [6, 7]:
                raise ValueError("Figure 6 requires the original 27 networks and 6/7 method columns")
            checks = {"graph_count": 27, "panel_method_counts": [6, 7],
                      "legacy_FVS_scalar_cells": payload["cell_counts"]["historical_scalar_cells"],
                      "ten_seed_FVS_cells": payload["cell_counts"]["ten_seed_FVS_cells"],
                      "ten_seed_CFN_cells": 189,
                      "CFN_seed_ids": list(range(10)), "sample_variance_ddof": 1,
                      "FVS_ten_seed_statistics_available": payload["paired_publication"],
                      "paired_run_id": payload["paired_run_id"],
                      "average_rule": payload["average_rule"]}
        elif number in (7, 11):
            coverage = {}
            for path in paths:
                values = json_read(path)
                if not isinstance(values, dict) or not values:
                    raise ValueError("Empty parameter results")
                coverage[path.name] = {}
                for parameter, table in values.items():
                    counts = {}
                    for family in FAMILIES:
                        subset = {name: value for name, value in table.items() if name.startswith(family)}
                        if len(subset) < 99:
                            raise ValueError(f"{path.name}/{parameter}/{family}: fewer than 99 graphs")
                        finite_values(subset, family, bounded=True)
                        counts[family] = len(subset)
                    coverage[path.name][parameter] = counts
            checks = {"coverage": coverage, "plotting_seed": 0,
                      "sampling": "99 per family without replacement, 100 repetitions",
                      "shade": "0.1 * average of four family resampling SDs (ddof=0); NOT ten-seed SD"}
        elif number == 8:
            overlay = resource_ensemble_overlay()
            statistics = resource_statistics(*(json_read(path) for path in paths),
                                             ensemble_overlay=overlay)
            checks = {"graph_count": 50, "graphs_per_scale": 10,
                      "runtime_methods": statistics["methods"],
                      "CFN_methods": statistics["methods"],
                      "CFN_source_keys": RESOURCE_SIZE_KEYS,
                      "statistical_unit": statistics["statistical_unit"],
                      "error_bars": False,
                      "runtime_definition": statistics["runtime_definition"],
                      "new_mixed_gpu_t2t_times_used": False}
            if overlay is not None:
                checks["ensemble_state"] = "ACCEPTED"
                checks["ensemble_runtime_definition"] = statistics["ensemble_runtime_definition"]
                hashes = overlay["provenance"]["input_SHA256"]
                hash_guard(hashes)
                return dict(result, status="READY", checks=checks, input_SHA256=dict(hashes))
        elif number == 9:
            reference = None
            for path in paths:
                data = json_read(path)
                for method in METHODS_NOISE:
                    for level in (0, .05, .1, .15, .2, .25, .3):
                        table = data[method][str(level)]
                        if len(table) != 27:
                            raise ValueError("Noise comparison requires 27 networks at every level")
                        finite_values(table, f"{method}/{level}", bounded=True)
                        if reference is None:
                            reference = set(table)
                        elif set(table) != reference:
                            raise ValueError("Noise graph domains differ")
            checks = {"graph_count": 27, "noise_levels": [0, .05, .1, .15, .2, .25, .3],
                      "methods": list(METHODS_NOISE)}
        return dict(result, status="READY", checks=checks,
                    input_SHA256={str(path.resolve()): sha(path) for path in paths})
    except Exception as exc:
        return dict(result, status="FAILED_VALIDATION", reason=str(exc))


def dependency_check():
    names = ("matplotlib", "numpy", "seaborn", "pandas", "networkx", "igraph", "powerlaw", "torch")
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing plotting dependencies: " + ", ".join(missing))


def protected_inputs(checks):
    audit = json_read(AUDIT)
    protected = dict(audit["protected_result_sha256"])
    if len(protected) != 5818 or len(audit["transaction"]) != 7:
        raise ValueError("Accepted result-protection manifest changed")
    for entry in audit["transaction"]:
        protected[entry["target"]] = entry["new_sha256"]
    module = analysis_module()
    protected = module.verified_result_hashes(RESULTS / "real_seed_variance", protected)
    for directory in (IMPORTS / "20261010_figures_2", IMPORTS / "20261010_figures_3_qafix"):
        for path in directory.iterdir():
            if path.is_file():
                protected[str(path.resolve())] = sha(path)
    for check in checks.values():
        protected.update(check.get("input_SHA256", {}))
    for path in MODEL_TEST.rglob("*.py"):
        protected[str(path.resolve())] = sha(path)
    protected[str(Path(__file__).resolve())] = sha(__file__)
    protected[str(AUDIT.resolve())] = sha(AUDIT)
    return protected


def hash_guard(expected):
    changed = [path for path, value in expected.items() if not Path(path).is_file() or sha(path) != value]
    if changed:
        raise ValueError("Read-only source/result protection failed: " + "; ".join(changed[:3]))
    return digest(expected)


def artist_numeric_digest(fig):
    """Record numeric artists only; never recreate or transform plot data."""
    import numpy as np

    def json_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    values = []
    for axis in fig.axes:
        entry = {"xlim": list(axis.get_xlim()), "ylim": list(axis.get_ylim()),
                 "lines": [], "patches": [], "collections": [],
                 "texts": [(item.get_text(), list(item.get_position())) for item in axis.texts]}
        for line in axis.lines:
            entry["lines"].append({"x": line.get_xdata(orig=False).tolist(),
                                   "y": line.get_ydata(orig=False).tolist()})
        for patch in axis.patches:
            if hasattr(patch, "get_height"):
                entry["patches"].append([patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()])
        for collection in axis.collections:
            entry["collections"].append([path.vertices.tolist() for path in collection.get_paths()])
        values.append(entry)
    return digest(json.loads(json.dumps(values, default=json_scalar, allow_nan=False)))


def renderer_check(fig):
    from matplotlib.text import Text
    from unittest.mock import patch
    drawn = []
    original_draw = Text.draw

    def record_draw(text, renderer):
        if text.get_visible() and text.get_text():
            drawn.append(text)
        return original_draw(text, renderer)

    # Tick objects not drawn by their parent must not be treated as visible text.
    with patch.object(Text, "draw", record_draw):
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Print-size figures must fit their declared canvas; legacy exports remain tight.
    bounds = (fig.bbox if getattr(fig, "_exact_export_size", False) else
              fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans).padded(0.1 * fig.dpi))
    texts = list({id(text): text for text in drawn}.values())
    outside = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 1 or box.x1 > bounds.x1 + 1 or box.y0 < bounds.y0 - 1 or box.y1 > bounds.y1 + 1:
            outside.append(text.get_text())
    if outside:
        raise ValueError("Text outside export bounds: " + "; ".join(outside[:3]))
    minimum_size = min(text.get_fontsize() for text in texts) if texts else None
    required_size = getattr(fig, "_minimum_export_font", 7)
    if getattr(fig, "_exact_export_size", False) and minimum_size < required_size:
        raise ValueError("Print-size figure labels fall below their approved font floor")
    embedded_width = getattr(fig, "_embedded_width_mm", None)
    embedded_minimum = (minimum_size * embedded_width / (fig.get_size_inches()[0] * 25.4)
                        if embedded_width is not None else None)
    if embedded_minimum is not None and embedded_minimum < 7:
        raise ValueError("Figure text would be smaller than 7 pt at its actual CAS column width")
    return {"visible_text_count": len(texts), "outside_export_bounds": 0,
            "figure_size_inches": list(fig.get_size_inches()), "dpi": 300,
            "minimum_visible_font_pt": minimum_size,
            "CAS_embedded_width_mm": embedded_width,
            "CAS_embedded_minimum_font_pt": embedded_minimum,
            "visual_review": "PENDING", "numeric_artists_sha256": artist_numeric_digest(fig)}


def auc_numeric_guard(fig):
    """Verify every original stacked AUC contribution and aggregate annotation."""
    metrics = ("LCC", "ACC", "NC", "GE")
    axes = getattr(fig, "_verified_data_axes", ())
    if len(axes) != 4:
        raise ValueError("AUC comparison requires the four original panels")
    bars_checked, totals_checked = 0, 0
    for metric, axis in zip(metrics, axes):
        source = json_read(ANALYSIS / f"graphs_properties_{metric}_AUC.json")
        graphs = list(source)
        methods = list(source[graphs[0]])
        bars = axis._auc_bar_artists
        if (axis._auc_metric != metric or len(bars) != 6 * len(methods)
                or [text.get_text() for text in axis.get_xticklabels()]
                != methods):
            raise ValueError("Original AUC methods, networks or ordering changed")
        seen = set()
        for item in bars:
            graph, method, artist = item["graph"], item["method"], item["artist"]
            identity = graph, method
            if identity in seen or graph not in source or method not in methods:
                raise ValueError("AUC bars contain duplicate or unknown source identities")
            seen.add(identity)
            bottom = sum(1000 * source[g][method] for g in graphs[:graphs.index(graph)])
            if (not math.isclose(artist.get_height(), 1000 * source[graph][method], rel_tol=0, abs_tol=1e-12)
                    or not math.isclose(artist.get_y(), bottom, rel_tol=0, abs_tol=1e-10)
                    or not math.isclose(artist.get_x() + artist.get_width() / 2,
                                        methods.index(method), rel_tol=0, abs_tol=1e-12)):
                raise ValueError("AUC bar height, stack baseline or method position changed")
            bars_checked += 1
        if [item["method"] for item in axis._auc_total_artists] != methods:
            raise ValueError("AUC total annotation coverage changed")
        for item in axis._auc_total_artists:
            total = sum(100 * source[graph][item["method"]] for graph in graphs)
            if item["artist"].get_text() != f"{total:.3f}":
                raise ValueError("Original AUC aggregate annotation changed")
            totals_checked += 1
    wanted_legend = [name.replace("\\n", "").replace("\n", "").replace("\r", "")
                     for name in json_read(ANALYSIS / "graphs_properties_LCC_AUC.json")]
    if (len(fig.legends) != 1 or fig.legends[0]._ncols != 6
            or [text.get_text() for text in fig.legends[0].get_texts()] != wanted_legend):
        raise ValueError("AUC panels require one shared single-row six-network legend")
    return {"original_stacked_contributions_checked": bars_checked,
            "original_aggregate_labels_checked": totals_checked,
            "all_original_values_and_method_orders_unchanged": True,
            "shared_six_network_legend": True}


def resource_numeric_guard(fig, statistics):
    """Keep the original forty-value check distinct from any ensemble overlay."""
    axes = getattr(fig, "_verified_data_axes", ())
    if len(axes) != 2:
        raise ValueError("Resource comparison requires two panels")
    if axes[0].get_xlim() != axes[1].get_xlim():
        raise ValueError("Resource panels must share the original network-size limits")
    methods = statistics["methods"]
    if methods not in (list(RESOURCE_METHODS), [*RESOURCE_METHODS, "DiffGDM_ens"]):
        raise ValueError("Resource comparison changed the original method order")
    limits = statistics.get("CFN_y_limits_percent", [25., 80.])
    if tuple(axes[1].get_ylim()) != tuple(limits):
        raise ValueError("CFN panel changed its validated compact display range")
    if any(not limits[0] <= value <= limits[1] for values in statistics["CFN_mean_percent"].values()
           for value in values):
        raise ValueError("CFN display range must include all plotted means without clipping")
    for axis, metric in zip(axes, ("runtime_relative", "CFN_mean_percent")):
        containers = axis.containers
        if (len(containers) != len(methods) or len(axis.lines) != len(methods)
                or [item.get_label() for item in containers] != methods):
            raise ValueError("Resource comparison changed its validated methods")
        if axis.get_yscale() != "linear":
            raise ValueError("Resource panels must retain linear y axes")
        for method, container in zip(methods, containers):
            line, caps, bars = container.lines
            if caps or bars:
                raise ValueError("Resource comparison does not display error bars")
            if list(line.get_xdata(orig=False)) != list(SCALES):
                raise ValueError("Resource comparison changed its five network scales")
            actual = list(line.get_ydata(orig=False))
            expected = statistics[metric][method]
            if len(actual) != 5 or any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                                       for a, b in zip(actual, expected)):
                raise ValueError(f"{metric}/{method}: plotted values differ from original means")
    if (len(fig.legends) != 1
            or [text.get_text() for text in fig.legends[0].get_texts()] != methods):
        raise ValueError("Resource panels require one shared validated-method legend")
    return {"runtime_relative_points_checked": 5 * len(methods),
            "CFN_mean_points_checked": 5 * len(methods),
            "original_runtime_points_checked": 20, "original_CFN_points_checked": 20,
            "accepted_ensemble_points_checked": 10 if len(methods) == 5 else 0,
            "all_40_values_match_original_records": True, "runtime_normalization_unchanged": True,
            "linear_y_axes": True, "error_bars": False,
            "CFN_y_limits_percent": limits, "all_CFN_means_inside_limits": True,
            "source_key_mapping_is_display_only": True}


def render(number, overwrite, profile="accepted-tenseed"):
    if number in (2, 3) and profile == "accepted-tenseed":
        result = analysis_module().run_verified_figures(WORKSPACE, str(number),
                                                       output_dir=OUTPUTS, overwrite=overwrite)
        audit = dict(result["figure_audits"][str(number)])
        payload = audit.pop("payload")
        if number == 3:
            audit["average_statistics"] = {key: panel["average_statistics"] for key, panel in payload["panels"].items()}
        audit["renderer_QA"]["visual_review"] = "PENDING"
        return audit
    import matplotlib.pyplot as plt
    fig = None
    statistics = {}
    try:
        if profile == "original-plus-controls":
            module = analysis_module()
            statistics = module.original_controls_payload(WORKSPACE, number)
            if number in (2, 3):
                drawing = importlib.import_module("analysis_visi.LCC_size_removed_nodes")
                function = (drawing.draw_verified_synthetic_comparison if number == 2
                            else drawing.draw_verified_route_comparison)
                fig = function(statistics)
            elif number == 6:
                fig = module.plt_DiffGDM_direction_analysis(
                    None, None, result_dir=ANALYSIS, seed_file=None, show=False,
                    verified_payload=statistics)
            else:
                raise ValueError("Unsupported original-plus-controls figure")
        elif number == 4:
            fig = analysis_module().plt_graph_auc_analysis(result_dir=ANALYSIS, show=False)
        elif number == 6:
            fig, statistics = analysis_module().plt_DiffGDM_direction_analysis(
                None, None, result_dir=ANALYSIS, seed_file=source_paths(6)[-1],
                return_stats=True, show=False)
        elif number in (7, 11):
            module = importlib.import_module("dismantle_model_params_analysis")
            fig, statistics = module.draw_paper_parameter_sensitivity(PARAMETERS, rewrite_only=(number == 11), seed=0)
        elif number == 8:
            module = importlib.import_module("analysis_visi.LCC_size_removed_nodes")
            tables = [json_read(path) for path in source_paths(number)]
            overlay = resource_ensemble_overlay()
            statistics = resource_statistics(*tables, ensemble_overlay=overlay)
            fig = module.draw_paper_resource_comparison(
                *tables, ensemble_overlay=overlay,
                cfn_limits=statistics.get("CFN_y_limits_percent"))
        elif number == 9:
            module = importlib.import_module("dismantle_dataset_analysis")
            fig = module.plt_dataset_analysis("DiffGDM", list(METHODS_NOISE), result_dir=NOISE, save=False, show=False)
        else:
            raise ValueError("No rendering dispatch for this figure")
        fig.set_dpi(300)
        qa = renderer_check(fig)
        numeric = {}
        if profile == "original-plus-controls":
            numeric = analysis_module().original_controls_numeric_guard(fig, statistics, number)
            analysis_module()._verified_hash_guard(statistics["input_SHA256"])
        elif number == 6:
            artists = getattr(fig, "_verified_numeric_artists", ())
            if len(artists) != 364:
                raise ValueError("Figure 6 must display all 351 network cells and 13 Average cells")
            sd_labels = {"a": 0, "b": 0}
            for item in artists:
                panel = statistics["panels"][item["panel"]]
                if item["graph"] == "Average":
                    cell = panel["average_statistics"][item["method"]]
                    sd = cell["rms_within_network_sd"]
                else:
                    cell = panel["graph_values"][item["graph"]][item["method"]]
                    sd = cell["sd"]
                expected_label = f"{100 * cell['mean']:.2f}"
                if sd is not None:
                    expected_label += f"\n±{100 * sd:.2f}"
                    sd_labels[item["panel"]] += 1
                if item["artist"].get_text() != expected_label:
                    raise ValueError("Figure 6 numeric label does not match its source statistics")
            expected_sd_labels = {"a": 168 if statistics["paired_publication"] else 0, "b": 196}
            if sd_labels != expected_sd_labels:
                raise ValueError("Figure 6 SD labels differ from its committed paired/historical source state")
            numeric = {"network_cells": 351, "Average_cells": 13,
                       "CFN_SD_labels": sd_labels["b"], "FVS_SD_labels": sd_labels["a"],
                       "paired_publication": statistics["paired_publication"],
                       "all_364_labels_checked": True, "variance_ddof": 1}
        elif number == 4:
            numeric = auc_numeric_guard(fig)
        elif number == 8:
            numeric = resource_numeric_guard(fig, statistics)
            if "ensemble_provenance" in statistics:
                hash_guard(statistics["ensemble_provenance"]["input_SHA256"])
            elif (RESOURCE / "scalability_ensemble.json").exists():
                raise ValueError("Ensemble sidecar appeared during rendering; revalidate before export")
        files = [OUTPUTS / f"Figure{number:02d}{suffix}" for suffix in (".pdf", ".png")]
        if not overwrite and any(path.exists() for path in files):
            raise FileExistsError("Existing output requires --overwrite")
        temporary = [path.with_suffix(path.suffix + ".tmp") for path in files]
        try:
            bbox = None if getattr(fig, "_exact_export_size", False) else "tight"
            fig.savefig(temporary[0], format="pdf", bbox_inches=bbox, metadata={
                "Creator": "DiffGDM original plotting functions", "CreationDate": None, "ModDate": None})
            fig.savefig(temporary[1], format="png", dpi=300, bbox_inches=bbox, metadata={
                "Software": "DiffGDM original plotting functions"})
            for source, target in zip(temporary, files):
                os.replace(source, target)
        finally:
            for path in temporary:
                if path.exists():
                    path.unlink()
        function_name = ".".join(FUNCTIONS[number])
        if profile == "original-plus-controls" and number in (2, 3):
            function_name = "analysis_visi.LCC_size_removed_nodes." + function.__name__
        return {"function": function_name, "profile": profile, "renderer_QA": qa,
                "plotting_statistics": statistics, "numeric_QA": numeric,
                "output_SHA256": {path.name: sha(path) for path in files}}
    finally:
        if fig is not None:
            plt.close(fig)


def save_manifest(checks, run, protection, runtime):
    old = json_read(MANIFEST) if MANIFEST.is_file() else {"figures": {}}
    for number, info in checks.items():
        previous = old["figures"].get(str(number), {})
        item = dict(previous)
        item.pop("reason", None)
        item.update(info)
        if number in run:
            item.update(run[number])
        item["last_attempt_status"] = item["status"]
        if number in run and run[number]["status"] == "GENERATED":
            item["export_input_SHA256"] = dict(info.get("input_SHA256", {}))
            item["export_profile"] = info.get("profile", "accepted-tenseed")
        item["export_valid_for_current_inputs"] = ("export" in item and
            item.get("export_input_SHA256") == info.get("input_SHA256") and
            item.get("export_profile", "accepted-tenseed") == info.get("profile", "accepted-tenseed"))
        # Failed or overwrite-refused attempts must not erase prior export evidence.
        if number in run and run[number]["status"] == "GENERATED":
            prior = previous.get("export", {})
            if prior.get("output_SHA256") == item["export"].get("output_SHA256"):
                for key in ("visual_review", "visual_review_notes", "visual_reviewed_at"):
                    if key in prior.get("renderer_QA", {}):
                        item["export"]["renderer_QA"][key] = prior["renderer_QA"][key]
        old["figures"][str(number)] = item
    old.update(schema_version=1, generated_at=datetime.now().astimezone().isoformat(),
               runtime=runtime, forbidden_figures=[12], non_experiment_figures=[1],
               last_run={"selected": list(checks), "statuses": {str(n): run.get(n, checks[n])["status"] for n in checks},
                         "fully_complete": all(run.get(n, checks[n])["status"] == "GENERATED" for n in checks)},
               protection=protection,
               policy="Existing results only. Original plotting functions. No inference/training/merging. Old figures/provenance untouched.")
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(old, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)


def guidance_benefit_numeric_guard(fig, payload, metric_fig):
    """Check both split figures against the complete unchanged paired source."""
    import numpy as np
    from collections import Counter
    methods = ["NC", "Density", "GE"]
    row_keys = ("improved", "tied", "worse")
    column_keys = ("decreased", "tied", "increased")
    if payload.get("schema_version") != 4 or payload["methods"] != methods:
        raise ValueError("The forest-count analysis requires schema 4 and NC/Density/GE")
    legacy3 = {key: value for key, value in payload.items()
               if key not in ("metric_decrease_summary", "legacy_schema3_SHA256")}
    legacy3["schema_version"] = 3
    if digest(legacy3) != payload["legacy_schema3_SHA256"]:
        raise ValueError("The original schema-3 scatter and case payload changed")
    legacy = {key: value for key, value in legacy3.items() if key not in (
        "component_cases", "legacy_schema2_SHA256", "component_case_validation",
        "component_case_definition")}
    legacy["schema_version"] = 2
    if digest(legacy) != payload["legacy_schema2_SHA256"]:
        raise ValueError("The original schema-2 scatter and statistics payload changed")
    if len(payload["graphs"]) != 27 or len(set(payload["graphs"])) != 27:
        raise ValueError("The forest-count analysis requires 27 distinct networks")
    counts, overlap_groups = {}, {}
    for method in methods:
        rows = payload["points"][method]
        if len(rows) != 27 or {r["graph"] for r in rows} != set(payload["graphs"]):
            raise ValueError("Forest-count observations have incomplete network coverage")
        counts[method] = {row: {column: 0 for column in column_keys} for row in row_keys}
        coordinates = []
        for row in rows:
            n = row["N"]
            threshold = (n + 99) // 100
            if (row["threshold"] != threshold
                    or any(size <= threshold for size in row["oversized_sizes"])
                    or any(size <= threshold for size in row["base_oversized_sizes"])
                    or row["oversized_trees"] != len(row["oversized_sizes"])
                    or row["base_oversized_trees"] != len(row["base_oversized_sizes"])):
                raise ValueError("Oversized-tree counts do not match the saved forest sizes")
            delta_k = row["oversized_trees"] - row["base_oversized_trees"]
            delta_d = row["final_nodes"] - row["base_final_nodes"]
            if (row["delta_trees"] != delta_k
                    or not math.isclose(row["delta_cfn_pp"], 100 * delta_d / n,
                                        rel_tol=0, abs_tol=1e-12)
                    or not math.isclose(row["delta_tree_density"], delta_k / n,
                                        rel_tol=0, abs_tol=1e-12)):
                raise ValueError("Scatter variables differ from paired integer endpoint changes")
            outcome = "improved" if delta_d < 0 else "worse" if delta_d > 0 else "tied"
            direction = "decreased" if delta_k < 0 else "increased" if delta_k > 0 else "tied"
            counts[method][outcome][direction] += 1
            coordinates.append((delta_k, row["delta_cfn_pp"]))
        if counts[method] != payload["direction_counts"][method]:
            raise ValueError("The direction-count table differs from the paired observations")
        if any(sum(counts[method][row].values()) != payload["outcomes"][method][row]
               for row in row_keys):
            raise ValueError("CFN outcome totals differ from the direction-count table")
        overlap_groups[method] = {xy: count for xy, count in Counter(coordinates).items() if count > 1}
    plotted = {method: [] for method in methods}
    groups = set()
    for item in fig._guidance_scatter_artists:
        group = (item["method"], item["outcome"])
        if group in groups or item["method"] not in methods or item["outcome"] not in row_keys:
            raise ValueError("Unexpected or duplicate scatter outcome group")
        groups.add(group)
        rows = {r["graph"]: r for r in payload["points"][item["method"]]}
        for graph in item["graphs"]:
            row = rows[graph]
            outcome = ("improved" if row["final_nodes"] < row["base_final_nodes"] else
                       "worse" if row["final_nodes"] > row["base_final_nodes"] else "tied")
            if outcome != item["outcome"]:
                raise ValueError("A plotted observation is assigned to the wrong CFN outcome")
        wanted = np.asarray([[rows[g]["delta_trees"], rows[g]["delta_cfn_pp"]]
                             for g in item["graphs"]], dtype=float).reshape(-1, 2)
        actual = np.asarray(item["artist"].get_offsets(), dtype=float).reshape(-1, 2)
        if actual.shape != wanted.shape or not np.allclose(actual, wanted, rtol=0, atol=1e-12):
            raise ValueError("Forest-count scatter offsets differ from the validated endpoints")
        plotted[item["method"]].extend(item["graphs"])
    if (groups != {(method, outcome) for method in methods for outcome in row_keys}
            or any(sorted(names) != sorted(payload["graphs"]) for names in plotted.values())):
        raise ValueError("Each guidance scatter must contain all 27 networks exactly once")
    if any(sum(sum(row.values()) for row in counts[method].values()) != 27 for method in methods):
        raise ValueError("Every full source direction-count matrix must still sum to 27")
    overlaps = set()
    for item in fig._guidance_overlap_labels:
        method, xy = item["method"], tuple(item["coordinates"])
        group = (method, xy)
        if (group in overlaps or xy not in overlap_groups[method]
                or item["count"] != overlap_groups[method][xy]
                or item["artist"].get_text() != "n=" + str(item["count"])
                or not np.allclose(item["artist"].xy, xy, rtol=0, atol=1e-12)):
            raise ValueError("A repeated-point label differs from its exact observation multiplicity")
        leader = item["artist"].arrow_patch
        if leader is None or not 0 < leader.get_linewidth() <= .8:
            raise ValueError("Every repeated-point count must have a thin leader line")
        overlaps.add(group)
    if overlaps != {(m, xy) for m in methods for xy in overlap_groups[m]}:
        raise ValueError("Every exact repeated coordinate must have a multiplicity label")
    renderer = fig.canvas.get_renderer()
    for method in methods:
        labels = [item["artist"] for item in fig._guidance_overlap_labels if item["method"] == method]
        for index, label in enumerate(labels):
            # Compare the text boxes, not the enclosing rectangles of slanted leaders.
            if any(label.get_bbox_patch().get_window_extent(renderer).overlaps(
                       other.get_bbox_patch().get_window_extent(renderer))
                   for other in labels[index + 1:]):
                raise ValueError("Repeated-point multiplicity labels overlap")
    for figure in (fig, metric_fig):
        if (len(figure.legends) != 1 or [text.get_text() for text in figure.legends[0].get_texts()]
                != ["Improved", "Unchanged", "Worse"]):
            raise ValueError("The CFN outcome legend must read Improved / Unchanged / Worse")
    axes = fig._guidance_axes["scatter"]
    metric_axes = metric_fig._guidance_axes["metric"]
    if (set(fig._guidance_axes) != {"scatter"} or set(metric_fig._guidance_axes) != {"metric"}
            or set(axes) != set(methods) or set(metric_axes) != set(methods)
            or len(fig.axes) != 3 or len(metric_fig.axes) != 3
            or set(fig.axes) != set(axes.values()) or set(metric_fig.axes) != set(metric_axes.values())
            or getattr(fig, "_guidance_layout", None) != "forest-count-vertical"
            or getattr(metric_fig, "_guidance_layout", None) != "metric-direction-horizontal"):
        raise ValueError("The analysis requires the approved main vertical and supplementary horizontal triptychs")
    x_limits = [tuple(axes[m].get_xlim()) for m in methods]
    y_limits = [tuple(axis.get_ylim()) for figure in (fig, metric_fig) for axis in figure.axes]
    if len(set(x_limits)) != 1 or len(set(y_limits)) != 1:
        raise ValueError("The forest-count x limits and all six CFN y limits must be shared")
    from matplotlib.colors import to_rgba
    cache = payload["metric_decrease_summary"]
    expected_fields = ["graph", "N", "base_metric_value", "metric_value", "delta_metric",
                       "metric_direction", "delta_fvs_count", "delta_K", "delta_Q", "delta_L",
                       "delta_cfn_pp", "outcome"]
    if (cache.get("status") != "COMPLETE" or cache.get("graphs") != payload["graphs"]
            or cache.get("methods") != methods or cache.get("row_fields") != expected_fields
            or cache.get("legacy_schema3_SHA256") != payload["legacy_schema3_SHA256"]
            or cache.get("input_SHA256") != payload["input_SHA256"]
            or cache.get("records_SHA256") != digest(cache.get("rows"))):
        raise ValueError("Metric summary source identities or its exact-audit record digest differ")
    metric_counts, metric_matrices, metric_rows = {}, {}, {}
    for method in methods:
        rows = cache["rows"][method]
        if (len(rows) != 27 or any(len(row) != len(expected_fields) for row in rows)
                or [row[0] for row in rows] != payload["graphs"]):
            raise ValueError("Metric summary source must contain all 27 networks in the original order")
        paired = {point["graph"]: point for point in payload["points"][method]}
        matrix = {direction: {outcome: 0 for outcome in ("improved", "unchanged", "worse")}
                  for direction in ("decreased", "unchanged", "increased")}
        for values in rows:
            row = dict(zip(expected_fields, values))
            point = paired[row["graph"]]
            final_delta = point["final_nodes"] - point["base_final_nodes"]
            outcome = "improved" if final_delta < 0 else "worse" if final_delta > 0 else "unchanged"
            if (row["N"] != point["N"] or row["outcome"] != outcome
                    or row["delta_fvs_count"] != point["fvs_nodes"] - point["base_fvs_nodes"]
                    or row["delta_K"] != point["delta_trees"]
                    or not math.isclose(row["delta_cfn_pp"], point["delta_cfn_pp"], rel_tol=0, abs_tol=1e-12)
                    or row["metric_direction"] not in matrix):
                raise ValueError("Metric summary record differs from the original paired forest/CFN data")
            if method in ("NC", "GE"):
                delta = row["metric_value"] - row["base_metric_value"]
                tolerance = 1e-10 * max(1.0, abs(row["metric_value"]), abs(row["base_metric_value"]))
                direction = ("unchanged" if abs(delta) <= tolerance else
                             "decreased" if delta < 0 else "increased")
                if direction != row["metric_direction"]:
                    raise ValueError("A summary metric direction differs from the exact-audit tie definition")
            matrix[row["metric_direction"]][outcome] += 1
        if matrix != cache["direction_counts"][method] or sum(sum(r.values()) for r in matrix.values()) != 27:
            raise ValueError("Metric summary 3-by-3 counts differ from the complete paired observations")
        selected = matrix["decreased"]
        wanted = {"decreased_count": sum(selected.values()), "counts": {
            "improved": selected["improved"], "tied": selected["unchanged"], "worse": selected["worse"]}}
        if wanted != cache["decreased_outcomes"][method]:
            raise ValueError("The conditional metric-decreased counts do not match the complete source records")
        metric_counts[method] = wanted
        metric_matrices[method] = matrix
        metric_rows[method] = {row[0]: dict(zip(expected_fields, row)) for row in rows}
    categories = {"decreased": 0, "unchanged": 1, "increased": 2}
    upper_groups, upper_plotted = set(), {method: [] for method in methods}
    upper_collections = {method: set() for method in methods}
    upper_positions = {method: {category: [] for category in categories.values()} for method in methods}
    for item in metric_fig._guidance_metric_artists:
        method, outcome = item["method"], item["outcome"]
        group = method, outcome
        if method not in methods or outcome not in row_keys or group in upper_groups:
            raise ValueError("Unexpected or duplicate upper metric-direction outcome group")
        names = item["graphs"]
        if len(names) != len(set(names)) or any(graph not in metric_rows[method] for graph in names):
            raise ValueError("Upper metric-direction points contain duplicate or unknown networks")
        source_outcome = "unchanged" if outcome == "tied" else outcome
        wanted_categories = [categories[metric_rows[method][graph]["metric_direction"]] for graph in names]
        wanted_y = np.asarray([metric_rows[method][graph]["delta_cfn_pp"] for graph in names], dtype=float)
        artist = item["artist"]
        actual = np.asarray(artist.get_offsets(), dtype=float).reshape(-1, 2)
        if (item["categories"] != wanted_categories or artist.axes is not metric_axes[method]
                or any(metric_rows[method][graph]["outcome"] != source_outcome for graph in names)
                or actual.shape != (len(names), 2)
                or not np.isfinite(actual).all()
                or not np.allclose(actual[:, 1], wanted_y, rtol=0, atol=1e-12)
                or any(abs(x - category) > .34 + 1e-12 for x, category in zip(actual[:, 0], wanted_categories))):
            raise ValueError("Upper metric-direction points must preserve their exact category, CFN and outcome")
        lower = next(group["artist"] for group in fig._guidance_scatter_artists
                     if group["method"] == method and group["outcome"] == outcome)
        if (not np.allclose(artist.get_facecolors(), lower.get_facecolors(), rtol=0, atol=1e-12)
                or len(artist.get_paths()) != 1 or len(lower.get_paths()) != 1
                or not np.array_equal(artist.get_paths()[0].vertices, lower.get_paths()[0].vertices)
                or not np.array_equal(artist.get_paths()[0].codes, lower.get_paths()[0].codes)):
            raise ValueError("Upper and lower points must share CFN outcome colors and marker shapes")
        upper_groups.add(group)
        upper_collections[method].add(artist)
        upper_plotted[method].extend(names)
        for xy, category in zip(actual, wanted_categories):
            upper_positions[method][category].append(xy)
    if (upper_groups != {(method, outcome) for method in methods for outcome in row_keys}
            or any(sorted(names) != sorted(payload["graphs"]) for names in upper_plotted.values())):
        raise ValueError("Every metric-direction panel must contain all 27 networks exactly once")
    checked_categories = set()
    category_totals = {}
    for item in metric_fig._guidance_metric_groups:
        method, direction, category = item["method"], item["direction"], item["category"]
        key = method, direction
        if (method not in methods or direction not in categories or key in checked_categories
                or category != categories[direction] or item["axis"] is not metric_axes[method]):
            raise ValueError("Unexpected or duplicate metric-direction category metadata")
        wanted_count = sum(metric_matrices[method][direction].values())
        observed_count = sum(item_row["metric_direction"] == direction for item_row in metric_rows[method].values())
        if item["count"] != wanted_count or wanted_count != observed_count:
            raise ValueError("Metric-direction category count differs from the exact full-network audit")
        checked_categories.add(key)
        category_totals.setdefault(method, {})[direction] = wanted_count
    if checked_categories != {(method, direction) for method in methods for direction in categories}:
        raise ValueError("All nine metric-direction categories must retain their network counts")
    category_label_gaps_pt = []
    for method, axis in metric_axes.items():
        labels = [direction.capitalize() + "\n(n=" + str(category_totals[method][direction]) + ")"
                  for direction in categories]
        if (not np.allclose(axis.get_xlim(), [-.55, 2.55], rtol=0, atol=1e-12)
                or not np.allclose(axis.get_xticks(), [0, 1, 2], rtol=0, atol=1e-12)
                or [text.get_text() for text in axis.get_xticklabels()] != labels
                or axis.patches or set(axis.collections) != upper_collections[method]
                or len(axis.lines) != 1
                or not np.allclose(axis.lines[0].get_ydata(), 0, rtol=0, atol=1e-12)
                or not np.allclose(axis.lines[0].get_xdata(), [0, 1], rtol=0, atol=1e-12)):
            raise ValueError("Upper panels require only all observed points, exact group labels and the CFN zero reference")
        boxes = [text.get_window_extent(metric_fig.canvas.get_renderer())
                 for text in axis.get_xticklabels()]
        for left_box, right_box in zip(boxes, boxes[1:]):
            gap = (right_box.x0 - left_box.x1) * 72 / metric_fig.dpi
            category_label_gaps_pt.append(float(gap))
            if gap < 1.0:
                raise ValueError("Adjacent metric category labels need at least 1 pt of clear space")
    spacing = getattr(metric_fig, "_guidance_metric_spacing_points", None)
    if spacing != 5.0:
        raise ValueError("The approved deterministic metric-direction point spacing must remain 5.0 pt")
    smallest_spacing = None
    for method, by_category in upper_positions.items():
        for positions in by_category.values():
            if len(positions) < 2:
                continue
            pixels = metric_axes[method].transData.transform(np.asarray(positions))
            point_coordinates = pixels * 72 / metric_fig.dpi
            for index, xy in enumerate(point_coordinates):
                separations = np.sqrt(np.sum((point_coordinates[index + 1:] - xy) ** 2, axis=1))
                if len(separations):
                    value = float(np.min(separations))
                    smallest_spacing = value if smallest_spacing is None else min(smallest_spacing, value)
                    if value < spacing - .02:
                        raise ValueError("Metric-direction observations overlap at the approved final point spacing")
    quadrant_cells = set()
    quadrant_marker_clearances_pt = []
    for item in fig._guidance_quadrants:
        method, row, column = item["method"], item["row"], item["column"]
        cell = (method, row, column)
        if cell in quadrant_cells or row not in ("improved", "worse") or column not in ("decreased", "increased"):
            raise ValueError("Unexpected or duplicate quadrant")
        axis = axes[method]
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        x0, x1 = (left, 0) if column == "decreased" else (0, right)
        y0, y1 = (bottom, 0) if row == "improved" else (0, top)
        patch = item["patch"]
        should_shade = ((row == "worse" and column == "decreased")
                        or (row == "improved" and column == "increased"))
        if should_shade:
            if patch is None:
                raise ValueError("The upper-left and lower-right quadrants must be shaded")
            actual = [patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()]
            if (not np.allclose(actual, [x0, y0, x1 - x0, y1 - y0], rtol=0, atol=1e-12)
                    or not np.allclose(patch.get_facecolor(), to_rgba("#F0F6FA" if row == "improved" else "#FCF4EA"))
                    or patch.get_zorder() >= 0):
                raise ValueError("Selected quadrant shading must follow exact zero boundaries and CFN colors")
        elif patch is not None:
            raise ValueError("The upper-right and lower-left counterexample quadrants must remain unshaded")
        label = ("Fewer trees" if column == "decreased" else "More trees")
        label += "; " if row == "improved" else "\n"
        label += ("Improved" if row == "improved" else "Worse") + ": " + str(counts[method][row][column])
        if item["artist"].get_text() != label:
            raise ValueError("Quadrant count differs from the matching table cell")
        box = item["artist"].get_window_extent(fig.canvas.get_renderer())
        pixel_points = axis.transData.transform(
            [(point["delta_trees"], point["delta_cfn_pp"]) for point in payload["points"][method]])
        clearance = min(np.hypot(max(box.x0 - x, 0, x - box.x1),
                                 max(box.y0 - y, 0, y - box.y1))
                        for x, y in pixel_points) * 72 / fig.dpi
        quadrant_marker_clearances_pt.append(float(clearance))
        if clearance < 3.5:
            raise ValueError("Quadrant labels must not touch scatter markers")
        quadrant_cells.add(cell)
    if quadrant_cells != {(m, r, c) for m in methods for r in ("improved", "worse")
                           for c in ("decreased", "increased")}:
        raise ValueError("All 12 quadrant count labels must be rendered, including counterexamples")
    for axis in axes.values():
        if len(axis.patches) != 2 or len(axis.lines) != 2:
            raise ValueError("Scatter panels require exactly two shaded quadrants and no fitted curves")
        if not all(np.allclose(line.get_xdata(), 0) or np.allclose(line.get_ydata(), 0)
                   for line in axis.lines):
            raise ValueError("Only horizontal and vertical zero reference lines are allowed")
    for method, axis in axes.items():
        actual_groups = [item["artist"] for item in fig._guidance_scatter_artists if item["method"] == method]
        if (set(axis.collections) != set(actual_groups)
                or any(artist.axes is not axis for artist in actual_groups)):
            raise ValueError("Lower panels may show only the three unchanged full-network outcome groups")
    cases = {case["id"]: case for case in payload["component_cases"]}
    if len(payload["component_cases"]) != 2 or set(cases) != {"d", "e"}:
        raise ValueError("Exactly the approved positive case and counterexample are required")
    expected_cases = {"d": ("econ-wm1", ["NC", "Density", "GE"]),
                      "e": ("opsahl-openflights", ["Density", "GE"])}
    block_maps, membership_checks = {}, {}

    def checked_nodes(raw, n, label):
        if (not isinstance(raw, list)
                or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n for v in raw)
                or len(raw) != len(set(raw))):
            raise ValueError("Invalid or duplicate original node identities: " + label)
        return set(raw)

    for case_id, case in cases.items():
        graph, guides = expected_cases[case_id]
        if case["graph"] != graph or case["methods"] != guides:
            raise ValueError("Membership example differs from the approved network/guidance selection")
        n, threshold = case["N"], case["threshold"]
        if threshold != (n + 99) // 100:
            raise ValueError("The membership example uses a different dismantling threshold")
        components, fvs_sets, blocks, retained = {}, {}, {}, {}
        for side in ("base", "guided"):
            fvs_sets[side] = checked_nodes(case["fvs"][side], n, case_id + "/" + side + "/FVS")
            components[side], side_nodes = {}, set()
            for component in case["components"][side]:
                component_id = component["id"]
                nodes = checked_nodes(component["nodes"], n, case_id + "/component")
                if (component_id in components[side] or not nodes or side_nodes & nodes
                        or component["size"] != len(nodes)
                        or component["oversized"] != (len(nodes) > threshold)):
                    raise ValueError("Invalid component identity, size, disjointness or threshold class")
                components[side][component_id] = nodes
                side_nodes.update(nodes)
            retained[side] = side_nodes
            endpoint = case[side]
            if (side_nodes != set(range(n)) - fvs_sets[side]
                    or endpoint["fvs_nodes"] != len(fvs_sets[side])
                    or endpoint["remaining_nodes"] != len(side_nodes)
                    or endpoint["component_count"] != len(components[side])
                    or endpoint["K"] != sum(len(v) > threshold for v in components[side].values())):
                raise ValueError("Component members do not conserve the retained nodes or endpoint counts")
            blocks[side], covered_components = {}, set()
            for block in case["blocks"][side]:
                block_id, ids = block["id"], block["component_ids"]
                if (block_id in blocks[side] or len(ids) != len(set(ids)) or not ids
                        or not set(ids) <= set(components[side]) or covered_components & set(ids)):
                    raise ValueError("Displayed component groups are duplicate, overlapping or incomplete")
                members = set().union(*(components[side][key] for key in ids))
                sizes = sorted((len(components[side][key]) for key in ids), reverse=True)
                oversized_count = sum(value > threshold for value in sizes)
                if (checked_nodes(block["nodes"], n, case_id + "/block") != members
                        or block["size"] != len(members)
                        or block["component_count"] != len(ids)
                        or block["oversized_count"] != oversized_count
                        or sorted(block["component_sizes"], reverse=True) != sizes):
                    raise ValueError("Displayed component group size or threshold accounting is inconsistent")
                if (block["kind"] not in ("single", "identical_pool", "small_pool")
                        or (block["kind"] == "single" and len(ids) != 1)
                        or (block["kind"] == "small_pool" and oversized_count != 0)):
                    raise ValueError("A component pool is misrepresented as one tree or contains oversized trees")
                blocks[side][block_id] = block
                covered_components.update(ids)
            if covered_components != set(components[side]):
                raise ValueError("Every residual component must be represented exactly once")
            point_rows = {row["graph"]: row for row in payload["points"][guides[0]]}
            row = point_rows[graph]
            if (n != row["N"] or endpoint["fvs_nodes"] != row[
                    "base_fvs_nodes" if side == "base" else "fvs_nodes"]
                    or endpoint["final_nodes"] != row[
                        "base_final_nodes" if side == "base" else "final_nodes"]
                    or endpoint["K"] != row[
                        "base_oversized_trees" if side == "base" else "oversized_trees"]):
                raise ValueError("Membership case endpoint differs from its unchanged scatter observation")
        for guide in guides:
            row = next(row for row in payload["points"][guide] if row["graph"] == graph)
            if (row["fvs_nodes"] != case["guided"]["fvs_nodes"]
                    or row["final_nodes"] != case["guided"]["final_nodes"]
                    or row["oversized_trees"] != case["guided"]["K"]):
                raise ValueError("Guidance aliases assigned to the example have different endpoints")
        for side, other in (("base", "guided"), ("guided", "base")):
            for block in blocks[side].values():
                if block["kind"] == "identical_pool":
                    other_sets = {frozenset(value) for value in components[other].values()}
                    if any(frozenset(components[side][key]) not in other_sets
                           for key in block["component_ids"]):
                        raise ValueError("The unchanged pool contains a changed component membership")
            exclusive = retained[side] - retained[other]
            saved = case["exclusive"][side]
            if (saved["count"] != len(exclusive)
                    or checked_nodes(saved["nodes"], n, case_id + "/exclusive") != exclusive
                    or set(saved["by_block"]) != set(blocks[side])):
                raise ValueError("Side-specific retained node counts are inconsistent")
            for block_id, block in blocks[side].items():
                if checked_nodes(saved["by_block"][block_id], n, case_id + "/exclusive block") != (
                        set(block["nodes"]) & exclusive):
                    raise ValueError("Side-specific retained nodes are assigned to the wrong display block")
        shared = retained["base"] & retained["guided"]
        if (case["shared_retained"]["count"] != len(shared)
                or checked_nodes(case["shared_retained"]["nodes"], n, case_id + "/shared") != shared):
            raise ValueError("The shared-retained population is inconsistent")
        expected_links = {(left_id, right_id): set(left["nodes"]) & set(right["nodes"])
            for left_id, left in blocks["base"].items()
            for right_id, right in blocks["guided"].items()
            if set(left["nodes"]) & set(right["nodes"])}
        actual_links = {}
        for link in case["links"]:
            pair = (link["base_id"], link["guided_id"])
            nodes = checked_nodes(link["nodes"], n, case_id + "/link")
            if (pair in actual_links or pair not in expected_links
                    or not nodes or nodes != expected_links[pair] or link["count"] != len(nodes)):
                raise ValueError("A shared-node connection differs from the exact component intersection")
            actual_links[pair] = nodes
        if (set(actual_links) != set(expected_links)
                or set().union(*actual_links.values()) != shared
                or sum(len(value) for value in actual_links.values()) != len(shared)):
            raise ValueError("The displayed membership connections do not conserve all shared nodes")
        if case_id == "d":
            pools = [[block for block in blocks[side].values() if block["kind"] == "identical_pool"]
                     for side in ("base", "guided")]
            if (any(len(pool) != 1 or pool[0]["component_count"] != 81
                    or pool[0]["oversized_count"] != 8 for pool in pools)
                    or set(pools[0][0]["nodes"]) != set(pools[1][0]["nodes"])):
                raise ValueError("The positive example must preserve the 81 unchanged components (8 oversized)")
        elif any(block["kind"] == "identical_pool" for side in blocks.values() for block in side.values()):
            raise ValueError("The counterexample must show oversized components and the small-component pool")
        block_maps[case_id] = blocks
        membership_checks[case_id] = {"graph": graph, "shared_nodes": len(shared),
            "base_only_retained": len(retained["base"] - retained["guided"]),
            "guided_only_retained": len(retained["guided"] - retained["base"]),
            "display_blocks": {side: len(value) for side, value in blocks.items()},
            "exact_membership_connections": len(actual_links)}
    source_paths = [path for path in payload["input_SHA256"]
                    if Path(path).as_posix().endswith("decycle_dataset_real_all/graphs_decycle_nodelist.json")]
    if len(source_paths) != 1:
        raise ValueError("The exact saved FVS source must be unambiguous")
    saved_candidates = json_read(source_paths[0])
    canonical = {name.replace("\\n", "").replace("\n", "").replace("\r", ""): value
                 for name, value in saved_candidates.items()}
    if len(canonical) != len(saved_candidates):
        raise ValueError("Saved candidate graph aliases collide")
    source_keys = {"NC": "DiffGDM_D_cost_estimator_natural_connectivity",
                   "Density": "DiffGDM_D_cost_estimator_density",
                   "GE": "DiffGDM_D_cost_estimator_global_efficiency"}
    for case_id, case in cases.items():
        saved = canonical[case["graph"]]
        if checked_nodes(saved["DiffGDM_base"], case["N"], "saved base") != set(case["fvs"]["base"]):
            raise ValueError("The base membership map differs from its saved FVS")
        for guide in case["methods"]:
            if checked_nodes(saved[source_keys[guide]], case["N"], "saved guide") != set(case["fvs"]["guided"]):
                raise ValueError("Guides combined in one case do not have identical saved FVS memberships")
    # Historical cases remain validated in the shared source, not displayed.
    from matplotlib.text import Text
    if any(text.get_fontsize() < 7 for figure in (fig, metric_fig) for text in figure.findobj(Text)
           if text.get_visible() and text.get_text()):
        raise ValueError("Every visible label in the independent analysis must be at least 7 pt")
    return {**payload["validation"], "all_81_scatter_offsets_checked": True,
            "all_27_source_direction_count_cells_checked": True,
            "all_three_source_count_matrices_sum_to_27": True,
            "exact_overlap_groups_checked": len(overlaps),
            "thin_overlap_leader_lines_checked": len(overlaps),
            "multiplicity_labels_do_not_overlap": True,
            "outcome_legend_labels_checked": True,
            "all_12_quadrant_labels_match_source_counts": True,
            "quadrant_text_minimum_marker_center_clearance_pt": min(quadrant_marker_clearances_pt),
            "metric_category_label_minimum_gap_pt": min(category_label_gaps_pt),
            "only_upper_left_and_lower_right_shaded": True,
            "unshaded_counterexamples_and_zero_change_points_retained": True,
            "common_axes_no_fitted_curves": True, "no_seed_statistics": True,
            "legacy_schema3_semantic_SHA256": payload["legacy_schema3_SHA256"],
            "legacy_schema2_semantic_SHA256": payload["legacy_schema2_SHA256"],
            "legacy_scatter_and_statistics_unchanged": True,
            "all_81_cached_exact_metric_records_checked": True,
            "all_three_metric_direction_tables_recounted": True,
            "all_81_metric_direction_observations_checked": True,
            "all_nine_metric_category_counts_checked": True,
            "metric_category_labels_include_network_counts": True,
            "both_triptychs_share_CFN_y_limits": True,
            "main_observation_count": 81, "supplementary_observation_count": 81,
            "paired_main_and_supplementary_CFN_values_checked": True,
            "metric_point_y_values_not_jittered": True,
            "metric_horizontal_offsets_bounded_by_0_34": True,
            "metric_outcome_colors_and_shapes_match_forest_panels": True,
            "metric_point_minimum_spacing_points": smallest_spacing,
            "metric_point_required_spacing_points": spacing,
            "metric_direction_category_counts": category_totals,
            "metric_decreased_outcomes": metric_counts,
            "exact_original_node_memberships_checked": True,
            "component_and_node_conservation_checked": True,
            "saved_shared_node_intersections_and_exclusives_checked": True,
            "historical_component_cases_retained_not_rendered": True,
            "all_visible_text_at_least_7pt": True,
            "two_triptychs_without_bars_or_case_panels": True,
            "layouts": {"main": fig._guidance_layout, "supplementary": metric_fig._guidance_layout},
            "component_case_checks": membership_checks}


def run_guidance_benefit(args):
    """Independent preview through the original statistics and drawing modules.

    The flag is deliberately outside the numbered-paper dispatch: it cannot
    overwrite Figure02, Figure03, Figure06, or silently join --all.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    payload = analysis_module().guidance_benefit_payload(WORKSPACE, include_metric_summary=True)
    print("Guidance relation: READY - 27 networks / 81 comparisons / 189 saved forests", flush=True)
    if args.check or args.list:
        return 0
    stems = {"main": OUTPUTS / "Guidance_Benefit_Analysis",
             "supplementary": OUTPUTS / "Guidance_Metric_Changes"}
    files_by_role = {role: [stem.with_suffix(s) for s in (".pdf", ".png")]
                     for role, stem in stems.items()}
    files = [path for paths in files_by_role.values() for path in paths]
    source_data = OUTPUTS / "Guidance_Benefit_Analysis_source.json"
    if not args.overwrite and any(path.exists() for path in [*files, source_data]):
        raise FileExistsError("Independent analysis already exists; --overwrite affects only this analysis")
    expected = protected_inputs({"guidance-benefit": {"input_SHA256": payload["input_SHA256"]}})
    if digest(json_read(source_data)) != digest(payload):
        raise ValueError("The split layout must reuse the complete existing source cache unchanged")
    expected[str(source_data.resolve())] = sha(source_data)
    # Protect every prior output as well as all accepted historical result data.
    for path in (OUTPUTS.iterdir() if OUTPUTS.is_dir() else ()):
        if path.is_file() and path.suffix.lower() in (".pdf", ".png") and path not in files:
            expected[str(path.resolve())] = sha(path)
    for path in [WORKSPACE / "DiffGDM_Final_Theorem_Proof.tex", *WORKSPACE.glob("*.pdf"),
                 *WORKSPACE.glob("*.md")]:
        expected[str(path.resolve())] = sha(path)
    expected[str(HERE / "README.md")] = sha(HERE / "README.md")
    before = hash_guard(expected)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    old = json_read(MANIFEST) if MANIFEST.is_file() else {"figures": {}}
    prior_figure_digest = digest(old.get("figures", {}))
    drawing = importlib.import_module("analysis_visi.LCC_size_removed_nodes")
    temporary = [p.with_suffix(p.suffix + ".tmp") for p in files]
    fig = metric_fig = None
    try:
        with matplotlib.rc_context({"font.family": "Arial", "font.size": 8,
                "pdf.fonttype": 42, "text.usetex": False, "axes.spines.top": False,
                "axes.spines.right": False, "axes.linewidth": .7}):
            fig, metric_fig = drawing.draw_guidance_benefit_analysis(payload)
            figures = {"main": fig, "supplementary": metric_fig}
            qa = {}
            for role, figure in figures.items():
                figure.set_dpi(300)
                qa[role] = renderer_check(figure)
            numeric = guidance_benefit_numeric_guard(fig, payload, metric_fig)
            for index, figure in enumerate(figures.values()):
                figure.savefig(temporary[2 * index], format="pdf", bbox_inches=None, metadata={
                    "Creator": "DiffGDM original plotting module; independent guidance analysis",
                    "CreationDate": None, "ModDate": None})
                figure.savefig(temporary[2 * index + 1], format="png", dpi=300, bbox_inches=None)
        after = hash_guard(expected)
        for source, target in zip(temporary, files):
            os.replace(source, target)
        exports = {role: {"status": "GENERATED", "layout": figure._guidance_layout,
                         "rendered_observation_count": 81, "independent_network_count": 27,
                         "output_SHA256": {path.name: sha(path) for path in files_by_role[role]},
                         "renderer_QA": qa[role], "source_data": source_data.name,
                         "numeric_QA": {"all_81_observations_checked": True,
                                        "all_27_networks_per_guide_retained": True,
                                        "paired_CFN_values_equal_in_both_exports": True,
                                        "original_source_cache_unchanged": True}}
                   for role, figure in figures.items()}
        export = {"status": "GENERATED", "generated_at": datetime.now().astimezone().isoformat(),
            "function": "analysis_visi.LCC_size_removed_nodes.draw_guidance_benefit_analysis",
            "statistics_function": "decycling_dismantling_analysis.guidance_benefit_payload",
            "reproduction": "python experiments_figure/generate_figures.py --guidance-benefit --overwrite",
            "input_SHA256": payload["input_SHA256"],
            "output_SHA256": {p.name: sha(p) for p in [*files, source_data]},
            "analysis_schema_version": payload["schema_version"],
            "layout": "main-forest-vertical-and-supplementary-metric-horizontal",
            "exports": exports,
            "rendered_observation_count": 162,
            "independent_network_count": 27,
            "historical_component_cases_rendered": False,
            "outcomes": payload["outcomes"], "direction_counts": payload["direction_counts"],
            "associations": payload["associations"],
            "legacy_schema2_SHA256": payload["legacy_schema2_SHA256"],
            "legacy_schema3_SHA256": payload["legacy_schema3_SHA256"],
            "metric_decrease_summary": {
                "records_SHA256": payload["metric_decrease_summary"]["records_SHA256"],
                "direction_counts": payload["metric_decrease_summary"]["direction_counts"],
                "decreased_outcomes": payload["metric_decrease_summary"]["decreased_outcomes"],
                "cache_only_no_spectral_or_shortest_path_recomputation": True},
            "component_case_definition": payload["component_case_definition"],
            "component_case_validation": payload["component_case_validation"],
            "component_cases": [{"id": case["id"], "graph": case["graph"],
                "methods": case["methods"], "threshold": case["threshold"],
                "base": case["base"], "guided": case["guided"],
                "membership_SHA256": case["membership_SHA256"],
                "exclusive_retained": {side: case["exclusive"][side]["count"]
                                       for side in ("base", "guided")}}
                for case in payload["component_cases"]],
            "renderer_QA": qa, "numeric_QA": numeric,
            "source_data": source_data.name,
            "runtime": {"python": sys.executable, "matplotlib": matplotlib.__version__, "backend": "Agg"},
            "protection": {"protected_file_count": len(expected), "before_digest": before,
                           "after_digest": after, "all_SHA256_match": before == after,
                           "previous_numbered_figure_entries_unchanged": True}}
        old.setdefault("analyses", {})["guidance-benefit"] = export
        if digest(old.get("figures", {})) != prior_figure_digest:
            raise ValueError("Independent analysis changed numbered-figure manifest entries")
        pending_manifest = MANIFEST.with_suffix(".json.tmp")
        pending_manifest.write_text(json.dumps(old, ensure_ascii=False, indent=2,
                                               allow_nan=False) + "\n", encoding="utf-8")
        os.replace(pending_manifest, MANIFEST)
        print("Guidance benefit: GENERATED -> " + str(files[0]), flush=True)
        print("Guidance metric changes: GENERATED -> " + str(files[2]), flush=True)
        print("All historical inputs and prior figure files passed SHA protection.", flush=True)
        return 0
    finally:
        for figure in (fig, metric_fig):
            if figure is not None:
                plt.close(figure)
        for path in temporary:
            if path.exists():
                path.unlink()


def _audit_windows_job(process):
    """Contain only this explicitly created worker, including its descendants."""
    if os.name != "nt":
        return None, None
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
                    ("Flags", wintypes.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                    ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("ReadCount", "WriteCount", "OtherCount", "ReadBytes", "WriteBytes", "OtherBytes")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("Basic", BasicLimits), ("Io", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise OSError(ctypes.get_last_error(), "Cannot create private topology-audit job")
    limits = ExtendedLimits()
    limits.Basic.Flags = 0x200 | 0x2000  # Aggregate job memory limit; kill on last handle close.
    limits.JobMemoryLimit = 4 * 1024 ** 3
    if (not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            or not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle)))):
        error = ctypes.get_last_error()
        kernel.CloseHandle(handle)
        raise OSError(error, "Cannot apply private topology-audit resource containment")
    return kernel, handle


def _guidance_topology_worker():
    """Private stdin-gated child. It never writes reports, plots, or manifests."""
    import psutil
    request = json.loads(sys.stdin.readline())
    if (request.get("kind") != "read-only-guidance-topology-audit"
            or Path(request.get("workspace", "")).resolve() != WORKSPACE.resolve()):
        raise ValueError("Invalid topology-audit worker request")
    remaining = float(request["remaining_seconds"])
    if not math.isfinite(remaining) or not 0 < remaining <= 1800:
        raise ValueError("Invalid private topology-audit time budget")
    deadline = time.monotonic() + remaining
    state = {"completed_forests": 0}
    process = psutil.Process()

    def progress(event):
        state.update(event)
        print(json.dumps({"progress": event}, ensure_ascii=True, allow_nan=False),
              file=sys.stderr, flush=True)

    def resource_check():
        if time.monotonic() >= deadline:
            raise TimeoutError("The 30-minute topology-audit budget is exhausted")
        if process.memory_info().rss > 4 * 1024 ** 3:
            raise MemoryError("The topology-audit worker exceeded 4 GiB RSS")

    try:
        # Library chatter is kept off the machine-readable result stream.
        with redirect_stdout(sys.stderr):
            result = analysis_module().guidance_topology_audit(
                WORKSPACE, progress=progress, deadline=deadline, resource_check=resource_check)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
        return 0
    except (TimeoutError, MemoryError) as exc:
        result = {"schema_version": 1, "analysis": "historical-guidance-topology-audit",
                  "status": "LIMIT_STOPPED", "reason": str(exc),
                  "completed_forests": state.get("completed_forests", 0),
                  "uncompleted": {"forest_count": 108 - state.get("completed_forests", 0),
                                  "last_progress": state}, "algorithm_failure": False}
        print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
        return 4
    except Exception as exc:
        result = {"schema_version": 1, "analysis": "historical-guidance-topology-audit",
                  "status": "FAILED_VALIDATION", "reason": str(exc),
                  "error_type": type(exc).__name__, "completed_forests": state.get("completed_forests", 0),
                  "uncompleted": {"forest_count": 108 - state.get("completed_forests", 0),
                                  "last_progress": state}, "algorithm_failure": False}
        print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
        return 1


def run_guidance_topology_audit(args):
    """Bounded local-only audit in a private worker; stdout is one JSON result."""
    import psutil
    started = time.monotonic()
    deadline = started + 1800
    expected = protected_inputs({})
    preservation = IMPORTS / "20261010_local_preservation/before_manifest.json"
    expected[str(preservation.resolve())] = sha(preservation)
    for path in MODEL_TEST.joinpath("dataset/dataset_real_all").glob("*.graphml"):
        expected[str(path.resolve())] = sha(path)
    for path in OUTPUTS.rglob("*"):
        if path.is_file():
            expected[str(path.resolve())] = sha(path)
    for path in [*WORKSPACE.glob("*.pdf"), *WORKSPACE.glob("*.md"),
                 *WORKSPACE.glob("*.tex"), *WORKSPACE.glob("*.txt"), HERE / "README.md"]:
        if path.is_file():
            expected[str(path.resolve())] = sha(path)
    output_paths = {str(path.resolve()) for path in OUTPUTS.rglob("*") if path.is_file()}
    before = hash_guard(expected)
    if args.check or args.list:
        print(json.dumps({"analysis": "historical-guidance-topology-audit", "status": "READY",
                          "execution": "not_started", "protected_file_count": len(expected),
                          "protection_SHA256": before}, ensure_ascii=True), flush=True)
        return 0
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        environment[name] = "4"
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--_guidance-topology-worker"]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", env=environment,
        cwd=str(CODE), creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        start_new_session=(os.name != "nt"))
    kernel, job = None, None
    owner = psutil.Process(child.pid)
    owned = {(owner.pid, owner.create_time()): owner}
    events = queue.Queue()
    stdout_parts, output_bytes, latest = [], 0, {"completed_forests": 0}
    limit_reason, peak_rss, stop_started = None, 0, None
    stderr_tail = []

    def reader(stream, channel):
        try:
            for line in iter(stream.readline, ""):
                events.put((channel, line))
        finally:
            events.put((channel, None))

    def stop_owned():
        if job is not None:
            kernel.TerminateJobObject(job, 4)
        else:
            for process in reversed(list(owned.values())):
                try:
                    if process.is_running():
                        process.kill()
                except psutil.NoSuchProcess:
                    pass

    try:
        kernel, job = _audit_windows_job(child)
        for channel, stream in (("stdout", child.stdout), ("stderr", child.stderr)):
            threading.Thread(target=reader, args=(stream, channel), daemon=True).start()
        request = {"kind": "read-only-guidance-topology-audit", "workspace": str(WORKSPACE),
                   "remaining_seconds": max(0, deadline - time.monotonic())}
        child.stdin.write(json.dumps(request) + "\n")
        child.stdin.flush()
        child.stdin.close()
        ended = set()
        while child.poll() is None or len(ended) < 2:
            try:
                channel, line = events.get(timeout=.15)
            except queue.Empty:
                channel, line = None, None
            if channel and line is None:
                ended.add(channel)
            elif channel == "stdout":
                output_bytes += len(line.encode("utf-8"))
                stdout_parts.append(line)
                if output_bytes > 8 * 1024 ** 2:
                    limit_reason = "Worker result exceeded the 8 MiB output safety bound"
            elif channel == "stderr":
                stderr_tail.append(line.strip()[:500])
                del stderr_tail[:-12]
                try:
                    event = json.loads(line).get("progress")
                except (ValueError, AttributeError):
                    event = None
                if isinstance(event, dict):
                    latest.update(event)
                    if event.get("stage") in ("validate_saved_candidates", "forest_complete",
                                              "paired_network_statistics"):
                        print(json.dumps({"progress": {key: event[key] for key in
                            ("stage", "graph", "method", "completed_forests") if key in event}},
                            ensure_ascii=True), file=sys.stderr, flush=True)
            if child.poll() is None:
                try:
                    for process in [owner, *owner.children(recursive=True)]:
                        owned[(process.pid, process.create_time())] = process
                except psutil.NoSuchProcess:
                    pass
            try:
                rss = sum(process.memory_info().rss for process in owned.values()
                          if process.is_running())
                peak_rss = max(peak_rss, rss)
                if rss > 4 * 1024 ** 3:
                    limit_reason = "Private audit process tree exceeded 4 GiB RSS"
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() >= deadline and (child.poll() is None or len(ended) < 2):
                limit_reason = "The 30-minute topology-audit budget is exhausted"
            if limit_reason:
                stop_started = stop_started or time.monotonic()
                stop_owned()
                if time.monotonic() - stop_started >= 5:
                    break
        child.wait(timeout=10)
        survivors = [process for process in owned.values() if process.is_running()]
        if survivors:
            stop_owned()
            _, survivors = psutil.wait_procs(survivors, timeout=10)
        if survivors:
            raise RuntimeError("The private topology-audit process tree did not fully exit")
        text = "".join(stdout_parts).strip()
        if limit_reason:
            result = {"schema_version": 1, "analysis": "historical-guidance-topology-audit",
                "status": "LIMIT_STOPPED", "reason": limit_reason,
                "completed_forests": latest.get("completed_forests", 0),
                "uncompleted": {"forest_count": 108 - latest.get("completed_forests", 0),
                                "last_progress": latest}, "algorithm_failure": False}
        elif text:
            result = json.loads(text)
        else:
            result = {"schema_version": 1, "analysis": "historical-guidance-topology-audit",
                "status": "WORKER_EXITED_UNCONFIRMED",
                "reason": "Private audit worker exited without a machine-readable result",
                "worker_returncode": child.returncode, "stderr_tail": stderr_tail,
                "completed_forests": latest.get("completed_forests", 0),
                "uncompleted": {"forest_count": 108 - latest.get("completed_forests", 0),
                                "last_progress": latest}, "algorithm_failure": False}
        if result.get("status") == "COMPLETE":
            if (child.returncode != 0 or result.get("completed_forests") != 108
                    or result.get("uncompleted") != []):
                raise ValueError("The worker's COMPLETE state is inconsistent")
        for path, value in result.get("input_SHA256", {}).items():
            if expected.get(str(Path(path).resolve())) != value:
                raise ValueError("The audit used an unprotected or changed input: " + path)
        after = hash_guard(expected)
        if {str(path.resolve()) for path in OUTPUTS.rglob("*") if path.is_file()} != output_paths:
            raise ValueError("The read-only topology audit created or deleted output files")
        result["protection"] = {"protected_file_count": len(expected),
            "before_SHA256": before, "after_SHA256": after, "all_SHA256_match": before == after,
            "outputs_and_manifest_unchanged": True}
        result["watchdog"] = {"elapsed_seconds": time.monotonic() - started,
            "time_limit_seconds": 1800, "RSS_limit_bytes": 4 * 1024 ** 3,
            "peak_process_tree_RSS_bytes": peak_rss, "maximum_compute_threads": 4,
            "private_process_tree_exited": True, "new_report_or_figure_files": False}
        print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
        return 0 if result.get("status") == "COMPLETE" else 4
    finally:
        if child.poll() is None:
            stop_owned()
            child.wait(timeout=10)
        if job is not None:
            kernel.CloseHandle(job)
        for stream in (child.stdout, child.stderr):
            stream.close()


def figure_number(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Figure numbers must be integers") from None
    if number == 12:
        raise argparse.ArgumentTypeError("Figure 12 is explicitly disabled; no old file will be deleted")
    if number == 1:
        raise argparse.ArgumentTypeError("Figure 1 is a method schematic, not an experiment")
    if number not in ALLOWED:
        raise argparse.ArgumentTypeError("Allowed experiment figures: 2-11 (never 12)")
    return number


def main(argv=None):
    request = list(sys.argv[1:] if argv is None else argv)
    if '--paper-all' in request or '--paper-figures' in request:
        from paper_figures import main as paper_main
        return paper_main(request)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Check/generate figures 2-11; report missing inputs")
    selection.add_argument("--figures", type=figure_number, nargs="+", help="Select paper figure numbers")
    selection.add_argument("--guidance-benefit", action="store_true",
                           help="Paired main/supplementary historical guidance triptychs; no numbered-figure changes")
    selection.add_argument("--guidance-topology-audit", action="store_true",
                           help="Read-only exact forest-topology audit; stdout JSON; 4 threads / 4 GiB / 30 min")
    selection.add_argument("--_guidance-topology-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--list", action="store_true", help="List figure inputs and readiness; no images")
    parser.add_argument("--check", action="store_true", help="Validate selected inputs only; write nothing")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated outputs only, never old figures")
    parser.add_argument("--profile", choices=PROFILES, default="accepted-tenseed",
                        help="Select accepted seed statistics or historical paper plus published single controls")
    args = parser.parse_args(argv)
    if not (args.all or args.figures or args.list or args.guidance_benefit
            or args.guidance_topology_audit or args._guidance_topology_worker):
        parser.print_help()
        return 0
    numbers = (list(dict.fromkeys(args.figures)) if args.figures else
               ([2, 3, 6] if args.profile == "original-plus-controls" else list(ALLOWED)))
    if args.profile == "original-plus-controls" and any(number not in (2, 3, 6) for number in numbers):
        parser.error("original-plus-controls only supports Figures 2, 3 and 6")
    if args.list and args.overwrite:
        parser.error("--list does not overwrite files")
    if args.guidance_topology_audit and args.overwrite:
        parser.error("--guidance-topology-audit is read-only and does not accept --overwrite")
    os.environ["MPLBACKEND"] = "Agg"
    sys.dont_write_bytecode = True
    if str(MODEL_TEST) not in sys.path:
        sys.path.insert(0, str(MODEL_TEST))
    try:
        if args._guidance_topology_worker:
            return _guidance_topology_worker()
        if args.guidance_topology_audit:
            return run_guidance_topology_audit(args)
        dependency_check()
        import matplotlib
        matplotlib.use("Agg", force=True)
        if args.guidance_benefit:
            return run_guidance_benefit(args)
        checks = {number: preflight(number, args.profile) for number in numbers}
        for number, info in checks.items():
            print(f"Figure {number:02d}: {info['status']} - {info['title']}")
            if "reason" in info:
                print("  " + info["reason"])
        failed = any(item["status"] == "FAILED_VALIDATION" for item in checks.values())
        blocked = any(item["status"].startswith("BLOCKED") for item in checks.values())
        if args.list or args.check:
            return 1 if failed else (3 if blocked and args.check else 0)
        for number, info in checks.items():
            if info["status"] == "READY" and not args.overwrite and any(
                    (OUTPUTS / f"Figure{number:02d}{suffix}").exists() for suffix in (".pdf", ".png")):
                info.update(status="EXISTS_REQUIRES_OVERWRITE", reason="Use --overwrite to replace the isolated PDF/PNG")
                failed = True
        if not any(info["status"] == "READY" for info in checks.values()):
            return 1 if failed else 3
        expected = protected_inputs(checks)
        before = hash_guard(expected)
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        run = {}
        original_cwd = Path.cwd()
        os.chdir(MODEL_TEST)
        try:
            with matplotlib.rc_context({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42,
                                        "text.usetex": False}):
                for number, info in checks.items():
                    if info["status"] != "READY":
                        continue
                    try:
                        with redirect_stdout(io.StringIO()):
                            export = render(number, args.overwrite, args.profile)
                        run[number] = {"status": "GENERATED", "export": export}
                        print(f"Figure {number:02d}: GENERATED -> {OUTPUTS / f'Figure{number:02d}.pdf'}", flush=True)
                    except Exception as exc:
                        run[number] = {"status": "FAILED_RENDER", "reason": str(exc)}
                        failed = True
                        print(f"Figure {number:02d}: FAILED_RENDER - {exc}", flush=True)
        finally:
            os.chdir(original_cwd)
        after = hash_guard(expected)
        runtime = {"python": sys.executable, "python_version": sys.version.split()[0],
                   "matplotlib": matplotlib.__version__, "backend": "Agg", "plotting_seed": 0,
                   "profile": args.profile}
        save_manifest(checks, run, {"protected_file_count": len(expected), "before_digest": before,
                                  "after_digest": after, "all_SHA256_match": before == after,
                                  "input_SHA256": expected}, runtime)
        print(f"Generated {sum(item['status'] == 'GENERATED' for item in run.values())}/{len(numbers)} selected figures; input SHA protection passed.")
        return 1 if failed else (3 if blocked else 0)
    except Exception as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
