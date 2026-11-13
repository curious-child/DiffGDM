"""Generate the six-panel Toy-network experiment used in Figure 5.

Run from any working directory. The default command evaluates the three
methods; --plot-only redraws saved node sets, and --check checks the graph.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random

import networkx as nx


ROOT = Path(__file__).resolve().parent
GRAPH_SEED = 19
TARGET_PERCENT = 10
METHODS = (
    ("CoreDiffGDM_base", "DiffGDM_base"),
    ("CoreDiffGDM", "DiffGDM"),
    ("BPD", "BPD"),
)


def random_connected_caveman(l, k, seed=None):
    """Original generator: reconnect each clique to the preceding clique."""
    if seed is not None:
        random.seed(seed)

    graph = nx.caveman_graph(l, k)
    for start in range(0, l * k, k):
        this_clique = list(range(start, start + k))
        u = random.choice(this_clique)
        other_start = (start - k) % (l * k)
        other_clique = list(range(other_start, other_start + k))
        v = random.choice(other_clique)
        if u != v:
            this_clique.remove(u)
            graph.remove_edge(u, random.choice(this_clique))
            graph.add_edge(u, v)
    return graph


def build_toy_graph():
    """Return the original 20-clique, four-nodes-per-clique, seed-19 graph."""
    graph = random_connected_caveman(20, 4, seed=GRAPH_SEED)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return nx.convert_node_labels_to_integers(graph)


def graph_record(graph):
    edges = [list(edge) for edge in graph.edges()]
    canonical = sorted(tuple(sorted(edge)) for edge in edges)
    digest = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode()
    ).hexdigest()
    return {"seed": GRAPH_SEED, "cliques": 20, "clique_size": 4,
            "nodes": list(graph.nodes()), "edges": edges,
            "edge_sha256": digest}


def infer_parameters():
    return {
        "gradient_estimator_type": "cost_estimator",
        "guided_model_path": "method/modelGym/Diffusion_model/robustness_metrics_models",
        "use_tag": True,
        "parallel_sampling": 4,
    }


@contextmanager
def experiment_directory():
    """Keep the original relative model and native-tool paths working."""
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def decycle_dismantle_method(method, graph, target, **kwargs):
    from method.method import DecycleList, NodeBreakList

    # Earlier releases use "CoreDiffGDM" for the decycling route.
    if method.startswith("DiffGDM") and "CoreDiffGDM" in inspect.getsource(DecycleList):
        method = "Core" + method
    decycle_nodes, _ = DecycleList(graph, method, graph_name=None, **kwargs)
    dismantle_nodes, _ = NodeBreakList(
        graph, method=method, target=target, graph_name=None, **kwargs
    )
    return list(map(int, decycle_nodes)), list(map(int, dismantle_nodes))


def make_figure():
    from matplotlib import pyplot as plt

    plt.rc("font", family="Arial", size=12)
    return plt.subplots(len(METHODS), 2, figsize=(8, 12))


def draw_method(axes, row, graph, method, decycle_nodes, dismantle_nodes):
    from analysis_visi.network_dismantling_topology import draw_dimantle_nodes

    display_name = method.replace("Core", "")
    for column, (description, nodes) in enumerate((
        ("Decycling", decycle_nodes), ("Dismantling", dismantle_nodes)
    )):
        axis = axes[row][column]
        draw_dimantle_nodes(axis, graph, nodes)
        axis.set_title(f"{description} nodes of {display_name}: {len(nodes)}")
        axis.text(-0.05, 1.02, "abcdef"[2 * row + column],
                  transform=axis.transAxes, fontsize=16, fontweight="bold",
                  va="top", ha="left")


def validate_method(graph, row):
    for key in ("decycle_nodes", "dismantle_nodes"):
        nodes = row[key]
        if len(nodes) != len(set(nodes)) or not set(nodes) <= set(graph):
            raise ValueError(f"Invalid {key}: {row['method']}")
    forest = graph.copy()
    forest.remove_nodes_from(row["decycle_nodes"])
    if forest.number_of_nodes() and not nx.is_forest(forest):
        raise ValueError(f"Invalid FVS: {row['method']}")
    remainder = graph.copy()
    remainder.remove_nodes_from(row["dismantle_nodes"])
    largest = max(map(len, nx.connected_components(remainder)), default=0)
    if largest > math.ceil(len(graph) * TARGET_PERCENT / 100):
        raise ValueError(f"Dismantling threshold not met: {row['method']}")


def run_experiment():
    from utils.utils import seed_torch

    seed_torch(GRAPH_SEED)
    graph = build_toy_graph()
    parameters = infer_parameters()
    figure, axes = make_figure()
    rows = []
    for row, (original_name, api_name) in enumerate(METHODS):
        print(f"Running {api_name}", flush=True)
        decycle, dismantle = decycle_dismantle_method(
            api_name, graph, TARGET_PERCENT, DiffDGM_infer_param=parameters
        )
        result = {"method": api_name, "original_method": original_name,
                  "decycle_nodes": decycle, "dismantle_nodes": dismantle}
        validate_method(graph, result)
        # Preserve the original solve/draw ordering: the original drawing
        # function sets its own random seed before the next method is solved.
        draw_method(axes, row, graph, original_name, decycle, dismantle)
        rows.append(result)
    return figure, {"schema": "diffgdm-toy-figure-v1", "graph": graph_record(graph),
                    "target_percent": TARGET_PERCENT,
                    "inference_parameters": parameters, "results": rows}


def redraw_saved(record):
    graph = nx.Graph()
    graph.add_nodes_from(record["graph"]["nodes"])
    graph.add_edges_from(record["graph"]["edges"])
    if record["graph"] != graph_record(graph) or record["target_percent"] != TARGET_PERCENT:
        raise ValueError("Saved graph or threshold differs from the Toy experiment")
    expected = [name for _, name in METHODS]
    if [row["method"] for row in record["results"]] != expected:
        raise ValueError("Saved method order differs from the Toy experiment")
    figure, axes = make_figure()
    for index, row in enumerate(record["results"]):
        validate_method(graph, row)
        draw_method(axes, index, graph, row["method"],
                    row["decycle_nodes"], row["dismantle_nodes"])
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check the seed-19 graph only")
    mode.add_argument("--plot-only", type=Path, metavar="JSON", help="Redraw saved node sets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result/DiffGDM/toy_model")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        graph = build_toy_graph()
        print(json.dumps({"seed": GRAPH_SEED, "nodes": len(graph),
                          "edges": graph.number_of_edges(),
                          "connected": nx.is_connected(graph),
                          "edge_sha256": graph_record(graph)["edge_sha256"]}, indent=2))
        return 0

    destination = args.output_dir.resolve()
    saved = args.plot_only.resolve() if args.plot_only else None
    outputs = [destination / ("Figure05" + suffix) for suffix in (".pdf", ".png")]
    if not saved:
        outputs.append(destination / "Figure05.json")
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Toy output exists; use --overwrite to replace it")
    if saved and saved in outputs:
        raise ValueError("Choose an output location different from the saved input")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    with experiment_directory():
        if saved:
            record = json.loads(saved.read_text(encoding="utf-8-sig"))
            figure = redraw_saved(record)
        else:
            figure, record = run_experiment()
        destination.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(outputs[0], metadata={"CreationDate": None, "ModDate": None})
        figure.savefig(outputs[1], dpi=100)
        if not saved:
            outputs[2].write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if args.show:
            plt.show()
        plt.close(figure)
    print(f"Toy figure saved to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
