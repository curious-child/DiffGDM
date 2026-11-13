import math
import json
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

import matplotlib


def _s3_seed_cli():
    """Explicit server-only S3 entry; importing/plotting never starts experiments."""
    import argparse
    from pathlib import Path
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-seed-plan", required=True)
    parser.add_argument("--s3-run-dir", required=True)
    parser.add_argument("--s3-continue-after-pilot", action="store_true")
    args = parser.parse_args()
    with open(args.s3_seed_plan, encoding="utf-8") as stream:
        if json.load(stream).get("postprocess") != "direct":
            raise ValueError("Hyperparameter sensitivity requires explicit postprocess='direct'; legacy decycle records remain historical only")
    workspace = Path(__file__).resolve().parents[2]
    if not (workspace / "experiment_ops" / "s3_seed_run.py").is_file():
        raise RuntimeError("Place this entry in the source workspace containing experiment_ops")
    sys.path.insert(0, str(workspace))
    from experiment_ops.s3_seed_run import main
    cmd = ["--plan", args.s3_seed_plan, "--run-dir", args.s3_run_dir]
    if args.s3_continue_after_pilot:
        cmd.append("--continue-after-pilot")
    return main(cmd)


def _s3_ab_trace_cli():
    """Explicit original-entry handoff for the approved single-seed a/b run."""
    import argparse
    import sys
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-ab-run-dir', type=Path, required=True)
    args = parser.parse_args()
    with (args.s3_ab_run_dir / 'plan.json').open(encoding='utf-8') as stream:
        if json.load(stream).get('postprocess') != 'direct':
            raise ValueError("Hyperparameter sensitivity requires explicit postprocess='direct'")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiment_ops.s3_ab_trace_run import controller
    return controller(args.s3_ab_run_dir.resolve())


def file_graph_filter_setup(dataset_path,graph_filter,file_filters):
    for file_filter in file_filters:
        for file in sorted(glob(dataset_path + '/' + file_filter + ".graphml")):
            graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
            graph_filter.append(graph_name)

def dismantle_statistics_methods(target,method,dataset_path,Rein,
                                 graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,
                                 save_path="result",filter="*",**kwargs):
    import igraph
    import networkx as nx
    from method.method import NodeBreakList

    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)

    graphs_name = []
    #graphs_name_filter = ["subelj_jdk_jdk"]
    graphs_name_filter = []
    graph_max_nodes = 200000  # 系统最大整数值 可视为Inf

    for file in sorted(glob(dataset_path + '/' + filter + ".graphml")):

        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
        G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
        G.remove_nodes_from(list(nx.isolates(G)))  # 去孤立节点
        G = nx.convert_node_labels_to_integers(G)  # 将节点编号从0开始,重新编号
        #features_for_graph(G, features_list=features)
        G_nodes=G.number_of_nodes()
        if G_nodes<=graph_max_nodes:
            graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
            if graph_name in graphs_name_filter:
                continue
            graphs_name.append(graph_name)
            print("-----------networks------------")
            print("{}_size:{}".format(graph_name, G_nodes))
            if graphs_dismantle_nodelist.get(graph_name) is not None:

                print("the result {}  already exists".format( graph_name))

                continue
            remove_list, cost_time = NodeBreakList(G, method=method,
                                                   graph_name=graph_name,
                                                   target=target,
                                                   models_REIN=Rein,**kwargs)
            graphs_dismantle_time[graph_name] = cost_time
            graphs_dismantle_size[graph_name] = len(remove_list) / nx.number_of_nodes(G)
            graphs_dismantle_nodelist[graph_name] = remove_list

def dismantle_model_params_experiment(experiment_name,analysis_param,params_list):
    """Sensitivity uses ranked direct removal followed by original reinsertion.

    Never inherit NodeBreakList's decycling default or reuse route-unknown caches.
    """
    Refine = True  # 网络节点重插
    target = 1  # 拆解阈值 （%）
    method = "DiffGDM"  #
    dataset_type = "synth_large"
    dataset_path = "dataset/dataset_{}".format(dataset_type)
    save_path = os.path.join("result/{}".format(experiment_name), "dismantle_analysis_direct_dataset_{}".format(dataset_type))
    os.makedirs(save_path, exist_ok=True)

    #params_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    if os.path.exists(
            save_path + '/dismantle_perform_{}.json'.format(analysis_param)):
        with open(save_path + '/dismantle_perform_{}.json'.format(analysis_param), "r") as f:
            dismantle_perform_params = json.load(f)
    else:
        dismantle_perform_params = {}
    for params_value in params_list:
        #  print(type(params_value))
        if str(params_value) in dismantle_perform_params.keys():
            continue

        DiffDGM_infer_param = {
            "gradient_estimator_type": "cost_estimator",
            "guided_model_path": "method/modelGym/Diffusion_model/robustness_metrics_models",
            "use_tag": True,
            "parallel_sampling": 4,
        }
        DiffDGM_infer_param.update({analysis_param: params_value})
        if os.path.exists(
                save_path + '/graphs_dismantling_size.json'):
            with open(save_path + '/graphs_dismantling_size.json', 'r') as f:
                graphs_dismantle_size = json.load(f)
            with open(save_path + "/graphs_dismantling_time.json", 'r') as f:
                graphs_dismantle_time = json.load(f)
            with open(save_path + "/graphs_dismantling_nodelist.json", 'r') as f:
                graphs_dismantle_nodelist = json.load(f)
            os.remove(os.path.join(save_path, 'graphs_dismantling_size.json'))
            os.remove(os.path.join(save_path, 'graphs_dismantling_time.json'))
            os.remove(os.path.join(save_path, 'graphs_dismantling_nodelist.json'))
        else:
            graphs_dismantle_time = {}
            graphs_dismantle_size = {}
            graphs_dismantle_nodelist = {}
        try:
            dismantle_statistics_methods(graphs_dismantle_time=graphs_dismantle_time,
                                         graphs_dismantle_size=graphs_dismantle_size,
                                         graphs_dismantle_nodelist=graphs_dismantle_nodelist,
                                         target=target, method=method,
                                         dataset_path=dataset_path, save_path=save_path,
                                         Rein=Refine, DiffDGM_infer_param=DiffDGM_infer_param,
                                         DiffGDM_postprocess="direct")
        except Exception as e:
            with open(save_path + '/graphs_dismantling_size.json', 'w') as f:
                json.dump(graphs_dismantle_size, f, indent=4, separators=(',', ':'))
            with open(save_path + '/graphs_dismantling_time.json', 'w') as f:
                json.dump(graphs_dismantle_time, f, indent=4, separators=(',', ':'))
            with open(save_path + '/graphs_dismantling_nodelist.json', 'w') as f:
                json.dump(graphs_dismantle_nodelist, f, indent=4, separators=(',', ':'))
            raise ValueError("exception:{}".format(e))
        dismantle_perform_params[params_value] = graphs_dismantle_size
        with open(save_path + '/dismantle_perform_{}.json'.format(analysis_param), 'w') as f:
            json.dump(dismantle_perform_params, f, indent=4, separators=(',', ':'))

def plt_model_params_sensitivity(ax,model_param,save_path, *, seed=None, seed_statistics=None, family_breakdown=False, family_name=None, show_band=True, full_graph_mean=False):
    """Plot existing results, preserving the historical sampling/shading rule.

    An explicit seed uses a private Python RNG and does not change the caller's
    global random state. The shaded range is historical 0.1 * mean resampling SD,
    not a ten-inference-seed SD or a confidence interval.
    """
    if full_graph_mean:
        if seed_statistics is not None or family_breakdown or show_band:
            raise ValueError('Full historical means require mean-only plotting, without seed statistics')
        with open(os.path.join(save_path, 'dismantle_perform_{}.json'.format(model_param)), encoding='utf-8') as stream:
            records = json.load(stream)
        keys = sorted(records, key=float)
        values = [float(k) for k in keys]
        means, family_means, selected_ids, ignored = [], {}, {}, {}
        for key in keys:
            row = records[key]
            current = {}
            for prefix in ('barabasi', 'erdos', 'small', 'static'):
                ids = sorted(g for g in row if g.startswith(prefix))
                if len(ids) != 100 or (prefix in selected_ids and selected_ids[prefix] != ids):
                    raise ValueError('Require the same 100 historical graphs per family: ' + prefix)
                numbers = [row[g] for g in ids]
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and 0 <= v <= 1 for v in numbers):
                    raise ValueError('Invalid historical CFN')
                selected_ids[prefix] = ids
                current[prefix] = math.fsum(numbers) / 100 * 100
            family_means[key] = current
            means.append(math.fsum(current.values()) / 4)
            ignored[key] = sorted(set(row) - {g for ids in selected_ids.values() for g in ids})
        ax.plot(values, means, '.-', color='#2878A3', linewidth=1.4, markersize=5)
        ax.set_xticks(values)
        ax.set_xlabel({'Rewrite_Ratio': 'Rewrite ratio', 'Inference_Diffusion_Steps': 'Denoising steps',
                      'GNN_Dims': 'Hidden dimension', 'GNN_Layers': 'GNN layers', 'Rewrite_Steps': 'Rewrite steps'}[model_param])
        ax.set_ylabel('CFN (%)')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.spines[['top', 'right']].set_visible(False)
        return {'parameters': values, 'mean': means, 'graphs_per_family': 100,
                'graphs_per_point': 400, 'family_mean_percent': family_means,
                'graph_ids': selected_ids, 'excluded_nonfamily_keys': ignored,
                'mean_rule': 'Equal mean of BA/ER/WS/PL family means, all 100 saved graphs per family',
                'shade': None, 'resampling': False, 'new_inference': False}
    if family_breakdown:
        with open(os.path.join(save_path, 'dismantle_perform_{}.json'.format(model_param)), encoding='utf-8') as stream:
            records = json.load(stream)
        keys = sorted(records, key=float)
        values = [float(k) for k in keys]
        result = {}
        for prefix, label, color, marker in [('barabasi', 'BA', '#2878A3', 'o'),
                ('erdos', 'ER', '#CB843E', '^'), ('small', 'WS', '#409787', 's'),
                ('static', 'PL', '#8977AD', 'D')]:
            if family_name is not None and label != family_name:
                continue
            means = []; ids = None
            for key in keys:
                group = {g: v for g, v in records[key].items() if g.startswith(prefix)}
                if len(group) != 100 or (ids is not None and set(group) != ids):
                    raise ValueError('Historical family graph coverage differs: ' + label)
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and np.isfinite(v) and 0 <= v <= 1 for v in group.values()):
                    raise ValueError('Invalid historical CFN')
                ids = set(group)
                means.append(float(np.mean(list(group.values()))) * 100)
            ax.plot(values, means, marker=marker, color=color, linewidth=1.3, markersize=3.5, label=label)
            result[label] = {'mean_percent': means, 'graphs': sorted(ids), 'n': 100}
        ax.set_xticks(values)
        ax.set_xlabel({'Rewrite_Ratio': 'Rewrite ratio', 'Inference_Diffusion_Steps': 'Denoising steps',
                      'GNN_Dims': 'Hidden dimension', 'GNN_Layers': 'GNN layers'}[model_param])
        ax.set_ylabel('CFN (%)')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.spines[['top', 'right']].set_visible(False)
        return {'parameters': values, 'families': result, 'statistic': 'Direct mean of 100 saved graph-level CFNs per family; no seed SD inferred'}
    if seed_statistics is not None:
        import statistics as st
        panel = {"Rewrite_Ratio": "a", "Inference_Diffusion_Steps": "b",
                 "GNN_Dims": "c", "GNN_Layers": "d"}[model_param]
        expected = {"a": [i / 10 for i in range(1, 9)], "b": list(range(5, 36, 5)),
                    "c": list(range(16, 129, 16)), "d": list(range(2, 17, 2))}[panel]
        rows = sorted((r for r in seed_statistics["points"] if r["panel"] == panel), key=lambda r: r["value"])
        if [r["value"] for r in rows] != expected:
            raise ValueError("Missing or duplicated five-seed parameter points")
        for row in rows:
            if (not row["complete"] or row["seed_counts"] != [40] * 5 or len(row["seed_means"]) != 5
                    or not all(np.isfinite(row["seed_means"]))
                    or not math.isclose(st.mean(row["seed_means"]), row["mean_cfn"], abs_tol=1e-14)
                    or not math.isclose(st.stdev(row["seed_means"]), row["sd_cfn"], abs_tol=1e-14)):
                raise ValueError("Incomplete or mismatched five-seed statistics")
        x = [r["value"] for r in rows]
        mean = np.array([r["mean_cfn"] for r in rows]) * 100
        sd = np.array([r["sd_cfn"] for r in rows]) * 100
        ax.plot(x, mean, '.-', color='#2878A3', linewidth=1.4, markersize=5)
        if show_band:
            ax.fill_between(x, mean - sd, mean + sd, color='#2878A3', alpha=.18, linewidth=0)
        ax.set_xticks(x)
        ax.set_xlabel({"a": "Rewrite ratio", "b": "Denoising steps",
                       "c": "Hidden dimension", "d": "GNN layers"}[panel])
        ax.set_ylabel("CFN (%)")
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.spines[['top', 'right']].set_visible(False)
        return {"parameters": x, "mean": mean.tolist(), "sd": sd.tolist(),
                "seed_means": [r["seed_means"] for r in rows], "graphs_per_seed": 40,
                "seeds": [0, 1, 2, 3, 4], "shade_rule": "sample SD of five 40-graph means; ddof=1; no scaling"}
    with open(os.path.join(save_path, 'dismantle_perform_{}.json'.format(model_param)), "r") as f:
        dismantle_perform_params = json.load(f)
    if not isinstance(dismantle_perform_params, dict) or not dismantle_perform_params:
        raise ValueError("No parameter results for {}".format(model_param))
    params_list = list(dismantle_perform_params.keys())
    plt_dismantling_static = []
    plt_dismantling_static_std=[]
    family_counts = {}
    sample = random.sample if seed is None else random.Random(seed).sample
    graph_types=[
        "barabasi",
        "erdos",
         "small",
        "static"
    ]
    for params_value in params_list:
        std=[]
        mean=[]
        records = dismantle_perform_params[params_value]
        if not isinstance(records, dict):
            raise ValueError("Invalid graph results for {}={}".format(model_param, params_value))
        family_counts[params_value] = {}
        for graph_type in graph_types:

            graph_type_record=[]
            average_dismantle_steps = [v for k, v in records.items()
                                      if k.startswith(graph_type)]
            if len(average_dismantle_steps) < 99:
                raise ValueError("{}={} / {} needs at least 99 graph results; found {}".format(
                    model_param, params_value, graph_type, len(average_dismantle_steps)))
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and np.isfinite(v) for v in average_dismantle_steps):
                raise ValueError("Non-finite or non-numeric results for {}={} / {}".format(
                    model_param, params_value, graph_type))
            family_counts[params_value][graph_type] = len(average_dismantle_steps)
            for _ in range(100):
                random_average_dismantle_steps = sample(average_dismantle_steps, 99)
                average_dismantle = np.mean(random_average_dismantle_steps)
                graph_type_record.append(average_dismantle*100)
            mean.append(np.mean(graph_type_record))
            std.append(np.std(graph_type_record))


        plt_dismantling_static.append(np.mean(mean))
        plt_dismantling_static_std.append(0.1*np.mean(std))

    plt_dismantling_static = np.array(plt_dismantling_static)
    plt_dismantling_static_std = np.array(plt_dismantling_static_std)
   # print(plt_dismantling_static_std)
    ax.plot(params_list, plt_dismantling_static ,".-"         )

    if show_band:
        ax.fill_between(params_list, plt_dismantling_static-plt_dismantling_static_std, plt_dismantling_static+plt_dismantling_static_std, color='#ff7f0e', alpha=0.2)
    ax.set_ylabel("Average size of dismantling set (%)")

   # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    ax.set_xlabel("Parameter Value")
    # axes[n // 2][n % 2].tick_params(axis='y', rotation=-45)
    ax.set_title(model_param.replace("_", " "))
    return {
        "model_parameter": model_param,
        "parameters": params_list,
        "mean": plt_dismantling_static.tolist(),
        "shade": plt_dismantling_static_std.tolist(),
        "unit": "percentage_points",
        "seed": seed,
        "graph_family_counts": family_counts,
        "sampling_rule": "99 graphs without replacement per family, 100 repetitions",
        "mean_rule": "equal mean of four graph-family resampled means",
        "shade_rule": "0.1 * mean of four within-family resampling population SDs (ddof=0)",
        "not_ten_seed_sd": True,
    }


def draw_paper_parameter_sensitivity(result_dir, *, rewrite_only=False, seed=0, statistics_file=None, family_breakdown=False, family_name=None, ab_only=False, show_band=True, full_graph_mean=False):
    """Assemble paper panels using the original plotting function; no export."""
    if not os.path.isdir(result_dir):
        raise FileNotFoundError("Parameter-result directory does not exist: {}".format(result_dir))
    model_params = (["Rewrite_Steps"] if rewrite_only else
                    ["Rewrite_Ratio", "Inference_Diffusion_Steps", "GNN_Dims", "GNN_Layers"])
    if ab_only:
        if not family_breakdown or rewrite_only or statistics_file is not None:
            raise ValueError('Single-seed a/b mode requires the validated family results')
        model_params = model_params[:2]
    seed_statistics = None
    if family_breakdown and (statistics_file is not None or rewrite_only):
        raise ValueError('Historical family mode must remain separate from five-seed statistics')
    if family_name is not None and (not family_breakdown or family_name not in ('BA', 'ER', 'WS', 'PL')):
        raise ValueError('Choose one valid historical graph family')
    if statistics_file is not None:
        if rewrite_only:
            raise ValueError("Five-seed export requires all four panels")
        with open(statistics_file, encoding='utf-8') as stream:
            seed_statistics = json.load(stream)
        if (seed_statistics.get("complete") is not True or seed_statistics.get("successful") != 6200
                or seed_statistics.get("expected") != 6200 or len(seed_statistics.get("points", [])) != 31):
            raise ValueError("Require the complete accepted 6200-trial statistics")
    for model_param in ([] if seed_statistics else model_params):
        input_path = os.path.join(result_dir, 'dismantle_perform_{}.json'.format(model_param))
        if not os.path.isfile(input_path):
            raise FileNotFoundError(input_path)
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9 if seed_statistics or family_breakdown or full_graph_mean else 12, "pdf.fonttype": 42}):
        if rewrite_only:
            fig, ax = plt.subplots(figsize=(5.5, 3.4) if full_graph_mean else None)
            axes = [ax]
        elif ab_only:
            fig, panel_axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
            axes = panel_axes.ravel()
        else:
            fig, panel_axes = plt.subplots(2, 2, figsize=(7.1, 5.1) if seed_statistics or family_breakdown or full_graph_mean else (10, 8))
            axes = panel_axes.ravel()
        panel_statistics = {}
        try:
            for n, (model_param, ax) in enumerate(zip(model_params, axes)):
                panel_statistics[model_param] = plt_model_params_sensitivity(
                    ax=ax, model_param=model_param, save_path=result_dir, seed=seed,
                    seed_statistics=seed_statistics, family_breakdown=family_breakdown, family_name=family_name,
                    show_band=show_band, full_graph_mean=full_graph_mean)
                if ab_only:
                    if n == 0:
                        from matplotlib.ticker import MaxNLocator, FormatStrFormatter
                        plotted = np.concatenate([line.get_ydata() for line in ax.lines])
                        low, high = float(plotted.min()), float(plotted.max())
                        span = high - low
                        if span < 1e-10:
                            ax.set_ylim((low + high) / 2 - .001, (low + high) / 2 + .001)
                        else:
                            padding = max(span * .15, 1e-6)
                            ax.set_ylim(low - padding, high + padding)
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
                    else:
                        lower, upper = ax.get_ylim()
                        if upper - lower < .02:
                            middle = (lower + upper) / 2
                            ax.set_ylim(middle - .01, middle + .01)
                if not rewrite_only:
                    ax.text(0 if family_name else (-0.13 if seed_statistics or family_breakdown else -0.05),
                            1.08 if family_name else (1.06 if seed_statistics or family_breakdown else 1.1),
                            "abcd"[n], transform=ax.transAxes,
                            fontsize=11 if seed_statistics or family_breakdown or full_graph_mean else 16, fontweight='bold', va='top', ha='left')
            if family_breakdown:
                if family_name:
                    fig.suptitle(family_name + ' networks', fontsize=11)
                else:
                    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
            fig.tight_layout(rect=(0, 0, 1, .88 if ab_only else .94) if family_breakdown else (0, 0, 1, 1))
        except Exception:
            plt.close(fig)
            raise
    return fig, {"seed": None if seed_statistics or full_graph_mean else seed, "panels": panel_statistics,
                 "historical_sampling_preserved": seed_statistics is None and not family_breakdown and not full_graph_mean}


def _s3_historical_mean_cli():
    """Read-only mean previews of the original records; no model imports or solves."""
    import argparse
    import hashlib
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-mean-results', type=Path, required=True)
    parser.add_argument('--s3-mean-output', type=Path, required=True)
    parser.add_argument('--s3-rewrite-only', action='store_true')
    args = parser.parse_args()
    files = sorted(args.s3_mean_results.glob('dismantle_perform_*.json'))
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    fig, summary = draw_paper_parameter_sensitivity(args.s3_mean_results,
        rewrite_only=args.s3_rewrite_only, show_band=False, full_graph_mean=True)
    for ax, data in zip(fig.axes, summary['panels'].values()):
        if ax.collections or len(ax.lines) != 1 or not np.array_equal(ax.lines[0].get_ydata(), data['mean']):
            raise ValueError('Unexpected band or curve/mean mismatch')
    args.s3_mean_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.s3_mean_output) + '.pdf')
    fig.savefig(str(args.s3_mean_output) + '.png', dpi=300)
    plt.close(fig)
    if before != {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}:
        raise ValueError('Historical source data changed during export')
    summary.update(input_sha256=before, show_band=False,
                   data_scope='Historical records only; not the later decycle reruns',
                   route_evidence='Historical scheduler selected the direct interface; per-record seed/ranking provenance is not available')
    args.s3_mean_output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Exported original-record means: four families x 100 graphs; no resampling, no bands, no inference')


def _s3_family_cli():
    import argparse
    from pathlib import Path
    import hashlib
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-family-results', type=Path, required=True)
    parser.add_argument('--s3-family-output', type=Path, required=True)
    parser.add_argument('--s3-family-name', choices=['BA', 'ER', 'WS', 'PL'])
    parser.add_argument('--s3-family-ab-only', action='store_true')
    parser.add_argument('--s3-family-acceptance', type=Path)
    args = parser.parse_args()
    if args.s3_family_ab_only:
        if args.s3_family_acceptance is None:
            raise ValueError('Independent acceptance is required for the new single-seed a/b figures')
        accepted = json.loads(args.s3_family_acceptance.read_text(encoding='utf-8'))
        if (accepted.get('accepted') != 6000 or accepted.get('seed') != 123
                or not accepted.get('all_guidance_counts_match') or not accepted.get('all_fvs_forests')
                or not accepted.get('statistics_recomputed')):
            raise ValueError('Incomplete single-seed a/b acceptance')
    files = sorted(args.s3_family_results.glob('dismantle_perform_*.json'))
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    fig, summary = draw_paper_parameter_sensitivity(args.s3_family_results, family_breakdown=True,
                                                   family_name=args.s3_family_name, ab_only=args.s3_family_ab_only)
    if args.s3_family_ab_only:
        summary.update(seed=123, data_scope='New 400-graph a/b run, seed123; not historical or five-seed results',
                       a_axis_zoom='data range plus 15 percent padding; exact-flat curves use a 0.002 percentage-point window',
                       b_axis_minimum_span_percentage_points=.02, uncertainty_band=None,
                       axis_limits=[list(ax.get_ylim()) for ax in fig.axes])
    for ax, panel in zip(fig.axes, summary['panels'].values()):
        for line, family in zip(ax.lines, panel['families'].values()):
            if not np.allclose(line.get_ydata(), family['mean_percent'], rtol=0, atol=1e-12):
                raise ValueError('Family curve differs from historical data')
    args.s3_family_output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'pdf.fonttype': 42}):
        fig.savefig(str(args.s3_family_output) + '.pdf')
        fig.savefig(str(args.s3_family_output) + '.png', dpi=300)
    plt.close(fig)
    assert before == {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    summary['input_sha256'] = before
    Path(str(args.s3_family_output) + '.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Exported per-family means; 100 graphs per family; rendering only')


def export_original_s3_without_band(source_pdf, output_prefix):
    """Hide only the four original translucent bands; preserve all vector paths.

    Restricted to the inspected original Matplotlib PDF. Two same-width alpha
    edits preserve every content stream, text object, page box and xref offset.
    Refuse other PDF layouts rather than guessing or reconstructing mean curves.
    """
    import re
    import zlib
    import hashlib
    from pathlib import Path
    source_pdf, output_prefix = Path(source_pdf), Path(output_prefix)
    raw = source_pdf.read_bytes()
    state = b'/A2 << /Type /ExtGState /CA 0.2 /ca 0.2 >>'
    replacement = b'/A2 << /Type /ExtGState /CA 0.0 /ca 0.0 >>'
    if not raw.startswith(b'%PDF-') or raw.count(state) != 1:
        raise ValueError('Not the inspected original S3 PDF transparency layout')
    streams = []
    for match in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', raw, re.S):
        try:
            decoded = zlib.decompress(match.group(1))
            if b' gs' in decoded:
                streams.append(decoded)
        except zlib.error:
            pass
    if len(streams) != 1 or streams[0].count(b'/A2 gs') != 4 or streams[0].count(b'/A3 gs') != 12:
        raise ValueError('Unexpected original band or opaque-path count')
    changed = raw.replace(state, replacement)
    if len(changed) != len(raw) or sum(a != b for a, b in zip(raw, changed)) != 2:
        raise ValueError('Unexpected PDF edit beyond the two alpha digits')
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix('.pdf').write_bytes(changed)
    audit = {'original_pdf_sha256': hashlib.sha256(raw).hexdigest(),
             'output_pdf_sha256': hashlib.sha256(changed).hexdigest(),
             'changed_byte_count': 2, 'transparent_bands_hidden': 4,
             'all_content_streams_unchanged': True, 'curves_axes_text_and_layout_preserved': True,
             'rerun_or_resampling': False, 'inference_route': 'not recorded in original figure'}
    output_prefix.with_suffix('.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


def _s3_original_no_band_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-original-pdf', required=True)
    parser.add_argument('--s3-no-band-output', required=True)
    args = parser.parse_args()
    print(json.dumps(export_original_s3_without_band(args.s3_original_pdf, args.s3_no_band_output)))


def _s3_plot_cli():
    """Explicit read-only rendering of accepted results; never starts inference."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-plot-statistics', type=Path, required=True)
    parser.add_argument('--s3-acceptance', type=Path, required=True)
    parser.add_argument('--s3-plot-output', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.s3_acceptance.read_text(encoding='utf-8'))
    if (audit.get('accepted') != 6200 or not audit.get('all_fvs_forests')
            or not audit.get('statistics_independently_recomputed') or not audit.get('all_related_processes_exited')):
        raise ValueError('Final independent acceptance is required before export')
    fig, summary = draw_paper_parameter_sensitivity(
        args.s3_plot_statistics.parent, statistics_file=args.s3_plot_statistics)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extent = fig.bbox
    visible_texts = list(fig.texts)
    for ax in fig.axes:
        visible_texts.extend(list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title,
                                            ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()])
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(limits)
            for tick in axis.get_major_ticks():
                if lo - 1e-10 <= tick.get_loc() <= hi + 1e-10:
                    visible_texts.extend([tick.label1, tick.label2])
    for text in visible_texts:
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < extent.x0 - 2 or box.y0 < extent.y0 - 2 or box.x1 > extent.x1 + 2 or box.y1 > extent.y1 + 2:
                raise ValueError('Figure text falls outside the canvas: ' + text.get_text())
    for ax, panel in zip(fig.axes, summary['panels'].values()):
        if not np.allclose(ax.lines[0].get_ydata(), panel['mean'], rtol=0, atol=1e-12):
            raise ValueError('Rendered curve differs from accepted means')
    args.s3_plot_output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'pdf.fonttype': 42}):
        fig.savefig(str(args.s3_plot_output) + '.pdf')
        fig.savefig(str(args.s3_plot_output) + '.png', dpi=300)
    Path(str(args.s3_plot_output) + '.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plt.close(fig)
    print('Exported accepted five-seed Figure S3: PDF, PNG and numeric source')

if __name__=="__main__":
    import sys
    if "--s3-mean-results" in sys.argv:
        _s3_historical_mean_cli()
        raise SystemExit(0)
    if "--s3-original-pdf" in sys.argv:
        _s3_original_no_band_cli()
        raise SystemExit(0)
    if "--s3-ab-run-dir" in sys.argv:
        _s3_ab_trace_cli()
        raise SystemExit(0)
    if "--s3-family-results" in sys.argv:
        _s3_family_cli()
        raise SystemExit(0)
    if "--s3-plot-statistics" in sys.argv:
        _s3_plot_cli()
        raise SystemExit(0)
    if "--s3-seed-plan" in sys.argv:
        raise SystemExit(_s3_seed_cli())
    """
   
    """
    nep = False  # 节点爆炸渗流精细化节点拆解顺序
    experiment_name="DiffGDM"


    # analysis_param ="rewrite_ratio"  #
    # params_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    # dismantle_model_params_experiment(experiment_name,analysis_param,params_list)

    # plt.rc('font', family='Arial', size=12)
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # model_params_sensitivity_list=["Rewrite_Ratio","Inference_Diffusion_Steps","GNN_Dims","GNN_Layers"]
    # dataset_type = "synth_large"
    # dataset_path = "dataset/dataset_{}".format(dataset_type)
    # save_path = os.path.join("result/{}".format(experiment_name), "dismantle_analysis_model_params")
    # label=["a","b","c","d"]
    # for n,model_param in enumerate(model_params_sensitivity_list):
    #     plt_model_params_sensitivity(ax=axes[n//2][n%2],model_param=model_param,save_path=save_path)
    #     axes[n//2][n%2].text(-0.05, 1.1, label[n], transform=axes[n//2][n%2].transAxes,
    #             fontsize=16, fontweight='bold', va='top', ha='left')
    # plt.tight_layout()
    # plt.savefig(save_path + '/dismantle_model_sensitivity.png')
    #
    # plt.show()
    plt.rc('font', family='Arial', size=12)
    fig, ax = plt.subplots()
    model_param = "Rewrite_Ratio"
    dataset_type = "synth_large"
    dataset_path = "dataset/dataset_{}".format(dataset_type)
    save_path = os.path.join("result/{}".format(experiment_name), "dismantle_analysis_model_params")

    plt_model_params_sensitivity(ax=ax, model_param=model_param, save_path=save_path)
   # plt.savefig(save_path + '/Rewrite_steps_convergence.svg')
    plt.tight_layout()
    plt.show()





















