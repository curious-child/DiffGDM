"""Portable, result-only rendering for the current paper figure numbers."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'experiments_figure/data'
ORDER = [str(i) for i in range(1, 9)] + ['S1', 'S2', 'S3', 'S4']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def plotting_function(relative, name, namespace):
    """Load an existing pure plotting function without importing experiment code.

    Its same-file helper functions and literal constants are retained. Module
    imports and experiment main blocks are not executed.
    """
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            nodes.append(node)
    space = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(relative), 'exec'), space)
    return space[name]


def namespace():
    import copy
    import math
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import networkx as nx
    from matplotlib.colors import LinearSegmentedColormap
    return dict(copy=copy, math=math, random=random, json=json, os=os, Path=Path,
                mpl=matplotlib, plt=plt, np=np, pd=pd, sns=sns, nx=nx,
                LinearSegmentedColormap=LinearSegmentedColormap)


def draw(number, entry, ns):
    payload = read(DATA / entry['input']) if 'input' in entry else None
    plotting = 'model_test/analysis_visi/LCC_size_removed_nodes.py'
    if number in ('2', '3', '6'):
        function = {'2': 'draw_verified_synthetic_comparison',
                    '3': 'draw_verified_route_comparison',
                    '6': 'draw_verified_guidance_comparison'}[number]
        return plotting_function(plotting, function, ns)(payload)
    if number == '4':
        func = plotting_function(plotting, 'draw_dismantle_statistics_real', ns)
        local = dict(ns, draw_dismantle_statistics_real=func)
        # The original function also requests its font helper by module name.
        sys.path.insert(0, str(ROOT / 'model_test'))
        return plotting_function('model_test/decycling_dismantling_analysis.py',
                                 'plt_graph_auc_analysis', local)(result_dir=DATA / 'auc', show=False)
    if number in ('7', 'S4'):
        figures = plotting_function(plotting, 'draw_guidance_benefit_analysis', ns)(payload)
        chosen = 0 if number == '7' else 1
        ns['plt'].close(figures[1 - chosen])
        return figures[chosen]
    if number == '8':
        return plotting_function(plotting, 'draw_relative_resource_comparison', ns)(payload)
    if number in ('S2', 'S3'):
        return plotting_function('model_test/dismantle_model_params_analysis.py',
            'draw_paper_parameter_sensitivity', ns)(DATA / 'parameters',
            rewrite_only=number == 'S2', show_band=False, full_graph_mean=True)[0]
    if number == 'S1':
        fig = ns['plt'].figure()
        plotting_function('model_train/utils/data_visualization.py',
                          'model_evaluation_metrics_curves', ns)(fig, payload)
        return fig
    raise ValueError('No renderer for figure ' + number)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--paper-all', action='store_true')
    select.add_argument('--paper-figures', nargs='+', choices=ORDER)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'experiments_figure/outputs/paper')
    args = parser.parse_args(argv)
    manifest = read(DATA / 'manifest.json')
    selected = args.paper_figures or ORDER
    # The check path uses only the standard library and writes nothing.
    for relative, expected in manifest['sha256'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path) != expected:
            raise ValueError('Source checksum mismatch: ' + relative)
    unavailable = []
    for number in selected:
        item = manifest['figures'][number]
        print('Figure ' + number + ': ' + item['status'] +
              ((' - ' + item['reason']) if item.get('reason') else ''))
        if item['status'] == 'SOURCE_NEEDED':
            unavailable.append(number)
    if args.check:
        return 3 if unavailable else 0
    out = args.output_dir.resolve()
    if out.is_relative_to(DATA.resolve()) or out == ROOT.resolve():
        raise ValueError('Choose an output directory separate from source data')
    for number in selected:
        if number in unavailable:
            continue
        stem = 'Figure' + (number if number.startswith('S') else number.zfill(2))
        if not args.overwrite and any((out / (stem + ext)).exists() for ext in ('.pdf', '.png')):
            raise FileExistsError('Output exists; use --overwrite: ' + stem)
    ns = namespace()
    ns['mpl'].rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none',
                              'pdf.compression': 6, 'text.usetex': False})
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for number in selected:
        if number in unavailable:
            continue
        stem = 'Figure' + (number if number.startswith('S') else number.zfill(2))
        entry = manifest['figures'][number]
        if entry['status'] == 'VECTOR_ASSET':
            shutil.copyfile(DATA / entry['asset'], out / (stem + '.pdf'))
            try:
                import pypdfium2 as pdfium
            except ImportError:
                import subprocess
                if not shutil.which('pdftoppm'):
                    raise RuntimeError('Install pypdfium2 to export vector-asset PNG previews')
                subprocess.run(['pdftoppm', '-singlefile', '-r', '300', '-png',
                    '-scale-to', '2400',
                    str(DATA / entry['asset']), str(out / stem)], check=True)
            else:
                with pdfium.PdfDocument(DATA / entry['asset']) as doc:
                    page = doc[0]
                    scale = min(300 / 72, 2400 / max(page.get_size()))
                    page.render(scale=scale).to_pil().save(out / (stem + '.png'))
        else:
            style = {'font.family': 'DejaVu Sans' if number == 'S1' else 'Arial',
                     'font.size': 8 if number in ('7', 'S4') else 10,
                     'pdf.fonttype': 42, 'text.usetex': False}
            with ns['mpl'].rc_context(style):
                fig = draw(number, entry, ns)
                bbox = None if getattr(fig, '_exact_export_size', False) else 'tight'
                fig.savefig(out / (stem + '.pdf'), bbox_inches=bbox,
                            metadata={'CreationDate': None, 'ModDate': None})
                fig.savefig(out / (stem + '.png'), dpi=300, bbox_inches=bbox)
                ns['plt'].close(fig)
        results[number] = {ext: sha(out / (stem + '.' + ext)) for ext in ('pdf', 'png')}
    # Keep this run record in the output directory, never in the source tree.
    (out / 'run_manifest.json').write_text(json.dumps({'figures': results,
        'unavailable': unavailable, 'data_manifest_sha256': sha(DATA / 'manifest.json')}, indent=2), encoding='utf-8')
    for relative, expected in manifest['sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Source changed during rendering: ' + relative)
    return 3 if unavailable else 0


if __name__ == '__main__':
    raise SystemExit(main())
