"""Local contracts for paper input integrity, numeric content and relocation."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import paper_figures as paper


def load_toy():
    path = paper.ROOT / 'model_test/toy_model_analysis.py'
    spec = importlib.util.spec_from_file_location('toy_model_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ToyEntryTests(unittest.TestCase):
    def test_original_graph_and_seed(self):
        toy = load_toy()
        graph = toy.build_toy_graph()
        self.assertEqual((len(graph), graph.number_of_edges()), (80, 120))
        self.assertTrue(toy.nx.is_connected(graph))
        self.assertEqual(toy.graph_record(graph)['edge_sha256'],
            '856131fb28607944162dc4c64e9aa929550c07badfc598a848144e85b50b99e7')
        self.assertEqual(toy.TARGET_PERCENT, 10)
        self.assertEqual(toy.infer_parameters()['parallel_sampling'], 4)

    def test_original_solve_draw_order(self):
        toy = load_toy()
        calls = []
        utility = types.ModuleType('utils.utils')
        utility.seed_torch = lambda seed: calls.append(('seed', seed))

        def solve(method, graph, target, **kwargs):
            calls.append(('solve', method))
            return list(graph), list(graph)

        with patch.dict(sys.modules, {'utils.utils': utility}), \
             patch.object(toy, 'make_figure', return_value=(object(), None)), \
             patch.object(toy, 'decycle_dismantle_method', side_effect=solve), \
             patch.object(toy, 'draw_method', side_effect=lambda *a: calls.append(('draw', a[3]))):
            _, record = toy.run_experiment()
        self.assertEqual(calls, [
            ('seed', 19), ('solve', 'DiffGDM_base'), ('draw', 'CoreDiffGDM_base'),
            ('solve', 'DiffGDM'), ('draw', 'CoreDiffGDM'),
            ('solve', 'BPD'), ('draw', 'BPD')])
        self.assertEqual(len(record['results']), 3)

    def test_both_original_algorithm_calls_are_retained(self):
        toy = load_toy()
        calls = []
        backend = types.ModuleType('method.method')
        backend.DecycleList = lambda *a, **k: (calls.append(('FVS', a, k)) or [1], 0)
        backend.NodeBreakList = lambda *a, **k: (calls.append(('CFN', a, k)) or [2], 0)
        with patch.dict(sys.modules, {'method.method': backend}):
            result = toy.decycle_dismantle_method('DiffGDM', toy.build_toy_graph(), 10)
        self.assertEqual(result, ([1], [2]))
        self.assertEqual([row[0] for row in calls], ['FVS', 'CFN'])
        self.assertEqual(calls[1][2]['target'], 10)


class PaperFiguresTests(unittest.TestCase):
    def test_manifest_files_and_relative_paths(self):
        document=paper.read(paper.DATA/'manifest.json')
        for path,expected in document['sha256'].items():
            self.assertFalse(Path(path).is_absolute())
            self.assertEqual(paper.sha(paper.ROOT/path),expected,path)

    def test_current_single_run_profiles(self):
        for n in ('2','3','6'):
            payload=paper.read(paper.DATA/('figure'+n+'.json'))
            self.assertEqual(payload['profile'],'original-plus-controls')
        self.assertEqual(len(paper.read(paper.DATA/'figure3.json')['graphs']),27)

    def test_scale_ratios_and_local_raw_records(self):
        q=paper.read(paper.DATA/'figure8.json');s=q['statistics']
        for i,m in enumerate(q['methods']):
            for j,scale in enumerate(q['scales']):
                denom=q['random_decycle_mean_cfn_fraction'][j]
                ratio=s['normalized_cfn'][i][j]
                self.assertAlmostEqual(ratio,s['CFN_mean_percent'][m][j]/(100*denom),places=12)
        self.assertEqual(len(list((paper.DATA/'records/scalability_random').glob('*.json'))),50)
        self.assertEqual(len(paper.read(paper.DATA/'records/scalability_ensemble.json')['records']),50)

    def test_check_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'not-created'
            proc=subprocess.run([sys.executable,str(Path(paper.__file__)),
                '--paper-figures','S2','S3','--check','--output-dir',str(out)],capture_output=True)
            self.assertEqual(proc.returncode,0,proc.stderr.decode())
            self.assertFalse(out.exists())

    def test_s1_is_original_400_epoch_trace(self):
        q=paper.read(paper.DATA/'training/figure_s1_record_scores.json')
        self.assertEqual(q['epoch'],list(range(400)))
        self.assertEqual(len(q['train_scores']),400)
        self.assertEqual(len(q['val_scores']),400)

    def test_source_mutation_and_unsafe_paths_are_detected(self):
        # Validate the guard against a private, tiny fixture rather than modify
        # any actual data or its manifest.
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);p=folder/'sample.json';p.write_text('{}')
            manifest={'sha256':{'sample.json':paper.sha(p)},
                      'figures':{'S2':{'status':'READY'}}}
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with patch.object(paper,'DATA',folder),patch.object(paper,'ROOT',folder):
                self.assertEqual(paper.main(['--paper-figures','S2','--check']),0)
                p.write_text('{"changed":true}')
                with self.assertRaises(ValueError):paper.main(['--paper-figures','S2','--check'])
                manifest['sha256']={'../sample.json':'a'*64}
                (folder/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):paper.main(['--paper-figures','S2','--check'])


if __name__=='__main__':
    unittest.main()
