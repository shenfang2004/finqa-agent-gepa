"""Integration checks: failed calculation syntax must stop the live launcher."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import uuid
import unittest
from unittest.mock import patch

import dspy
import models
import run
from core import dump, synthetic_sets


class SmokeChecks(unittest.TestCase):
    def check_smoke(self, output, expected_status, expected_code):
        project = Path(__file__).resolve().parent
        scratch = project/'work'/('smoke-check-' + uuid.uuid4().hex)
        data = scratch/'data'
        for split, records in synthetic_sets().items():
            dump(data/f'{split}.json', records[:1])
        config = {'model': 'mock-only', 'base_url': 'https://example.test/v1',
                  'api_key': 'not-a-real-key', 'max_requests': 2, 'max_output_tokens': 100,
                  'input_cny_per_million': 0., 'output_cny_per_million': 0.,
                  'max_cost_cny': 1., 'max_metric_calls': 10}
        args = argparse.Namespace(config='unused', model=None, input_price=None,
                                  output_price=None, data_dir=str(data),
                                  train=1, dev=1, test=1, feedback='detailed')
        with contextlib.nullcontext(str(scratch)) as directory:
            with patch.object(run, 'ROOT', Path(directory)), \
                 patch.object(models, 'read_config', return_value=config), \
                 patch.object(models, 'CompatibleLM', side_effect=lambda cfg, ledger, role: models.ScriptedLM(ledger, role)), \
                 patch.object(dspy, 'Predict', return_value=lambda **kwargs: dspy.Prediction(program=output)), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = run.run_experiment('smoke', args)
            summaries = list(Path(directory).glob('runs/smoke-*/summary.json'))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text(encoding='utf-8'))
            self.assertEqual(code, expected_code)
            self.assertEqual(summary['status'], expected_status)
            self.assertEqual(summary['model'], 'mock-only')

    def test_invalid_program_stops_before_optimization(self):
        self.check_smoke('add(#0, 1)', 'invalid_program', 1)

    def test_valid_program_passes_format_check_even_if_answer_is_wrong(self):
        self.check_smoke('add(1, 1)', 'complete', 0)


if __name__ == '__main__':
    unittest.main()
