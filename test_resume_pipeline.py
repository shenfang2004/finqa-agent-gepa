import argparse
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import dspy
import models
import run
from core import dump, load
from optimization_checkpoint import guard_optimization_scores


class PipelineResumeChecks(unittest.TestCase):
    def test_resume_reuses_both_stages_and_frozen_prompt_without_restarting_gepa(self):
        root=Path(__file__).resolve().parent/'work'/('pipeline-resume-check-'+uuid.uuid4().hex)
        data=root/'data'
        records=[{'id':str(i),'question':f'q{i}','table':[], 'answer':3.,'gold_program':'add(1, 2)'} for i in range(3)]
        for split in ('train','dev','test'):dump(data/f'{split}.json',records if split=='test' else records[:1])
        config={'model':'mock-only','base_url':'https://example.test/v1','api_key':'dummy',
                'max_requests':20,'max_output_tokens':100,'max_cost_cny':1.,'max_metric_calls':10,
                'input_cny_per_million':0.,'output_cny_per_million':0.}
        args=argparse.Namespace(config='unused',model=None,input_price=None,output_price=None,
                                data_dir=str(data),train=1,dev=1,test=3,feedback='detailed',resume=None,retry_failed=False)
        calls=[]
        fail_first=True

        class Program:
            def __init__(self, signature):self.signature=signature
            def __call__(self, **values):
                optimized=self.signature.instructions.endswith(' OPTIMIZED')
                calls.append(('optimized' if optimized else 'baseline',values['question']))
                if fail_first and ((not optimized and values['question']=='q1') or (optimized and values['question']=='q2')):
                    raise TimeoutError('simulated failure')
                return SimpleNamespace(program='add(1, 2)')

        class Optimizer:
            def __init__(self, **kwargs):pass
            def compile(self, student, **kwargs):
                result=Program(student.signature.with_instructions(student.signature.instructions+' OPTIMIZED'))
                result.detailed_results=SimpleNamespace(candidates=[result],val_aggregate_scores=[1.])
                return result

        with patch.object(run,'ROOT',root), patch.object(models,'read_config',return_value=config), \
             patch.object(models,'CompatibleLM',side_effect=lambda cfg,ledger,role:models.ScriptedLM(ledger,role)), \
             patch.object(dspy,'Predict',side_effect=Program), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(dspy,'GEPA',Optimizer):
                self.assertEqual(run.run_experiment('live',args),2)
            source=next((root/'runs').glob('live-*'))
            first=load(source/'summary.json')
            self.assertEqual(first['status'],'partial')
            self.assertEqual(first['coverage']['baseline']['completed'],2)
            self.assertEqual(first['coverage']['optimized']['completed'],2)
            self.assertTrue((source/'optimized-instructions.json').exists())
            fail_first=False
            calls.clear()
            args.resume=str(source)
            args.retry_failed=True
            with patch.object(dspy,'GEPA',side_effect=AssertionError('must reuse frozen prompt')):
                self.assertEqual(run.run_experiment('live',args),0)
        resumed=[p for p in (root/'runs').glob('live-*') if p!=source][0]
        second=load(resumed/'summary.json')
        self.assertEqual(calls,[('baseline','q1'),('optimized','q2')])
        self.assertEqual(second['coverage']['baseline']['reused'],2)
        self.assertEqual(second['coverage']['optimized']['reused'],2)
        self.assertTrue(second['pipeline_complete'])
        self.assertEqual(second['paired_comparison']['completed_pairs'],3)
        self.assertEqual(load(resumed/'incomplete_questions.json'),[])

    def test_optimization_guard_rejects_request_failures_instead_of_accepting_zero_scores(self):
        import dspy.teleprompt.gepa.gepa_utils as utils
        ledger=SimpleNamespace(failed_predictions=0)
        def incomplete(*args, **kwargs):
            ledger.failed_predictions+=1
            return SimpleNamespace(scores=[0.])
        with patch.object(utils.DspyAdapter,'evaluate',incomplete), guard_optimization_scores(ledger):
            adapter=object.__new__(utils.DspyAdapter)
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                adapter.evaluate([], {})


if __name__=='__main__':unittest.main()
