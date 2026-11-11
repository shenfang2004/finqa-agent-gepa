import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

from checkpoints import EvaluationCheckpoint, stage_statistics, paired_statistics
from core import dump, load


class CapReached(RuntimeError):
    pass


class CheckpointChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent/'work'/('checkpoint-check-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.config = {'base_url':'https://example.test/v1','model':'test','max_output_tokens':100,'enable_thinking':False}
        self.records = [{'id':str(i),'question':f'question {i}','table':[], 'answer':3.,'gold_program':'add(1, 2)'} for i in range(3)]

    def evaluate(self, checkpoint, program):
        with contextlib.redirect_stdout(io.StringIO()):
            return checkpoint.evaluate(program,self.records,'baseline_test','same prompt',CapReached)

    def test_timeout_is_isolated_and_excluded_from_denominator(self):
        calls=[]
        def program(**values):
            calls.append(values['question'])
            if values['question']=='question 1':
                raise TimeoutError('offline simulated timeout')
            return SimpleNamespace(program='add(1, 2)')
        source=self.root/'first'
        rows,stopped=self.evaluate(EvaluationCheckpoint(source,self.config),program)
        self.assertEqual(len(calls),3)
        self.assertFalse(stopped)
        self.assertIsNone(rows[1]['score'])
        self.assertEqual(rows[1]['status'],'incomplete')
        stats=stage_statistics(rows,3)
        self.assertEqual(stats['completed'],2)
        self.assertEqual(stats['incomplete'],1)
        self.assertEqual(stats['accuracy_completed_only'],1.)
        self.assertEqual(load(source/'baseline_test.json'),rows)

    def test_resume_reuses_completed_rows_and_only_calls_missing_question(self):
        source=self.root/'source'
        def first(**values):
            if values['question']=='question 1':raise TimeoutError()
            return SimpleNamespace(program='add(1, 2)')
        self.evaluate(EvaluationCheckpoint(source,self.config),first)
        calls=[]
        def retry(**values):
            calls.append(values['question']);return SimpleNamespace(program='add(1, 2)')
        rows,stopped=self.evaluate(EvaluationCheckpoint(self.root/'resumed',self.config,source),retry)
        self.assertEqual(calls,['question 1'])
        self.assertEqual(stage_statistics(rows,3)['reused'],2)
        self.assertEqual(stage_statistics(rows,3)['completed'],3)
        self.assertFalse(stopped)

    def test_changed_generation_settings_or_input_are_not_reused(self):
        source=self.root/'source'
        self.evaluate(EvaluationCheckpoint(source,self.config),lambda **_:SimpleNamespace(program='add(1, 2)'))
        changed={**self.config,'enable_thinking':True}
        with self.assertRaises(ValueError):
            self.evaluate(EvaluationCheckpoint(self.root/'changed',changed,source),lambda **_:self.fail('must not call'))
        self.records[0]['question']='changed content'
        with self.assertRaises(ValueError):
            self.evaluate(EvaluationCheckpoint(self.root/'changed-input',self.config,source),lambda **_:self.fail('must not call'))

    def test_budget_stop_preserves_progress_and_marks_remaining_unattempted(self):
        calls=[]
        def program(**values):
            calls.append(values['question'])
            if len(calls)>1:raise CapReached('cap reached')
            return SimpleNamespace(program='add(1, 2)')
        rows,stopped=self.evaluate(EvaluationCheckpoint(self.root/'budget',self.config),program)
        self.assertTrue(stopped)
        self.assertEqual(len(calls),2)
        self.assertEqual([r['status'] for r in rows],['completed','not_attempted','not_attempted'])
        self.assertEqual(stage_statistics(rows,3)['completed'],1)

    def test_invalid_model_program_is_wrong_but_network_failure_is_unscored(self):
        def program(**values):
            if values['question']=='question 1':raise TimeoutError()
            return SimpleNamespace(program='add(#99, 1)')
        rows,_=self.evaluate(EvaluationCheckpoint(self.root/'invalid',self.config),program)
        self.assertEqual(rows[0]['status'],'completed')
        self.assertEqual(rows[0]['score'],0.)
        self.assertIsNone(rows[1]['score'])
        pairs=paired_statistics(rows,[{**rows[0],'score':1.},rows[1]])
        self.assertEqual(pairs['completed_pairs'],1)
        self.assertEqual(pairs['baseline_accuracy'],0.)
        self.assertEqual(pairs['optimized_accuracy'],1.)


if __name__=='__main__':unittest.main()
