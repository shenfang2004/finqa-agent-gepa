import unittest
from pathlib import Path
from unittest.mock import patch
from models import Ledger, BudgetExceeded, normalize_base_url, CompatibleLM, model_override

class EndpointAndBudgetChecks(unittest.TestCase):
    def test_qwen_request_explicitly_disables_thinking_and_uses_streaming(self):
        import io, json, uuid
        directory = Path.cwd()/'work'/('stream-check-' + uuid.uuid4().hex)/'run'
        config={'model':'Qwen3.8-Max','base_url':'https://example.test/v1','api_key':'dummy',
                'max_requests':1,'max_output_tokens':1500,'max_cost_cny':1.,
                'input_cny_per_million':0.,'output_cny_per_million':0.,'timeout_seconds':180,
                'stream':True,'enable_thinking':False}
        ledger=Ledger(directory,config)
        chunk={'choices':[{'delta':{'content':'add(1, 2)'},'finish_reason':'stop'}]}
        response=io.BytesIO(('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode())
        with patch('urllib.request.urlopen',return_value=response) as transport:
            outputs=CompatibleLM(config,ledger,'task')(messages=[{'role':'user','content':'task'}])
        body=json.loads(transport.call_args.args[0].data)
        self.assertFalse(body['enable_thinking'])
        self.assertTrue(body['stream'])
        self.assertEqual(outputs,['add(1, 2)'])
        self.assertFalse(ledger.summary()['token_usage_complete'])
        self.assertIsNone(ledger.summary()['prompt_tokens'])

    def test_transient_failure_retries_once_and_counts_both_attempts(self):
        import io, json, uuid, urllib.error
        directory = Path.cwd()/'work'/('retry-check-' + uuid.uuid4().hex)/'run'
        config = {'model':'test-model', 'base_url':'https://example.test/v1',
                  'api_key':'TEST_SECRET_NOT_A_REAL_KEY', 'max_requests':2,
                  'max_output_tokens':100, 'input_cny_per_million':1.,
                  'output_cny_per_million':2., 'max_cost_cny':1.,
                  'timeout_seconds':180, 'max_retries':1}
        ledger = Ledger(directory, config)
        lm = CompatibleLM(config, ledger, 'task')
        payload = {'choices':[{'message':{'content':'add(1, 2)'},'finish_reason':'stop'}]}
        with patch('urllib.request.urlopen', side_effect=[urllib.error.URLError(TimeoutError()),
                    io.BytesIO(json.dumps(payload).encode())]) as transport, patch('models.time.sleep'):
            self.assertEqual(lm(messages=[{'role':'user','content':'task'}]), ['add(1, 2)'])
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(ledger.calls, 2)
        self.assertEqual(transport.call_args.kwargs['timeout'], 180)
        events=[json.loads(x) for x in (directory/'calls.jsonl').read_text().splitlines()]
        self.assertTrue(events[0]['will_retry'])
        self.assertEqual(events[0]['error_type'], 'URLError')
        self.assertNotIn(config['api_key'], (directory/'calls.jsonl').read_text())

    def test_retry_stops_at_request_cap_and_does_not_retry_auth_errors(self):
        import uuid, urllib.error
        for code in (None, 401):
            directory = Path.cwd()/'work'/('retry-cap-check-' + uuid.uuid4().hex)/'run'
            config = {'model':'test-model', 'base_url':'https://example.test/v1',
                      'api_key':'dummy', 'max_requests':1, 'max_output_tokens':10,
                      'input_cny_per_million':0., 'output_cny_per_million':0.,
                      'max_cost_cny':1., 'timeout_seconds':1, 'max_retries':1}
            ledger=Ledger(directory,config)
            error = TimeoutError() if code is None else urllib.error.HTTPError('https://example.test',code,'auth',None,None)
            with patch('urllib.request.urlopen',side_effect=error) as transport, patch('models.time.sleep'):
                with self.assertRaises(BudgetExceeded if code is None else RuntimeError):
                    CompatibleLM(config,ledger,'task')(messages=[{'role':'user','content':'task'}])
            self.assertEqual(transport.call_count,1)

    def test_unpriced_run_keeps_caps_and_reports_unknown_cost(self):
        import uuid
        config = {'model':'old', 'input_cny_per_million':0., 'output_cny_per_million':0.,
                  'max_requests':1, 'max_output_tokens':100}
        chosen = model_override(config, 'new', allow_unpriced=True)
        directory = Path.cwd()/'work'/('unpriced-check-' + uuid.uuid4().hex)/'run'
        ledger = Ledger(directory, chosen)
        self.assertEqual(ledger.reserve([{'role':'user','content':'example'}]), 1)
        self.assertIsNone(ledger.summary()['estimated_cny'])
        self.assertIsNone(ledger.summary()['reserved_cny'])
        self.assertFalse(ledger.pricing_known)
        with self.assertRaises(BudgetExceeded):
            ledger.reserve([{'role':'user','content':'example'}])
        self.assertEqual(config['model'], 'old')

    def test_model_selection_does_not_reuse_prices_or_mutate_config(self):
        original = {'model': 'old', 'input_cny_per_million': 0., 'output_cny_per_million': 0.}
        with self.assertRaises(ValueError):
            model_override(original, 'new')
        with self.assertRaises(ValueError):
            model_override(original, 'new', 0.)
        result = model_override(original, 'new', 1., 2.)
        self.assertEqual(result['model'], 'new')
        self.assertEqual(result['output_cny_per_million'], 2.)
        self.assertEqual(original['model'], 'old')
        for price in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                model_override(original, 'new', price, 0.)

    def test_endpoint_accepts_console_root_or_completion_url(self):
        self.assertEqual(normalize_base_url('https://llmapi.paratera.com'), 'https://llmapi.paratera.com/v1')
        self.assertEqual(normalize_base_url('https://llmapi.paratera.com/v1/chat/completions'), 'https://llmapi.paratera.com/v1')
        with self.assertRaises(ValueError):
            normalize_base_url('http://example.com')

    def test_request_cap_stops_before_transport(self):
        ledger = Ledger.__new__(Ledger)
        import threading
        ledger.lock = threading.Lock()
        ledger.calls = 1
        ledger.config = {'max_requests': 1}
        with self.assertRaises(BudgetExceeded):
            ledger.reserve([{'role': 'user', 'content': 'hello'}])

    def test_spend_cap_reserves_output_before_request(self):
        ledger = Ledger.__new__(Ledger)
        import threading
        ledger.lock = threading.Lock()
        ledger.calls, ledger.reserved_total = 0, 0
        ledger.config = {'max_requests': 20, 'max_output_tokens': 4096,
            'input_cny_per_million': 1000, 'output_cny_per_million': 1000, 'max_cost_cny': .01}
        with self.assertRaises(BudgetExceeded):
            ledger.reserve([{'role': 'user', 'content': 'hello'}])
        self.assertEqual(ledger.calls, 0)

    def test_live_transport_contract_and_key_redaction(self):
        import io, json, uuid
        directory = Path.cwd() / 'work' / ('transport-check-' + uuid.uuid4().hex) / 'run'
        config = {'model':'test-model', 'base_url':'https://example.test/v1',
                  'api_key':'TEST_SECRET_NOT_A_REAL_KEY', 'max_requests':2,
                  'max_output_tokens':100, 'input_cny_per_million':1.,
                  'output_cny_per_million':2., 'max_cost_cny':1., 'timeout_seconds':1}
        ledger = Ledger(directory, config)
        lm = CompatibleLM(config, ledger, 'task')
        body = {'choices':[{'message':{'content':'[[ ## program ## ]]\nadd(1, 2)'},
                            'finish_reason':'stop'}],
                'usage':{'prompt_tokens':10,'completion_tokens':5}}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(body).encode())) as transport:
            outputs = lm(messages=[{'role':'user','content':'only public task input'}])
        self.assertIn('add(1, 2)', outputs[0])
        request = transport.call_args.args[0]
        self.assertEqual(request.full_url, 'https://example.test/v1/chat/completions')
        self.assertEqual(json.loads(request.data)['model'], 'test-model')
        self.assertEqual(ledger.calls, 1)
        self.assertNotIn(config['api_key'], (directory/'calls.jsonl').read_text())
        self.assertAlmostEqual(ledger.cost, .00002)

if __name__ == '__main__':
    unittest.main()
