import io
import json
import unittest
from unittest.mock import patch

from transport import read_completion


def events(*items):
    return io.BytesIO(b''.join(b'data: ' + (x.encode() if isinstance(x, str) else json.dumps(x).encode()) + b'\n\n'
                              for x in items))


class StreamingChecks(unittest.TestCase):
    def test_slow_incomplete_event_obeys_total_deadline_without_a_newline(self):
        class Trickle:
            headers={}
            def read1(self, size):return b'data: {"choices":'
        with patch('transport.time.monotonic',side_effect=[0,0,1,301]):
            with self.assertRaises(TimeoutError):
                read_completion(Trickle(),streaming=True,deadline_seconds=300)

    def test_answer_chunks_usage_and_reasoning_are_separated(self):
        response = events(
            {'choices':[{'delta':{'reasoning_content':'reasoning is not answer text'}}]},
            {'choices':[{'delta':{'content':'add('}}]},
            {'choices':[{'delta':{'content':'1, 2)'}, 'finish_reason':'stop'}]},
            {'choices':[], 'usage':{'prompt_tokens':10,'completion_tokens':5}}, '[DONE]')
        result=read_completion(response, streaming=True)
        self.assertEqual(result['choices'][0]['message']['content'], 'add(1, 2)')
        self.assertEqual(result['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(result['usage']['completion_tokens'],5)
        self.assertGreater(result['stream_diagnostics']['reasoning_characters'],0)
        self.assertNotIn('reasoning is not answer text',json.dumps(result))

    def test_disconnected_partial_stream_is_not_an_answer(self):
        with self.assertRaises(ConnectionError):
            read_completion(events({'choices':[{'delta':{'content':'add(1,'}}]}), streaming=True)

    def test_reasoning_only_stream_does_not_become_program(self):
        result=read_completion(events({'choices':[{'delta':{'reasoning_content':'hidden'},'finish_reason':'length'}]},
                                      '[DONE]'), streaming=True)
        self.assertEqual(result['choices'][0]['message']['content'],'')
        self.assertEqual(result['choices'][0]['finish_reason'],'length')

    def test_provider_json_fallback_and_stream_error(self):
        body={'choices':[{'message':{'content':'add(1, 2)'},'finish_reason':'stop'}]}
        response=io.BytesIO(json.dumps(body).encode())
        response.headers={'Content-Type':'application/json; charset=utf-8'}
        self.assertEqual(read_completion(response,streaming=True),body)
        with self.assertRaises(RuntimeError):
            read_completion(events({'error':{'message':'upstream unavailable'}}),streaming=True)


if __name__ == '__main__':
    unittest.main()
