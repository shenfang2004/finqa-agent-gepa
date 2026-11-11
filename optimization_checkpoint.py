"""Replay successful optimization calls; never accept transport failures as zero scores."""
from collections import defaultdict, deque
from contextlib import contextmanager
import json
from unittest.mock import patch

import dspy

from checkpoints import digest, generation_settings


class ReplayLM(dspy.BaseLM):
    def __init__(self, backend, ledger, config, role, source=None):
        super().__init__(config['model'], cache=False, max_tokens=config['max_output_tokens'])
        self.backend, self.ledger, self.config, self.role = backend, ledger, config, role
        self.saved = defaultdict(deque)
        if source and (source/'calls.jsonl').exists():
            for line in (source/'calls.jsonl').read_text(encoding='utf-8').splitlines():
                call = json.loads(line)
                if (call.get('role') == role and call.get('stage') == 'optimization_train_validation'
                        and call.get('response') and not call.get('error') and call.get('finish_reason') != 'length'):
                    self.saved[self.key(call.get('messages', []))].append(call['response'])

    def key(self, messages):
        return digest({'generation':generation_settings(self.config), 'role':self.role, 'messages':messages})

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        messages = messages or [{'role':'user','content':str(prompt)}]
        cached = self.saved[self.key(messages)]
        if self.ledger.stage == 'optimization_train_validation' and cached:
            response = cached.popleft()
            self.ledger.write({'mode':'REPLAYED_RESPONSE', 'role':self.role, 'stage':self.ledger.stage,
                               'messages':messages, 'response':response, 'finish_reason':'stop'})
            self.history.append({'messages':messages,'outputs':[response]})
            return [response]
        try:
            result = self.backend(prompt=prompt, messages=messages, **kwargs)
        except Exception:
            self.ledger.failed_predictions = getattr(self.ledger,'failed_predictions',0) + 1
            raise
        self.history.append({'messages':messages,'outputs':result})
        return result


@contextmanager
def guard_optimization_scores(ledger):
    """DSPy's adapter otherwise maps caught request failures to its failure_score."""
    import dspy.teleprompt.gepa.gepa_utils as utils
    original = utils.DspyAdapter

    class ReliableAdapter(original):
        def evaluate(self, *args, **kwargs):
            failures = getattr(ledger,'failed_predictions',0)
            result = super().evaluate(*args, **kwargs)
            if getattr(ledger,'failed_predictions',0) != failures:
                raise RuntimeError('Optimization evaluation incomplete due to request failures; candidate scores are not accepted. Resume to replay successful calls and retry missing requests.')
            return result

    with patch.object(utils,'DspyAdapter',ReliableAdapter):
        yield
