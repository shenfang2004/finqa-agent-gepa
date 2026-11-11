"""Two transparent backends: scripted teaching demo and real compatible API."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import dspy
from core import dump
from transport import read_completion

BASE_INSTRUCTION = '''Answer a financial question using only the supplied report.
Return a FinQA DSL program, without prose or markdown. Allowed operations:
add(a,b), subtract(a,b), multiply(a,b), divide(a,b), exp(a,b), greater(a,b),
table_sum(row,none), table_average(row,none), table_min(row,none), table_max(row,none).
Separate steps with commas; #0 refers to the result of step 0, #1 to step 1.
Use const_100 for 100 and const_m1 for -1 when needed. Percent literals such as 20%
mean 0.2. Use numbers as printed in the report and respect the question's units.
Do not use Python. Do not copy values from unrelated examples.
Output format examples (illustrative only; never reuse their numbers):
For revenue changing from 100 to 120, the relative change is:
subtract(120, 100), divide(#0, 100)
For the fraction 15 out of 60, use: divide(15, 60)
For the mean of 10, 20 and 30, use:
add(10, 20), add(#0, 30), divide(#1, 3)
Prefer these flat steps. #0 is the FIRST completed step, not a table cell;
never reference a current or future step. table_sum(revenue, none) aggregates
the entire row named revenue, not one year; to use a particular year, read
that numeric cell from the supplied table. Preserve signs, do not invent
missing values, and do not treat extraction artifacts as actual amounts.'''

class BudgetExceeded(RuntimeError):
    pass

class Ledger:
    def __init__(self, directory, config=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.lock = threading.Lock()
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.cost = 0.0
        self.missing_usage_responses = 0
        self.stage = 'initialization'
        self.reserved_total = 0.0
        self.budget_path = self.directory.parent / 'budget-reservations.json'

    @property
    def pricing_known(self):
        return all(isinstance(self.config.get(k), (int, float)) and math.isfinite(self.config[k])
                   and self.config[k] >= 0 for k in ('input_cny_per_million', 'output_cny_per_million'))

    def reserve(self, messages):
        with self.lock:
            if self.calls >= self.config['max_requests']:
                raise BudgetExceeded('API request cap reached; no new request sent.')
            if self.config.get('allow_unpriced'):
                # Explicit user opt-in: retain request/token caps, do not invent prices.
                account = json.loads(self.budget_path.read_text(encoding='utf-8')) if self.budget_path.exists() else {'reserved_cny': 0.0}
                account['unpriced_requests'] = account.get('unpriced_requests', 0) + 1
                dump(self.budget_path, account)
                self.calls += 1
                return self.calls
            # UTF-8 byte count is a deliberately conservative input-token proxy
            # for mainstream byte/subword tokenizers, not a provider billing quote.
            input_bound = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 512
            output_bound = self.config['max_output_tokens']
            reserve = (input_bound * self.config['input_cny_per_million'] +
                       output_bound * self.config['output_cny_per_million']) / 1_000_000
            if self.reserved_total + reserve > self.config['max_cost_cny']:
                raise BudgetExceeded('Conservative spend reservation cap reached; no new request sent.')
            # Persist reservations across runs. Stop conservatively even after
            # a timeout, since the provider may already have processed a request.
            account = json.loads(self.budget_path.read_text(encoding='utf-8')) if self.budget_path.exists() else {'reserved_cny': 0.0}
            if account['reserved_cny'] + reserve > 100:
                raise BudgetExceeded('Project-wide 100 CNY reservation ceiling reached.')
            account['reserved_cny'] += reserve
            dump(self.budget_path, account)
            self.reserved_total += reserve
            self.calls += 1
            return self.calls

    def write(self, event):
        with self.lock:
            with (self.directory / 'calls.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def summary(self):
        missing = self.missing_usage_responses
        return {'calls': self.calls, 'prompt_tokens': None if missing else self.prompt_tokens,
                'completion_tokens': None if missing else self.completion_tokens,
                'reported_prompt_tokens':self.prompt_tokens, 'reported_completion_tokens':self.completion_tokens,
                'missing_usage_responses':missing, 'token_usage_complete':not bool(missing),
                'estimated_cny': None if self.config.get('allow_unpriced') or missing else round(self.cost, 6),
                'reserved_cny': None if self.config.get('allow_unpriced') else round(self.reserved_total, 6),
                'billing_note': 'Prices not supplied; cost is unknown, not zero. Request and output-token caps remain active.'
                    if self.config.get('allow_unpriced') else
                    'Estimate only. Uses entered prices. Provider billing and tokenization are authoritative.'}

class ScriptedLM(dspy.BaseLM):
    """Not an LLM. Behavior intentionally changes when a scripted rule is present."""
    def __init__(self, ledger, role):
        super().__init__('scripted-teaching-backend', cache=False, temperature=0, max_tokens=1500)
        self.ledger, self.role = ledger, role

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        messages = messages or [{'role': 'user', 'content': str(prompt)}]
        if self.role == 'reflection':
            instruction = BASE_INSTRUCTION + '''
For relative revenue growth, subtract old revenue from new revenue, then divide
that difference by old revenue. For absolute increase, subtract old from new.
For a ratio, divide new by old. Return relative rates as decimal fractions.
'''
            response = json.dumps({'new_instruction': instruction})
        else:
            text = str(messages[-1]['content'])
            context_text = re.search(r'\[\[\s*## context ##\s*\]\]\s*(.*?)\s*\[\[\s*## question ##', text, re.S)
            if not context_text:
                raise ValueError('Scripted backend supports only the bundled ChatAdapter teaching task.')
            context = json.loads(context_text.group(1))
            old, new = context['table'][1][1], context['table'][2][1]
            question = text.split('## question ##', 1)[1].lower()
            improved = 'for relative revenue growth' in str(messages[0]['content']).lower()
            if improved and 'growth rate' in question:
                program = f'subtract({new}, {old}), divide(#0, {old})'
            elif improved and 'absolute increase' in question:
                program = f'subtract({new}, {old})'
            else:
                program = f'divide({new}, {old})'
            response = f'[[ ## program ## ]]\n{program}\n\n[[ ## completed ## ]]'
        self.ledger.calls += 1
        self.ledger.write({'mode': 'SCRIPTED_SIMULATION', 'role': self.role,
                           'stage': self.ledger.stage, 'messages': messages, 'response': response})
        self.history.append({'messages': messages, 'outputs': [response]})
        return [response]

def normalize_base_url(value):
    value = value.strip().rstrip('/')
    if value.endswith('/chat/completions'):
        value = value[:-len('/chat/completions')]
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError('base_url must be a complete HTTPS URL without query or fragment.')
    if not parsed.path or parsed.path == '/':
        value += '/v1'
    return value

def read_config(path, require_model=True, require_prices=True):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    config['base_url'] = normalize_base_url(config['base_url'])
    config['api_key'] = os.environ.get('FINQA_API_KEY') or config.get('api_key', '')
    if not config['api_key'] or '仅在本地' in config['api_key']:
        raise ValueError('Set FINQA_API_KEY or fill api_key in config.local.json locally. Never post the key in chat.')
    if require_model and (not config.get('model') or '请填写' in config['model']):
        raise ValueError('Model ID missing. Run `python run.py models` to query the provider, if supported.')
    if not 0 < config.get('max_cost_cny', 0) <= 100:
        raise ValueError('max_cost_cny must be between 0 and the authorized 100 CNY ceiling.')
    if require_prices:
        for field in ('input_cny_per_million', 'output_cny_per_million'):
            if not isinstance(config.get(field), (int, float)) or config[field] < 0:
                raise ValueError(f'Enter provider price (CNY per million tokens) in {field} before paid runs.')
    if not 1 <= config.get('max_requests', 0) <= 500:
        raise ValueError('max_requests must be 1..500 for this demo.')
    if not 1 <= config.get('max_output_tokens', 0) <= 4096:
        raise ValueError('max_output_tokens must be 1..4096 for this demo.')
    return config

def list_models(config):
    request = urllib.request.Request(config['base_url'] + '/models',
        headers={'Authorization': 'Bearer ' + config['api_key']})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return [x['id'] for x in json.load(response).get('data', [])]
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Model listing returned HTTP {exc.code}; this provider may not support GET /models.') from None


def model_override(config, model=None, input_price=None, output_price=None, allow_unpriced=False):
    """Choose a model for this run without modifying the user's saved config."""
    result = dict(config)
    if model is not None and not model.strip():
        raise ValueError('Model ID must not be empty.')
    if allow_unpriced:
        if model is not None:
            result['model'] = model
        result.update(allow_unpriced=True, input_cny_per_million=None, output_cny_per_million=None)
        return result
    changed = model is not None and model != config.get('model')
    supplied = input_price is not None or output_price is not None
    if (changed or supplied) and (input_price is None or output_price is None):
        raise ValueError('Changing model requires --input-price and --output-price in CNY per million tokens; use 0 only for a confirmed free model.')
    if model is not None:
        result['model'] = model
    for key, value in [('input_cny_per_million', input_price), ('output_cny_per_million', output_price)]:
        if value is not None:
            result[key] = value
        if not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError('Enter finite non-negative model prices in CNY per million tokens.')
    return result

class CompatibleLM(dspy.BaseLM):
    """Minimal HTTP backend for DSPy 3.4, with shared hard request limits.

    Uses the 3.4 legacy subclass interface deliberately pinned in requirements;
    migrate to a DSPy engine before upgrading to 3.5. Never stores API keys in logs.
    """
    def __init__(self, config, ledger, role):
        super().__init__(config['model'], cache=False, temperature=0, max_tokens=config['max_output_tokens'])
        self.config, self.ledger, self.role = config, ledger, role

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        messages = messages or [{'role': 'user', 'content': str(prompt)}]
        body = {'model': self.config['model'], 'messages': messages, 'stream': self.config.get('stream', False),
                'max_tokens': self.config['max_output_tokens'],
                'temperature': 0.7 if self.role == 'reflection' else 0.0}
        if 'enable_thinking' in self.config:
            body['enable_thinking'] = self.config['enable_thinking']
        request = urllib.request.Request(self.config['base_url'] + '/chat/completions',
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.config['api_key']})
        retries = self.config.get('max_retries', 0)
        if not isinstance(retries, int) or not 0 <= retries <= 2:
            raise ValueError('max_retries must be 0..2.')
        for attempt in range(retries + 1):
            # Each physical HTTP attempt consumes a request and its own reservation.
            request_id = self.ledger.reserve(messages)
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.config['timeout_seconds']) as response:
                    payload = read_completion(response, streaming=body['stream'],
                        deadline_seconds=self.config.get('stream_deadline_seconds', 300))
                break
            except urllib.error.HTTPError as exc:
                retry = attempt < retries and exc.code in (408, 429, 500, 502, 503, 504)
                # Bodies/headers may echo credentials; retain only the status code.
                self.ledger.write({'request': request_id, 'role': self.role, 'stage': self.ledger.stage,
                                   'attempt': attempt+1, 'will_retry': retry,
                                   'error': f'HTTP {exc.code}', 'elapsed': round(time.time()-started, 3)})
                if not retry:
                    raise RuntimeError(f'Provider HTTP {exc.code}; stopped after {attempt+1} attempt(s).') from None
            except (TimeoutError, OSError) as exc:
                retry = attempt < retries
                self.ledger.write({'request': request_id, 'role': self.role, 'stage': self.ledger.stage,
                                   'attempt': attempt+1, 'will_retry': retry,
                                   'error': 'network/timeout; may have been billed',
                                   'error_type': type(exc).__name__,
                                   'reason_type': type(getattr(exc, 'reason', None)).__name__,
                                   'elapsed': round(time.time()-started, 3)})
                if not retry:
                    raise RuntimeError(f'Provider network/timeout failure after {attempt+1} attempt(s); reservation retained.') from None
            time.sleep(min(attempt+1, 2))
        content = payload['choices'][0]['message'].get('content') or ''
        usage = payload.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        reasoning_tokens = usage.get('completion_tokens_details', {}).get('reasoning_tokens')
        stream_info = payload.get('stream_diagnostics') or {}
        reasoning_observed = (isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0) or stream_info.get('reasoning_characters', 0) > 0
        if body.get('enable_thinking') is False and reasoning_observed:
            print('Provider still returned reasoning despite enable_thinking=false; streaming remains active.', flush=True)
        with self.ledger.lock:
            if 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
                self.ledger.missing_usage_responses += 1
            self.ledger.prompt_tokens += prompt_tokens
            self.ledger.completion_tokens += completion_tokens
            if self.ledger.pricing_known:
                self.ledger.cost += (prompt_tokens*self.config['input_cny_per_million'] +
                                     completion_tokens*self.config['output_cny_per_million']) / 1_000_000
        self.ledger.write({'mode': 'LIVE_API', 'request': request_id, 'role': self.role,
                           'stage': self.ledger.stage, 'messages': messages, 'response': content,
                           'usage': usage, 'elapsed': round(time.time()-started, 3),
                           'finish_reason': payload['choices'][0].get('finish_reason'),
                           'parameters': {k: v for k, v in body.items() if k not in ('messages',)},
                           'stream_diagnostics': payload.get('stream_diagnostics'),
                           'reasoning_observed': reasoning_observed})
        if payload['choices'][0].get('finish_reason') == 'length':
            raise RuntimeError('Model output was truncated by its output limit; partial program is not scored.')
        if not content.strip():
            raise RuntimeError('Empty answer content, possibly reasoning exhausted output budget. Inspect provider model settings.')
        self.history.append({'messages': messages, 'outputs': [content]})
        return [content]
