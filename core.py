"""Data preparation and a small, non-Python FinQA program interpreter."""
from __future__ import annotations
import hashlib
import base64
import json
import math
from pathlib import Path
import random
import re
import urllib.request
import uuid
from interpreter import execute_expression, EXECUTOR_VERSION

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OPS = {'add', 'subtract', 'multiply', 'divide', 'exp', 'greater',
       'table_sum', 'table_average', 'table_min', 'table_max'}

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the sibling name short for atomic replacement in nested Windows checkouts.
    temporary = path.with_name('.' + uuid.uuid4().hex[:12] + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def number(text):
    text = str(text).strip().replace(',', '').replace('$', '')
    if text.startswith('const_'):
        text = text[6:]
        if text == 'm1':
            return -1.0
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    value = float(text[:-1]) / 100 if text.endswith('%') else float(text)
    if not math.isfinite(value):
        raise ValueError('non-finite number')
    return value

def execute(program, table):
    """Execute only bounded arithmetic, DSL operations and literal table rows."""
    return execute_expression(program, table, number)


def _legacy_execute(program, table):
    """Original strict evaluator, retained for offline before/after audits only."""
    text = str(program).strip()
    if text.startswith('```'):
        text = re.sub(r'^```[^\n]*\n|\n?```$', '', text).strip()
    matches = list(re.finditer(r'([a-z_]+)\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)', text))
    if not matches or len(matches) > 32:
        raise ValueError('format: expected op(a,b), at most 32 steps')
    cursor, values = 0, []
    for match in matches:
        if text[cursor:match.start()].strip(' ,\n\r\t'):
            raise ValueError('format: unexpected text before operation')
        op, left, right = (x.strip() for x in match.groups())
        if op not in OPS:
            raise ValueError(f'unsupported operation: {op}')
        def arg(value):
            if value.startswith('#'):
                index = int(value[1:])
                if not 0 <= index < len(values):
                    raise ValueError('reference must point to an earlier step')
                return values[index]
            return number(value)
        if op.startswith('table_'):
            row = next((r[1:] for r in table if str(r[0]).strip() == left), None)
            if row is None:
                raise ValueError(f'table row not found: {left}')
            nums = [number(x) for x in row]
            if not nums:
                raise ValueError('empty table row')
            result = {'table_sum': lambda: sum(nums),
                      'table_average': lambda: sum(nums) / len(nums),
                      'table_min': lambda: min(nums),
                      'table_max': lambda: max(nums)}[op]()
        else:
            a, b = arg(left), arg(right)
            if isinstance(a, str) or isinstance(b, str):
                raise ValueError('cannot use a comparison as a numeric argument')
            if op == 'exp' and (abs(b) > 16 or abs(a) > 1e16):
                raise ValueError('exponent outside demonstration limits')
            result = {'add': lambda: a+b, 'subtract': lambda: a-b,
                      'multiply': lambda: a*b, 'divide': lambda: a/b,
                      'exp': lambda: a**b,
                      'greater': lambda: 'yes' if a>b else 'no'}[op]()
        if not isinstance(result, str) and (isinstance(result, complex) or not math.isfinite(result)):
            raise ValueError('non-finite result')
        values.append(result)
        cursor = match.end()
    if text[cursor:].strip(' ,\n\r\t') not in ('', 'EOF'):
        raise ValueError('format: unexpected text after final operation')
    return values[-1] if isinstance(values[-1], str) else round(values[-1], 5)

def is_equal(actual, expected):
    if isinstance(actual, str) or isinstance(expected, str):
        if str(actual).lower() == str(expected).lower():
            return True
    try:
        return round(number(actual), 5) == round(number(expected), 5)
    except (TypeError, ValueError):
        return False

def score_prediction(record, program, detailed=True):
    try:
        actual = execute(program, record['table'])
        correct = is_equal(actual, record['answer'])
        error = None
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        actual, correct, error = None, False, str(exc)
    feedback = 'Correct.' if correct else 'Incorrect final execution result.'
    if detailed and not correct:
        if error:
            feedback = f'Program cannot execute: {error}.'
        else:
            feedback = f'Executed result: {actual}; expected: {record["answer"]}.'
        feedback += f' Reference program for this TRAINING example: {record["gold_program"]}. '
        feedback += 'Check source values, baseline period, operator order, units and percent representation. Do not memorize example-specific values.'
    return {'score': float(correct), 'feedback': feedback, 'actual': actual,
            'executable': error is None, 'error': error}

def inputs(record):
    """The only fields ever sent to the task model. No gold labels."""
    context = json.dumps({'pre_text': record.get('pre_text', []),
                          'table': record['table'],
                          'post_text': record.get('post_text', [])}, ensure_ascii=False)
    return {'context': context, 'question': record['question']}

def document_id(record):
    return record['id'].rsplit('-', 1)[0]

def prepare_finqa(train_n=12, dev_n=6, test_n=10):
    DATA.mkdir(exist_ok=True)
    request = urllib.request.Request('https://api.github.com/repos/czyssrs/FinQA/commits/main',
                                     headers={'User-Agent': 'finqa-gepa-educational-demo'})
    manifest_path = DATA / 'source.json'
    if manifest_path.exists():
        manifest = load(manifest_path)
        commit = manifest['commit']
    else:
        with urllib.request.urlopen(request, timeout=30) as response:
            commit = json.load(response)['sha']
        manifest = {'repository': 'https://github.com/czyssrs/FinQA', 'commit': commit,
                    'license': 'MIT; see upstream LICENSE', 'files': {}}
        dump(manifest_path, manifest)
    used_docs, sets = set(), {}
    # Preserve the official split assignments; omit overlapping report pages.
    for split, size in [('train', train_n), ('dev', dev_n), ('test', test_n)]:
        target = DATA / 'raw' / f'{split}.json'
        url = f'https://raw.githubusercontent.com/czyssrs/FinQA/{commit}/dataset/{split}.json'
        if not target.exists():
            # Some networks block raw.githubusercontent.com. Use GitHub's
            # official blob API as a fallback, pinned to the same commit.
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    payload = response.read()
            except (OSError, TimeoutError):
                headers = {'User-Agent': 'finqa-gepa-educational-demo'}
                tree_url = f'https://api.github.com/repos/czyssrs/FinQA/git/trees/{commit}?recursive=1'
                with urllib.request.urlopen(urllib.request.Request(tree_url, headers=headers), timeout=30) as response:
                    tree = json.load(response)['tree']
                blob_sha = next(x['sha'] for x in tree if x['path'] == f'dataset/{split}.json')
                blob_url = f'https://api.github.com/repos/czyssrs/FinQA/git/blobs/{blob_sha}'
                with urllib.request.urlopen(urllib.request.Request(blob_url, headers=headers), timeout=60) as response:
                    blob = json.load(response)
                payload = base64.b64decode(blob['content'])
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(payload)
        payload = target.read_bytes()
        manifest['files'][split] = {'url': url, 'sha256': hashlib.sha256(payload).hexdigest()}
        dump(manifest_path, manifest)
        records = json.loads(payload)
        random.Random(42).shuffle(records)
        chosen, local_docs = [], set()
        for item in records:
            qa = item['qa']
            if 'exe_ans' not in qa or 'program' not in qa:
                continue
            record = {k: item[k] for k in ('id', 'table', 'pre_text', 'post_text')}
            record.update(question=qa['question'], answer=qa['exe_ans'], gold_program=qa['program'])
            doc = document_id(record)
            if doc in used_docs or doc in local_docs:
                continue
            try:
                if not is_equal(execute(record['gold_program'], record['table']), record['answer']):
                    continue
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                continue
            chosen.append(record)
            local_docs.add(doc)
            if len(chosen) == size:
                break
        if len(chosen) < size:
            raise ValueError(f'Only {len(chosen)} usable examples in {split}, requested {size}')
        sets[split] = chosen
        used_docs.update(local_docs)
        dump(DATA / f'{split}.json', chosen)
    manifest['selection'] = {'seed': 42, 'counts': {k: len(v) for k,v in sets.items()},
                             'rule': 'official split, one item per report page, no page overlap; gold programs executable under demo DSL',
                             'limitation': 'filtered small subset, not the official full benchmark score'}
    dump(manifest_path, manifest)
    return sets

def synthetic_sets():
    """Synthetic teaching examples: never label these as FinQA results."""
    sets = {}
    for split, offset, count in [('train', 0, 9), ('dev', 50, 6), ('test', 100, 9)]:
        rows = []
        for i in range(count):
            old = 100 + (offset+i)*10
            new = old + 20 + i
            kind = ['growth', 'difference', 'ratio'][i % 3]
            question = {'growth': 'What is the relative revenue growth rate?',
                        'difference': 'What is the absolute increase in revenue?',
                        'ratio': 'What is the ratio of new revenue to old revenue?'}[kind]
            program = {'growth': f'subtract({new}, {old}), divide(#0, {old})',
                       'difference': f'subtract({new}, {old})',
                       'ratio': f'divide({new}, {old})'}[kind]
            table = [['period', 'revenue'], ['old', str(old)], ['new', str(new)]]
            rows.append({'id': f'synthetic/{split}/{i}-0', 'table': table, 'pre_text': [],
                         'post_text': [], 'question': question, 'gold_program': program,
                         'answer': execute(program, table)})
        sets[split] = rows
    return sets

def prepare_from_cached_train(target, train_n=12, dev_n=6, test_n=10):
    """An explicitly labeled local holdout split of upstream TRAIN, no network."""
    source = DATA / 'raw' / 'train.json'
    records = load(source)
    random.Random(42).shuffle(records)
    rows, seen = [], set()
    for item in records:
        qa = item['qa']
        record = {k:item[k] for k in ('id','table','pre_text','post_text')}
        record.update(question=qa['question'], answer=qa['exe_ans'], gold_program=qa['program'])
        doc = document_id(record)
        if doc in seen:
            continue
        try:
            if not is_equal(execute(record['gold_program'], record['table']), record['answer']):
                continue
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
        rows.append(record)
        seen.add(doc)
        if len(rows) == train_n + dev_n + test_n:
            break
    if len(rows) < train_n + dev_n + test_n:
        raise ValueError('Not enough usable report pages in cached training data.')
    sets = {'train': rows[:train_n], 'dev': rows[train_n:train_n+dev_n],
            'test': rows[train_n+dev_n:]}
    for name, value in sets.items():
        dump(Path(target) / f'{name}.json', value)
    provenance = load(DATA / 'source.json')
    provenance.update(original_split='upstream train only',
        local_split_rule='seed=42; one example per report page; locally disjoint train/dev/test',
        raw_train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        counts={k:len(v) for k,v in sets.items()},
        limitation='Educational holdout from upstream TRAIN, not official FinQA test evaluation.')
    dump(Path(target) / 'source.json', provenance)
    return sets
