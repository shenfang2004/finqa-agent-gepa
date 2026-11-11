"""Question checkpoints, isolated failures and coverage-aware evaluation."""
from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

from core import dump, inputs, load, score_prediction


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def generation_settings(config):
    if config is None:
        return {'model': 'scripted-teaching-backend'}
    return {k: config.get(k, 'provider_default') for k in
            ('base_url', 'model', 'max_output_tokens', 'enable_thinking')}


def resolve_resume(root, value):
    if not value:
        return None
    if value == 'latest':
        options = [p for p in (root/'runs').glob('live-*') if
                   (p/'summary.json').is_file() or (p/'run-manifest.json').is_file()]
        if not options:
            raise ValueError('No existing live run to resume.')
        return max(options, key=lambda p: p.stat().st_mtime).resolve()
    path = Path(value)
    path = path if path.is_absolute() else root/path
    if not ((path/'summary.json').is_file() or (path/'run-manifest.json').is_file()):
        raise ValueError('Resume directory must contain summary.json or an in-progress run-manifest.json.')
    return path.resolve()


def stage_statistics(rows, expected):
    complete = [r for r in rows if r.get('status', 'completed') == 'completed' and r.get('score') is not None]
    count = len(complete)
    correct = sum(r['score'] for r in complete)
    attempted = sum(r.get('status') != 'not_attempted' for r in rows)
    return {'expected': expected, 'completed': count, 'incomplete': expected-count,
            'attempted': attempted, 'correct': int(correct),
            'accuracy_completed_only': correct/count if count else None,
            'executable_rate_completed_only': sum(r['executable'] for r in complete)/count if count else None,
            'reused': sum(bool(r.get('reused')) for r in complete)}


def paired_statistics(before, after):
    left = {r['id']:r for r in before if r.get('status','completed') == 'completed' and r.get('score') is not None}
    right = {r['id']:r for r in after if r.get('status','completed') == 'completed' and r.get('score') is not None}
    ids = sorted(left.keys() & right.keys())
    return {'completed_pairs': len(ids), 'ids': ids,
            'baseline_accuracy': sum(left[i]['score'] for i in ids)/len(ids) if ids else None,
            'optimized_accuracy': sum(right[i]['score'] for i in ids)/len(ids) if ids else None}


class EvaluationCheckpoint:
    def __init__(self, directory, config, source=None, baseline_instructions=''):
        self.directory, self.config, self.source = Path(directory), config, source
        self.baseline_instructions = baseline_instructions
        self.meta = {'generation': generation_settings(config), 'stages': {}}
        self.active_record_id = None

    def source_rows(self, stage, instructions):
        if not self.source or not (self.source/f'{stage}.json').exists():
            return {}
        meta_path = self.source/'checkpoint-meta.json'
        if meta_path.exists():
            saved = load(meta_path)
            if saved['generation'] != self.meta['generation']:
                raise ValueError('Cannot reuse results after changing model, thinking mode or output limit. Start a new run for the changed setting.')
            previous = saved.get('stages', {}).get(stage)
            if previous and previous['instructions_digest'] != digest(instructions):
                # A newly optimized prompt may legitimately differ from the old winner.
                return {}
        else:
            # Legacy run import: validate the generation config and actual baseline prompt.
            saved = load(self.source/'config.sanitized.json')
            if generation_settings(saved) != self.meta['generation']:
                raise ValueError('Legacy results use different generation settings; resume without model/thinking overrides.')
            if stage == 'baseline_test':
                calls = self.source/'calls.jsonl'
                matched = False
                if calls.exists():
                    for line in calls.read_text(encoding='utf-8').splitlines():
                        call = json.loads(line)
                        if call.get('stage') == stage and call.get('response'):
                            system = '\n'.join(m.get('content','') for m in call.get('messages',[]) if m.get('role')=='system')
                            objective = system.split('In adhering to this structure, your objective is:', 1)
                            matched = len(objective) == 2 and textwrap.dedent(objective[1]).strip() == self.baseline_instructions.strip()
                            break
                if not matched:
                    raise ValueError('Legacy baseline prompt cannot be verified; not reusing its predictions.')
            elif stage == 'optimized_test':
                p = self.source/'optimized-instructions.json'
                if not p.exists() or load(p).get('instructions') != instructions:
                    return {}
            else:
                return {}
        return {r['id']:r for r in load(self.source/f'{stage}.json')
                if r.get('status','completed') == 'completed' and r.get('program') is not None and r.get('score') is not None}

    def evaluate(self, program, records, stage, instructions, budget_error):
        reusable = self.source_rows(stage, instructions)
        self.meta['stages'][stage] = {'instructions_digest': digest(instructions),
                                     'ids': [r['id'] for r in records]}
        dump(self.directory/'checkpoint-meta.json', self.meta)
        rows, budget_stopped = [], False
        for index, record in enumerate(records, 1):
            self.active_record_id = record['id']
            model_input = inputs(record)
            row = {'id':record['id'], 'question':record['question'], 'input':model_input,
                   'gold_answer':record['answer'], 'reused':False}
            old = reusable.get(record['id'])
            if old is not None and old.get('input') != model_input:
                raise ValueError(f'Input changed for cached question {record["id"]}; refusing reuse.')
            print(f'[{stage}] {index}/{len(records)} {record["id"]}' + (' [reuse]' if old is not None else ''), flush=True)
            try:
                if old is not None:
                    prediction = old['program']
                    row.update(reused=True, source_run=str(self.source))
                elif budget_stopped:
                    row.update(status='not_attempted', score=None, executable=None, program=None,
                               error='Request or spend cap reached; not attempted.')
                    rows.append(row)
                    dump(self.directory/f'{stage}.json', rows)
                    continue
                else:
                    prediction = program(**model_input).program
                row.update(program=prediction, status='completed',
                           **score_prediction(record, prediction, detailed=False))
            except budget_error as exc:
                budget_stopped = True
                row.update(status='not_attempted', program=None, score=None, executable=None,
                           error=str(exc), error_type=type(exc).__name__)
            except Exception as exc:
                # No prediction was received: not a wrong answer and not a zero score.
                row.update(status='incomplete', program=None, score=None, executable=None,
                           error=str(exc)[:800], error_type=type(exc).__name__)
                print(f'  Incomplete: {type(exc).__name__}; continuing with the next question.', flush=True)
            rows.append(row)
            dump(self.directory/f'{stage}.json', rows)
        self.active_record_id = None
        return rows, budget_stopped
