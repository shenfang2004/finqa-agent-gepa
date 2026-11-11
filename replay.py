"""Re-score stored outputs offline; does not call models or overwrite old results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from core import ROOT, EXECUTOR_VERSION, dump, score_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Existing run directory, absolute or relative to project.')
    args = parser.parse_args()
    source = Path(args.run)
    source = source if source.is_absolute() else ROOT / source
    source = source.resolve()
    output = ROOT / 'runs' / ('replay-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
    reports = {}
    for filename in ('baseline_test.json', 'optimized_test.json'):
        rows = json.loads((source / filename).read_text(encoding='utf-8'))
        revised = []
        for row in rows:
            context = json.loads(row['input']['context'])
            # Answer is supplied only to the scorer, never to the expression parser.
            record = {'table': context['table'], 'answer': row['gold_answer']}
            result = score_prediction(record, row['program'], detailed=False)
            revised.append({'id': row['id'], 'program': row['program'],
                            'original_score': row['score'], 'original_executable': row['executable'],
                            'original_error': row.get('error'), **result})
        dump(output / filename, revised)
        reports[filename] = {
            'n': len(rows), 'original_correct': sum(r['score'] for r in rows),
            'replayed_correct': sum(r['score'] for r in revised),
            'original_executable': sum(r['executable'] for r in rows),
            'replayed_executable': sum(r['executable'] for r in revised),
        }
    summary = {'source_run': str(source), 'executor_version': EXECUTOR_VERSION,
               'mode': 'OFFLINE_REPLAY', 'model_api_calls': 0, 'results': reports,
               'limitations': ['Stored model outputs are unchanged.',
                   'This is an evaluator compatibility audit, not new model or GEPA improvement.',
                   'These inspected examples are diagnostic data; use fresh held-out data for final evaluation.']}
    dump(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Offline replay:', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
