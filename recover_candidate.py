"""Recover a validated GEPA candidate from JSON logs; never unpickle or call a model."""
import argparse
from datetime import datetime, timezone
import re
import shutil
import textwrap

from checkpoints import resolve_resume, stage_statistics
from core import ROOT, dump, load, inputs, score_prediction


def recover(source, data_dir):
    candidates = load(source/'gepa/candidates.json')
    dev = load(data_dir/'dev.json')
    import json
    calls = [json.loads(x) for x in (source/'calls.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    results = []
    for index, candidate in enumerate(candidates):
        scores = {}
        for call in calls:
            if call.get('stage') != 'optimization_train_validation' or call.get('role') != 'task' or not call.get('response'):
                continue
            system = '\n'.join(m.get('content','') for m in call.get('messages',[]) if m.get('role') == 'system')
            objective = system.split('In adhering to this structure, your objective is:',1)
            if len(objective) != 2 or textwrap.dedent(objective[1]).strip() != candidate['self'].strip():
                continue
            user = '\n'.join(m.get('content','') for m in call['messages'] if m.get('role') == 'user')
            match = re.search(r'\[\[\s*##\s*program\s*##\s*\]\]\s*(.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)', call['response'], re.S)
            if not match:
                continue
            for record in dev:
                if inputs(record)['context'] in user and ('[[ ## question ## ]]\n'+record['question']) in user:
                    result = score_prediction(record, match.group(1).strip(), detailed=False)
                    if record['id'] in scores and scores[record['id']] != result['score']:
                        raise ValueError('Repeated validation scores disagree; recovery requires an explicit selection protocol.')
                    scores[record['id']] = result['score']
        if len(scores) == len(dev):
            results.append({'index':index, 'validation_score':sum(scores.values())/len(dev),
                            'instructions':candidate['self'], 'validation_scores':scores})
    if not results:
        raise ValueError('No candidate has a complete, verifiable validation evaluation.')
    # Use validation only. Stable tie break matches the first best candidate.
    winner = max(results, key=lambda x:x['validation_score'])
    target = ROOT/'runs'/('live-recovered-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
    target.mkdir(parents=True)
    for name in ['config.sanitized.json','run-manifest.json','checkpoint-meta.json','data.source.json',
                 'baseline_test.json','calls.jsonl','feedback.jsonl']:
        if (source/name).exists():
            shutil.copyfile(source/name,target/name)
    before = load(source/'baseline_test.json')
    dump(target/'baseline.json',before)
    dump(target/'optimized-instructions.json',{'instructions':winner['instructions']})
    dump(target/'candidates.json',results)
    dump(target/'recovery.json',{'source_run':str(source),'winner_index':winner['index'],
                               'selection':'highest complete validation score; first candidate wins ties',
                               'validation_only':True,'model_api_calls':0})
    dump(target/'summary.json',{'status':'partial','mode':'LIVE_API','model':load(source/'config.sanitized.json')['model'],
                               'optimization_status':'recovered_validated_candidate','pipeline_complete':False,
                               'test_completed_before':len(before),'test_completed_after':0,
                               'coverage':{'baseline':stage_statistics(before,len(before))},
                               'recovered_from':str(source)})
    return target,winner


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--data-dir',default='data/train-derived')
    args=parser.parse_args()
    source=resolve_resume(ROOT,args.source)
    data_dir=ROOT/args.data_dir
    target,winner=recover(source,data_dir)
    print('Recovered:',target)
    print('Candidate:',winner['index'],'validation score:',winner['validation_score'])
    print('No model API calls. Resume this recovered run to perform final testing.')
