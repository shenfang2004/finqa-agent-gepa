import json
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

from core import dump, inputs, load
import recover_candidate


class CandidateRecoveryChecks(unittest.TestCase):
    def test_recovery_selects_complete_validation_winner_without_test_answers(self):
        root=Path(__file__).resolve().parent/'work'/('candidate-recovery-check-'+uuid.uuid4().hex)
        source=root/'source'
        data=root/'data'
        dev=[{'id':str(i),'question':f'dev {i}','table':[], 'answer':3.,'gold_program':'add(1, 2)'} for i in range(2)]
        dump(data/'dev.json',dev)
        dump(source/'gepa/candidates.json',[{'self':'baseline'},{'self':'improved'},{'self':'partial'}])
        dump(source/'config.sanitized.json',{'model':'mock'})
        dump(source/'baseline_test.json',[{'id':'heldout','score':0.,'executable':True,'program':'add(1, 1)'}])
        calls=[]
        for instruction in ['baseline','improved','partial']:
            for record in dev[:1] if instruction=='partial' else dev:
                model_input=inputs(record)
                program='add(1, 1)' if instruction=='baseline' else 'add(1, 2)'
                calls.append({'stage':'optimization_train_validation','role':'task',
                              'messages':[{'role':'system','content':'In adhering to this structure, your objective is:\n        '+instruction},
                                          {'role':'user','content':model_input['context']+'\n[[ ## question ## ]]\n'+record['question']}],
                              'response':'[[ ## program ## ]]\n'+program+'\n[[ ## completed ## ]]'})
        (source/'calls.jsonl').write_text('\n'.join(json.dumps(c) for c in calls),encoding='utf-8')
        with patch.object(recover_candidate,'ROOT',root):
            target,winner=recover_candidate.recover(source,data)
        self.assertEqual(winner['index'],1)
        self.assertEqual(winner['validation_score'],1.)
        self.assertEqual(len(load(target/'candidates.json')),2)
        self.assertEqual(load(target/'optimized-instructions.json')['instructions'],'improved')
        self.assertEqual(load(target/'recovery.json')['model_api_calls'],0)
        self.assertFalse((target/'optimized_test.json').exists())


if __name__=='__main__':unittest.main()
