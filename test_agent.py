import json
import unittest

import dspy

from agent import FinancialReportAgent, ReportTools


class AgentChecks(unittest.TestCase):
    def setUp(self):
        self.record={'pre_text':['Revenue in millions.'],'table':[['Year','Revenue'],['2022','100'],['2023','120']],
                     'answer':'SECRET_GOLD_ANSWER','gold_program':'SECRET_GOLD_PROGRAM',
                     'qa':{'answer':'SECRET_QA_LABEL'}}

    def policy(self,steps,seen=None):
        queue=iter(steps)
        def choose(**kwargs):
            if seen is not None:seen.append(kwargs)
            action,arguments=next(queue)
            return dspy.Prediction(action=action,arguments=arguments)
        return choose

    def test_agent_receives_tool_error_and_can_revise_it(self):
        seen=[]
        agent=FinancialReportAgent(max_steps=4)
        agent.policy=self.policy([
            ('read_report',{'source_ids':['table:1','table:2']}),
            ('calculate',{'program':'divide(20, 0)'}),
            ('calculate',{'program':'subtract(120,100), divide(#0,100)'}),
            ('finish',{'program':'subtract(120,100), divide(#0,100)',
                       'evidence_ids':['table:1','table:2'],'explanation':'Growth rate is 20%.'})],seen)
        result=agent(record=self.record,question='Revenue growth?')
        self.assertEqual(result.status,'answered')
        self.assertEqual(result.answer,.2)
        self.assertIn('division by zero',seen[2]['observations'])
        self.assertEqual(len(result.actions),4)
        self.assertFalse(result.checks['semantic_correctness_verified'])
        self.assertNotIn('SECRET_',json.dumps(seen))

    def test_cannot_finish_with_unread_or_invented_evidence(self):
        agent=FinancialReportAgent(max_steps=2)
        agent.policy=self.policy([
            ('calculate',{'program':'divide(20,100)'}),
            ('finish',{'program':'divide(20,100)','evidence_ids':['table:99']})])
        result=agent(record=self.record,question='Growth?')
        self.assertEqual(result.status,'needs_review')
        self.assertIsNone(result.answer)

    def test_refuses_to_invent_an_answer_when_evidence_is_missing(self):
        agent=FinancialReportAgent(max_steps=1)
        agent.policy=self.policy([('finish',{'program':'','explanation':'No profit figures in this report.'})])
        result=agent(record=self.record,question='Net profit?')
        self.assertEqual(result.status,'needs_review')
        self.assertIsNone(result.answer)

    def test_tool_registry_is_bounded_and_has_no_arbitrary_execution(self):
        agent=FinancialReportAgent(max_steps=1)
        agent.policy=self.policy([('run_python',{'code':'print(1)'})])
        result=agent(record=self.record,question='Question')
        self.assertEqual(result.status,'needs_review')
        self.assertIn('Unknown tool',result.actions[0]['observation']['error'])
        tools=ReportTools(self.record)
        self.assertFalse(tools.calculate('__import__(os)')['ok'])
        self.assertFalse(tools.read_report(['nonexistent'])['ok'])
        self.assertNotIn('SECRET_',tools.index())

    def test_search_returns_real_source_quotes(self):
        tools=ReportTools(self.record)
        result=tools.search_report('revenue 2023')
        self.assertTrue(result['ok'])
        self.assertTrue(all(x['id'] in tools.sources for x in result['evidence']))
        self.assertNotIn('SECRET_',json.dumps(result))


if __name__=='__main__':unittest.main()
