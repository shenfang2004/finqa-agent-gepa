import unittest
from core import execute, is_equal, inputs, synthetic_sets, score_prediction, document_id

class EvaluationChecks(unittest.TestCase):
    def test_reference_arithmetic_and_percentage(self):
        self.assertEqual(execute('subtract(120, 100), divide(#0, 100)', []), .2)
        self.assertEqual(execute('multiply(20%, const_100)', []), 20.)
        self.assertEqual(execute('greater(1, 2)', []), 'no')
        self.assertEqual(execute('divide(const_m1, 2)', []), -.5)

    def test_table_aggregation_and_rounding(self):
        table = [['revenue', '1,000', '2,000', '3,000']]
        self.assertEqual(execute('table_average(revenue, none)', table), 2000.)
        self.assertTrue(is_equal(execute('divide(1, 3)', []), .33333))

    def test_nested_expressions_and_infix_preserve_arithmetic(self):
        self.assertEqual(execute('divide(add(120,0),add(100,0))-1', []), .2)
        self.assertEqual(execute('divide(250000, 250000 + 2014 + 2014)', []), .98414)
        self.assertEqual(execute('add(2, 3), multiply(#0, 4) - 1', []), 19.)
        self.assertEqual(execute('(2+3)*4', []), 20.)
        self.assertEqual(execute('10-3-2', []), 5.)
        self.assertEqual(execute('-2 * (3 + 1)', []), -8.)
        self.assertEqual(execute('divide(20%, .5)', []), .4)

    def test_literal_rows_and_transport_wrappers(self):
        table = [['net revenue', '100', '120']]
        self.assertEqual(execute('table_sum("net revenue", none)', table), 220.)
        self.assertEqual(execute("table_average('net revenue', none)", table), 110.)
        self.assertEqual(execute('table_sum(net revenue, none)', table), 220.)
        self.assertEqual(execute('table_max(s&p 500 index, none)', [['s&p 500 index','10','20']]), 20.)
        self.assertEqual(execute('table_average(u.k ., none)', [['u.k .','10','20']]), 15.)
        self.assertEqual(execute('```text\nadd(1,2)\n```', []), 3.)
        self.assertEqual(execute('add(1,2) [[ ## completed ## ]]', []), 3.)
        self.assertEqual(execute('add(1,2), EOF', []), 3.)
        with self.assertRaises(ValueError):
            execute('table_sum("net revenue", "2019")', table)
        with self.assertRaises(ValueError):
            execute('table_sum(revenue, none)', [['revenue','10'],['revenue','20']])

    def test_unsafe_and_excessive_expressions_are_rejected(self):
        invalid = ['open("file", "w")', 'add(1,2).__class__', '2**100000',
                   '[1,2]', 'add(1,2) # comment', 'add(1,2),',
                   'table_sum(#0, none)', 'add(1,2) [[ ## completed ## ]] garbage',
                   'add(' * 40 + '1,2' + ')' * 40, '1+' * 40 + '1',
                   '1' * 8200, '1e999', 'exp(10, 1000)']
        for program in invalid:
            with self.subTest(program=program[:80]):
                with self.assertRaises((ValueError, ZeroDivisionError)):
                    execute(program, [])

    def test_invalid_programs_do_not_execute(self):
        for program in ['__import__(os)', 'divide(1, 0)', 'add(#0, 1)',
                        'subtract(120, 100); print(1)', 'add(1,2) garbage']:
            with self.subTest(program=program):
                with self.assertRaises((ValueError, ZeroDivisionError)):
                    execute(program, [])

    def test_task_input_excludes_all_gold(self):
        record = synthetic_sets()['train'][0]
        self.assertEqual(set(inputs(record)), {'context', 'question'})
        self.assertNotIn('gold_program', inputs(record)['context'])
        self.assertNotIn('answer', inputs(record)['context'])

    def test_teaching_split_is_disjoint_and_labels_execute(self):
        sets = synthetic_sets()
        docs = [{document_id(r) for r in rows} for rows in sets.values()]
        self.assertFalse(docs[0] & docs[1] or docs[0] & docs[2] or docs[1] & docs[2])
        for records in sets.values():
            for record in records:
                self.assertEqual(score_prediction(record, record['gold_program'])['score'], 1.)

if __name__ == '__main__':
    unittest.main()
