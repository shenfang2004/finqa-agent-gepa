"""Usage: python run.py demo | prepare | models | smoke | live."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import time

from core import ROOT, DATA, dump, load, inputs, prepare_finqa, prepare_from_cached_train, score_prediction, synthetic_sets, EXECUTOR_VERSION
from checkpoints import EvaluationCheckpoint, generation_settings, resolve_resume, stage_statistics, paired_statistics, digest

def html_report(directory, summary, before, after, candidates, examples):
    def pre(value):
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        return '<pre>' + html.escape(value) + '</pre>'
    mode_note = ('这是脚本模拟：使用合成题和预设模型行为，分数变化仅用于展示流程，不是FinQA或真实大模型结果。'
                 if summary['mode'] == 'SCRIPTED_SIMULATION' else
                 '这是远程模型API实验。小规模过滤子集结果，不是FinQA官方完整榜单结果。')
    by_id = {x['id']: x for x in after}
    rows = []
    for first in before:
        second = by_id.get(first['id'])
        rows.append('<details><summary>' + html.escape(first['question']) + '</summary>' +
                    pre({'report': first['input'], 'gold_answer_for_evaluation_only': first['gold_answer'],
                         'baseline': first, 'optimized': second}) + '</details>')
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DSPy + GEPA 完整流程</title>
<style>body{font:16px/1.7 system-ui,sans-serif;background:#f4f6fa;color:#182238;margin:0}
main{max-width:1050px;margin:35px auto;padding:25px}section{background:white;padding:24px;border-radius:14px;margin:20px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f9;padding:18px;border-radius:8px;font-size:13px}
.note{background:#fff0ce;padding:18px;border-left:5px solid #d89a13}summary{cursor:pointer;font-weight:600;padding:12px}
h1{font-size:30px}h2{font-size:22px}small{color:#546078}a{color:#1355ac}</style><main>
<h1>DSPy + GEPA：从做题到提示词优化</h1>'''
    content += '<p class="note">' + mode_note + '</p>'
    def percent(value):
        return '尚未完成' if value is None else f'{value:.1%}'
    content += '<section><h2>1. 这次运行发生了什么</h2>'
    content += '<p>固定提示词测试正确率：<strong>' + percent(summary['baseline_accuracy']) + '</strong>；优化后：<strong>' + percent(summary['optimized_accuracy']) + '</strong>。</p>'
    if 'coverage' in summary:
        for label, key in [('基线', 'baseline'), ('优化后', 'optimized')]:
            item = summary['coverage'][key]
            content += f'<p>{label}：计划{item["expected"]}题，完成{item["completed"]}题，未完成{item["incomplete"]}题，复用{item["reused"]}题。正确率分母仅为已完成题；网络失败不计为答错。</p>'
        content += '<p>两种提示词都完成的配对比较：</p>' + pre(summary.get('paired_comparison'))
    content += '<p>优化期间只使用训练/验证题，最后将选出的提示词固定，再回答测试题。下面可展开查看每次输出。模拟模式的改善是脚本规则预设的；GEPA搜索过程本身实际运行。</p>'
    content += '<details><summary>完整运行参数与统计</summary>' + pre(summary) + '</details></section>'
    content += '''<section><h2>2. 按这个顺序看代码</h2><ol>
<li>core.py / inputs：只将财报和问题交给做题模型。</li>
<li>run.py / FinancialProgram：DSPy定义输入字段和program输出。</li>
<li>core.py / execute：执行受限计算程序；score_prediction与标准答案比较。</li>
<li>run.py / metric：把分数和训练样本错误反馈交给GEPA。</li>
<li>dspy.GEPA.compile：真实GEPA包生成、比较候选提示词。</li>
<li>run.py / evaluate：冻结提示词，在未参与优化的测试题上评分。</li>
</ol><p>模型权重没有更新。标准答案只进入训练反馈和评分器；测试标签不参与提示词搜索。</p></section>'''
    content += '<section><h2>3. GEPA比较过的候选提示词</h2>' + pre(candidates) + '</section>'
    content += '<section><h2>4. 评分器产生的反馈示例（开发阶段）</h2>' + pre(examples[:6]) + '</section>'
    content += '<section><h2>5. 测试题：展开查看前后输出</h2>' + ''.join(rows) + '</section>'
    content += '<section><h2>6. 文件入口</h2><p><a href="summary.json">汇总JSON</a> · <a href="calls.jsonl">模型输入输出日志</a> · <a href="feedback.jsonl">优化评分与反馈</a> · <a href="candidates.json">候选提示词</a></p></section></main></html>'
    (directory / 'report.html').write_text(content, encoding='utf-8')

def run_experiment(mode, args):
    # Import only for actual execution; data preparation needs no third-party packages.
    import dspy
    from models import BASE_INSTRUCTION, ScriptedLM, CompatibleLM, Ledger, BudgetExceeded, read_config, model_override
    from optimization_checkpoint import ReplayLM, guard_optimization_scores

    source = resolve_resume(ROOT, getattr(args, 'resume', None))
    retry_only = getattr(args, 'retry_failed', False)
    if retry_only and source is None:
        raise ValueError('--retry-failed requires --resume RUN_DIRECTORY or --resume latest.')

    directory = ROOT / 'runs' / (mode + '-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
    directory.mkdir(parents=True)
    if mode == 'demo':
        config, sets = None, synthetic_sets()
        ledger = Ledger(directory)
        task_lm, reflection_lm = ScriptedLM(ledger, 'task'), ScriptedLM(ledger, 'reflection')
    else:
        current_config = read_config(args.config, require_prices=False)
        saved_config = load(source/'config.sanitized.json') if source else None
        if saved_config:
            if current_config['base_url'] != saved_config['base_url']:
                raise ValueError('Resume requires the same API endpoint as the source run.')
            current_config = {**saved_config, 'api_key':current_config['api_key']}
        config = model_override(current_config,
                                args.model, args.input_price, args.output_price,
                                getattr(args, 'unpriced', False) or current_config.get('allow_unpriced', False))
        if getattr(args, 'timeout', None) is not None:
            config['timeout_seconds'] = args.timeout
        if getattr(args, 'retries', None) is not None:
            config['max_retries'] = args.retries
        # Qwen hybrid models otherwise default to potentially long thinking runs.
        if config['model'].lower().startswith('qwen'):
            thinking = getattr(args, 'thinking', None)
            if thinking is not None or not source:
                config['enable_thinking'] = thinking == 'on'
            config['stream'] = True
            config['stream_deadline_seconds'] = 300
        if saved_config and generation_settings(config) != generation_settings(saved_config):
            raise ValueError('Resume must keep the original model, output limit and thinking mode; omit --model/--thinking overrides to reuse completed questions.')
        # Data must be prepared separately; no hidden downloads or changed splits.
        dataset_dir = Path(args.data_dir)
        if not dataset_dir.is_absolute():
            dataset_dir = ROOT / dataset_dir
        sets = {name: load(dataset_dir / f'{name}.json')[:n] for name,n in
                [('train', args.train), ('dev', args.dev), ('test', args.test)]}
        if any(not value for value in sets.values()):
            raise ValueError('Empty dataset split; run prepare first.')
        ledger = Ledger(directory, config)
        task_lm, reflection_lm = CompatibleLM(config, ledger, 'task'), CompatibleLM(config, ledger, 'reflection')
        task_lm = ReplayLM(task_lm, ledger, config, 'task', source)
        reflection_lm = ReplayLM(reflection_lm, ledger, config, 'reflection', source)
        dump(directory / 'config.sanitized.json', {k:v for k,v in config.items() if k != 'api_key'})
        if (dataset_dir / 'source.json').exists():
            dump(directory / 'data.source.json', load(dataset_dir / 'source.json'))
    dspy.configure(lm=task_lm, adapter=dspy.ChatAdapter(), track_usage=False)

    class FinancialProgram(dspy.Signature):
        context: str = dspy.InputField(desc='Only the report text and table, not gold evidence or answers.')
        question: str = dspy.InputField()
        program: str = dspy.OutputField(desc='A valid FinQA DSL calculation program.')
    FinancialProgram.__doc__ = BASE_INSTRUCTION
    signature = FinancialProgram.with_instructions(BASE_INSTRUCTION)
    student = dspy.Predict(signature)
    checkpoint = EvaluationCheckpoint(directory, config, source, BASE_INSTRUCTION)
    data_digest = digest(sets)
    if source and (source/'run-manifest.json').exists():
        if load(source/'run-manifest.json')['data_digest'] != data_digest:
            raise ValueError('Resume dataset or split sizes changed; use the same data and split-size arguments.')
    dump(directory/'run-manifest.json', {'data_digest':data_digest, 'resumed_from':str(source) if source else None,
                                        'generation':generation_settings(config), 'baseline_instructions':BASE_INSTRUCTION})
    if source:
        print('Resuming from:', source, flush=True)

    def as_example(record):
        return dspy.Example(**inputs(record), record=record).with_inputs('context', 'question')
    feedback_examples = []
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        result = score_prediction(gold.record, getattr(pred, 'program', ''), args.feedback == 'detailed')
        event = {'stage': ledger.stage, 'id': gold.record['id'], 'program': getattr(pred, 'program', ''), **result}
        with (directory / 'feedback.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        feedback_examples.append(event)
        if pred_name is not None:
            return dspy.Prediction(score=result['score'], feedback=result['feedback'])
        return result['score']

    def evaluate(program, records, stage):
        ledger.stage = stage
        instructions = BASE_INSTRUCTION if stage in ('baseline_test','smoke_dev') else optimized_instructions
        rows, stopped = checkpoint.evaluate(program, records, stage, instructions, BudgetExceeded)
        if stopped:
            raise BudgetExceeded('Question evaluation stopped at request/spend cap; checkpoint saved.')
        return rows

    before, after, candidates, status = [], [], [], 'complete'
    optimization_status, optimized_instructions = 'not_run', None
    try:
        print('1/4 Baseline test evaluation', flush=True)
        before = evaluate(student, sets['dev'][:1] if mode == 'smoke' else sets['test'], 'baseline_test' if mode != 'smoke' else 'smoke_dev')
        if mode == 'smoke' and not all(row.get('status') == 'completed' and row['executable'] for row in before):
            status = 'invalid_program'
            print('Model responded, but the calculation program cannot execute; fix format before optimization.', flush=True)
        if mode != 'smoke':
            frozen_path = source/'optimized-instructions.json' if source else None
            if frozen_path and frozen_path.exists():
                print('2/4 Reusing frozen optimized prompt; GEPA will not run again.', flush=True)
                optimized_instructions = load(frozen_path)['instructions']
                optimized = dspy.Predict(signature.with_instructions(optimized_instructions))
                candidates = load(source/'candidates.json') if (source/'candidates.json').exists() else []
                optimization_status = 'complete'
            elif not retry_only:
                print('2/4 GEPA prompt search (training + validation only)', flush=True)
                ledger.stage = 'optimization_train_validation'
                optimization_status = 'incomplete'
                optimizer = dspy.GEPA(metric=metric, reflection_lm=reflection_lm,
                    max_metric_calls=40 if mode == 'demo' else config['max_metric_calls'],
                    reflection_minibatch_size=3, num_threads=1, seed=42,
                    track_stats=True, log_dir=str(directory / 'gepa'))
                with guard_optimization_scores(ledger):
                    optimized = optimizer.compile(student=student,
                        trainset=[as_example(r) for r in sets['train']],
                        valset=[as_example(r) for r in sets['dev']])
                results = optimized.detailed_results
                for index, candidate in enumerate(results.candidates):
                    candidates.append({'index': index, 'validation_score': float(results.val_aggregate_scores[index]),
                                       'instructions': candidate.signature.instructions})
                optimized_instructions = optimized.signature.instructions
                optimization_status = 'complete'
            if optimized_instructions is not None:
                # Freeze before testing, so a later timeout cannot discard the winner.
                dump(directory/'optimized-instructions.json', {'instructions':optimized_instructions})
                dump(directory/'candidates.json', candidates)
                print('3/4 Frozen optimized prompt: held-out test evaluation', flush=True)
                after = evaluate(optimized, sets['test'], 'optimized_test')
            else:
                print('3/4 Retry-only scope: baseline missing questions only; GEPA was not started.', flush=True)
    except BudgetExceeded:
        status = 'stopped_at_budget_limit'
    except Exception as exc:
        status = 'failed'
        dump(directory / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)[:800],
            'stage': ledger.stage, 'record_id': checkpoint.active_record_id})
        print(f'Experiment failed: {type(exc).__name__}: {str(exc)[:300]}; see failure.json', file=sys.stderr)
    finally:
        if not before:
            partial = directory / ('smoke_dev.json' if mode == 'smoke' else 'baseline_test.json')
            if partial.exists():
                before = load(partial)
        if not after and (directory / 'optimized_test.json').exists():
            after = load(directory / 'optimized_test.json')
        expected_before = 1 if mode == 'smoke' else len(sets['test'])
        before_stats = stage_statistics(before, expected_before)
        after_stats = stage_statistics(after, len(sets['test']) if mode != 'smoke' else 0)
        asked_after = mode != 'smoke' and (not retry_only or optimized_instructions is not None)
        if status == 'complete' and (before_stats['incomplete'] or (asked_after and after_stats['incomplete'])):
            status = 'partial'
        incomplete = [{'stage':stage, **row} for stage, rows in [('baseline_test',before),('optimized_test',after)]
                      for row in rows if row.get('status') != 'completed']
        dump(directory/'incomplete_questions.json', incomplete)
        summary = {'mode': 'SCRIPTED_SIMULATION' if mode == 'demo' else 'LIVE_API',
            'model': 'scripted-teaching-backend' if config is None else config['model'],
            'executor_version': EXECUTOR_VERSION,
            'status': status, 'feedback': args.feedback,
            'resumed_from': str(source) if source else None, 'retry_failed_only':retry_only,
            'optimization_status':optimization_status,
            'pipeline_complete': status == 'complete' and optimization_status == 'complete' and not after_stats['incomplete'],
            'dataset': 'synthetic teaching data' if mode == 'demo' else f'FinQA filtered small subset: {args.data_dir}',
            'split_sizes': {k:len(v) for k,v in sets.items()},
            'test_completed_before': before_stats['completed'], 'test_completed_after': after_stats['completed'],
            'coverage': {'baseline':before_stats,'optimized':after_stats},
            'paired_comparison':paired_statistics(before,after),
            'candidate_count': len(candidates),
            'baseline_accuracy': before_stats['accuracy_completed_only'], 'optimized_accuracy': after_stats['accuracy_completed_only'],
            'baseline_executable_rate': before_stats['executable_rate_completed_only'],
            'optimized_executable_rate': after_stats['executable_rate_completed_only'],
            'usage': ledger.summary(), 'model_weights_updated': False,
            'limitations': ['Small educational run; no statistical significance claim.',
                'Mock improvement is intentionally scripted, not model learning.' if mode == 'demo' else
                'Verify chosen model prices and provider billing. Reasoning models may need larger output budgets.']}
        dump(directory / 'summary.json', summary)
        dump(directory / 'baseline.json', before)
        dump(directory / 'optimized.json', after)
        dump(directory / 'candidates.json', candidates)
        html_report(directory, summary, before, after, candidates, feedback_examples)
        print('4/4 Report:', directory / 'report.html', flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if status == 'complete' else (2 if status == 'partial' else 1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['demo','prepare','models','smoke','live'])
    parser.add_argument('--config', default=str(ROOT / 'config.local.json'))
    parser.add_argument('--model', help='Override the model for this run; saved config is unchanged.')
    parser.add_argument('--input-price', type=float, help='CNY per million input tokens for the selected model.')
    parser.add_argument('--output-price', type=float, help='CNY per million output tokens for the selected model.')
    parser.add_argument('--unpriced', action='store_true', help='Explicitly run without a cost estimate; request and token caps remain active.')
    parser.add_argument('--timeout', type=int, choices=range(1,301), metavar='SECONDS', help='Per-attempt HTTP timeout, 1..300 seconds.')
    parser.add_argument('--retries', type=int, choices=[0,1,2], help='Retries for transient HTTP/network failures; every attempt counts toward limits.')
    parser.add_argument('--thinking', choices=['off','on'], default=None, help='Qwen thinking mode: new runs default off; resumed runs inherit their original setting.')
    parser.add_argument('--resume', help='Continue a previous run, e.g. runs/live-... or latest; completed predictions are reused.')
    parser.add_argument('--retry-failed', action='store_true', help='With --resume, only fill missing evaluations; do not restart GEPA.')
    parser.add_argument('--feedback', choices=['score','detailed'], default='detailed')
    parser.add_argument('--train', type=int, default=12)
    parser.add_argument('--dev', type=int, default=6)
    parser.add_argument('--test', type=int, default=10)
    parser.add_argument('--data-dir', default='data/train-derived')
    parser.add_argument('--source', choices=['official','cached-train'], default='official')
    args = parser.parse_args()
    if min(args.train, args.dev, args.test) < 1:
        parser.error('All split sizes must be positive.')
    if args.command == 'prepare':
        if args.source == 'cached-train':
            target = Path(args.data_dir)
            target = target if target.is_absolute() else ROOT / target
            if target.resolve() == DATA.resolve():
                parser.error('Use --data-dir data/train-derived to keep local holdout separate from official splits.')
            sets = prepare_from_cached_train(target, args.train, args.dev, args.test)
        else:
            sets = prepare_finqa(args.train, args.dev, args.test)
        print({k:len(v) for k,v in sets.items()})
        print('Source provenance:', (target if args.source == 'cached-train' else DATA) / 'source.json')
        return 0
    if args.command == 'models':
        from models import read_config, list_models
        for model in list_models(read_config(args.config, require_model=False, require_prices=False)):
            print(model)
        return 0
    return run_experiment(args.command, args)

if __name__ == '__main__':
    raise SystemExit(main())
