"""Regression and invariants for the hash-locked frozen-v2 parser."""
import csv, inspect, io, json, re, sys, tempfile, tokenize
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.MAIN_PARSING.GSM8K._core import runner, frozen_v2_skeleton as sk
from src.MAIN_PARSING.GSM8K.Religion_Persona import (
    rules_core, rules_deepseek_r1_distill_32b, rules_gpt_oss_20b,
    rules_llama31_8b, rules_llama33_70b_frozen_v2, rules_ministral_8b,
    rules_qwen3_next_80b,
)
from src.MAIN_PARSING.GSM8K.Religion_Persona.run_frozen_v2 import verify_source_lock

PACKS = [rules_core, rules_deepseek_r1_distill_32b, rules_gpt_oss_20b,
         rules_llama31_8b, rules_llama33_70b_frozen_v2, rules_ministral_8b,
         rules_qwen3_next_80b]


def check(ok, msg):
    if not ok: raise AssertionError(msg)
    print('PASS', msg)


def main():
    verify_source_lock(); check(True, 'source hash lock')
    check(not any(re.search(r'gt|gold|ground_truth|answer_key', p, re.I)
                  for p in inspect.signature(sk.extract).parameters), 'V2 signature no gold')
    roots = [Path(inspect.getfile(sk)), *(Path(inspect.getfile(p)) for p in PACKS)]
    for path in roots:
        src = path.read_text(encoding='utf-8')
        toks = tokenize.generate_tokens(io.StringIO(src).readline)
        code = tokenize.untokenize((t.type, t.string) for t in toks if t.type != tokenize.COMMENT)
        check(not re.search(r'gsm8k_\d{4,5}', code), f'V6 no QID {path.name}')
        check(not re.search(r'\b(islam|muslim|christian|atheist|jewish|\w+_implicit_\d|\w+_explicit)\b', code, re.I),
              f'V7 static persona blind {path.name}')

    check(sk.extract('#### 18. Answer follows.', rules_core).value is None, 'hash heading rejected')
    check(sk.extract('Answer: It will take John 2 weeks to save $80.', rules_qwen3_next_80b).value == 2,
          'answer duration resolver')
    check(sk.extract('Answer: 448 mL of medicine would be in 8 doses.', rules_qwen3_next_80b).value == 448,
          'answer quantity resolver')
    check(sk.extract('Final Answer: 7', rules_core, 'error').status == 'missing_content', 'error is missing')
    check(sk.extract('So the total is 7.', rules_qwen3_next_80b, 'length').value is None, 'weak length abstains')

    with tempfile.TemporaryDirectory() as td:
        raw, out = Path(td)/'raw', Path(td)/'out'; personas=[f'p{i}' for i in range(16)]
        payload={'input':{'question_id':1,'ground_truth':'#### 18'},
                 'response':{'success':True,'content':'Final Answer: 18',
                             'raw_api_response':{'choices':[{'finish_reason':'stop'}]}}}
        for p in personas:
            d=raw/'model'/p; d.mkdir(parents=True); (d/'gsm8k_00001.json').write_text(json.dumps(payload))
        old=runner.extract; runner.extract=sk.extract
        try: result=runner.parse_model(str(raw),'model','group',personas,rules_core,str(out),progress=False)
        finally: runner.extract=old
        rows=list(csv.DictReader(open(result['results_path'],encoding='utf-8')))
        check(len(rows)==16 and len({(r['extracted_answer'],r['parse_source']) for r in rows})==1,
              'V7 dynamic runner 16 persona')
        model_bytes=(out/'results.csv').read_bytes()
        old=runner.extract; runner.extract=sk.extract
        try: runner.parse_model(str(raw),'model','group',personas,rules_core,str(out),progress=False,
                                results_filename='results_core.csv')
        finally: runner.extract=old
        check((out/'results.csv').read_bytes()==model_bytes, 'core isolation bytes')
        bad=raw/'model'/personas[0]/'gsm8k_00002.json'; bad.write_text('{bad')
        old=runner.extract; runner.extract=sk.extract
        try: inv=runner.parse_model(str(raw),'model','group',[personas[0]],rules_core,str(out),progress=False,
                                   results_filename='invalid.csv')
        finally: runner.extract=old
        rr=list(csv.DictReader(open(inv['results_path'],encoding='utf-8')))
        check(any(r['input_json_valid']=='False' for r in rr), 'invalid JSON explicit row')
        final=out/'atomic.csv'; final.write_bytes(b'OLD'); real=runner.os.replace; seen=[]
        def observe(src,dst): seen.append(Path(dst).read_bytes()); real(src,dst)
        with mock.patch.object(runner.os,'replace',side_effect=observe):
            runner._write_csv_atomic(str(final),['x'],[{'x':'new'}])
        check(seen==[b'OLD'], 'atomic old final visible until replace')
    print('ALL FROZEN V2 TESTS PASSED')


if __name__ == '__main__': main()
