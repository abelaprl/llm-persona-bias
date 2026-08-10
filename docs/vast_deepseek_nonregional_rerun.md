# DeepSeek non-regional selective rerun on Vast.ai

The default launcher selects only the audited failures and never scans the full
result tree:

- ETHICS: 59 main JSON files (Gender 21, Neutral 3, Religion 35);
- GSM8K: 684 main JSON files (Gender 149, Religion 535);
- total model calls: 743;
- one failed ETHICS warm-up cache is restored locally from the exact historical
  assistant turn embedded in a valid result, so it does not add a model call.

Regional Cultural is not present in either manifest and cannot be rerun by this
launcher.

## Files that must exist on Vast

Upload the generated selective bundle, not only the Python launcher. The bundle
contains the runners, manifests, all 743 source files, the warm-up source/cache,
and the ETHICS parsing dependency.

## Commands inside the Vast instance

Install the small client dependency:

```bash
python -m pip install -r requirements-vast-rerun.txt
```

Start vLLM in a separate shell, using the exact served-model alias expected by
the experiment:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --served-model-name deepseek/deepseek-r1-distill-qwen-32b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 16384
```

Validate all paths without API calls or writes:

```bash
python scripts/rerun_deepseek_nonregional_vast.py
```

Run a smoke test. `--limit 3` means three ETHICS and three GSM8K targets:

```bash
python scripts/rerun_deepseek_nonregional_vast.py \
  --execute --limit 3 --concurrency 1
```

After the smoke test succeeds, resume the complete rerun. The six completed
files are skipped automatically:

```bash
python scripts/rerun_deepseek_nonregional_vast.py \
  --execute --concurrency 4
```

Successful responses atomically replace only manifested targets. Each source is
backed up first. Failed attempts are written separately and never overwrite the
source file.
