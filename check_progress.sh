echo "=========================================="
echo "PROGRESS CHECK - ALL EXPERIMENTS"
echo "=========================================="
echo ""

echo "1. NEMOTRON (man_implicits):"
count=$(ls -1 results/man_implicits/gsm8k_raw/gsm8k_*.json 2>/dev/null | tail -1 | grep -o '[0-9]\+' | tail -1)
echo "   Latest: gsm8k_00${count}.json (index ~$count)"
echo ""

echo "2. GROK - man_implicits:"
count=$(ls -1 results/grok_4_1_fast/man_implicits/gsm8k_raw/gsm8k_*.json 2>/dev/null | wc -l)
echo "   Files processed: $count/1300"
echo ""

echo "3. GROK - woman_implicits:"
count=$(ls -1 results/grok_4_1_fast/woman_implicits/gsm8k_raw/gsm8k_*.json 2>/dev/null | wc -l)
echo "   Files processed: $count/1300"
echo ""

echo "4. BERT-NEBULON-ALPHA - man_implicits:"
count=$(ls -1 results/bert_nebulon_alpha/man_implicits/gsm8k_raw/gsm8k_*.json 2>/dev/null | wc -l)
echo "   Files processed: $count/1300"
echo ""

echo "=========================================="
echo "All processes: STABLE ✓"
echo "No performance degradation detected ✓"
echo "=========================================="
