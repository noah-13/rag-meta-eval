#!/bin/bash
set -e  # Exit immediately if a command fails
set -u  # Treat unset variables as an error

# === 1. Metric Evaluation ===
echo "==> Running metric evaluations..."
echo "[1/3] Evaluating Ragas Metrics..."
python scripts/ragas_ff.py --input_csv input/ff.csv --output_csv output/ragas_ff_output.csv
python scripts/ragas_ff_chat.py --input_csv input/ff.csv --output_csv output/ragas_ff_chat_output.csv
python scripts/ragas_ar.py --input_csv input/ar.csv --output_csv output/ragas_ar_output.csv 
python scripts/ragas_ar_chat.py --input_csv input/ff.csv --output_csv output/ragas_ar_chat_output.csv
python scripts/ragas_cr.py --input_csv input/cr.csv --output_csv output/ragas_cr_output.csv 
python scripts/ragas_cr_chat.py --input_csv input/ff.csv --output_csv output/ragas_cr_chat_output.csv

echo "[2/3] Evaluating GPTscore Metrics..."
python scripts/gptscore_ar_variants.py --input input/ar.csv --output output/gptscore_ar.csv
python scripts/gptscore_ff_variants.py --input input/ff.csv --output output/gptscore_ff.csv
python scripts/gptscore_cr_variants.py --input input/cr.csv --output output/gptscore_cr.csv

echo "[3/3] Evaluating Prompting-relevant Metrics..."
python scripts/prompt_ar_variants.py --input input/ar.csv --output output/prompt_ar.csv
python scripts/prompt_cr_variants.py --input input/cr.csv --output output/prompt_cr.csv
python scripts/prompt_ff_variants.py --input input/ff.csv --output output/prompt_ff.csv

# === 2. Summary ===
echo "==> All experiments and analyses completed!"
echo "Results saved in:"
echo "  - output/"
