# Meta-Evaluation of LLM-Based Reference-Free Metrics for RAG

This repository contains the code and data used in the paper:

**Meta-Evaluation of LLM-Based Reference-Free Metrics for RAG**  
Xinyuan Cheng · LMU Munich

## 🧠 Overview

We evaluate the robustness and validity of various **reference-free evaluation metrics** for Retrieval-Augmented Generation (RAG), including:

- **Direct Prompting** (DP-Free & DP-Token)
- **G-Eval**
- **GPTScore**
- **RAGAS**

Our evaluation is conducted on the [WikiEval benchmark](https://huggingface.co/datasets/ExplodingGradients/WikiEval) and covers three quality dimensions:

- **Faithfulness**
- **Answer Relevance**
- **Context Relevance**

We assess both **pairwise accuracy** and **correlation with human judgments**, and analyze the effects of prompt format and LLM confidence.

## 📁 Project Structure

<pre>llm-metric-metaeval/
├── experiment/
│ ├── scripts/ # Scripts for running metric evaluations
│ ├── input/ # Input files: QCA data
│ ├── output/ # Metric outputs (CSV)
├── results_analysis/
│ ├── scripts/ # Scripts for accuracy, correlation, and robustness analysis
│ ├── output/ # Analysis Results (CSV)</pre>

## 🚀 Run Experiments
You can run individual metric evaluation scripts or reproduce all experiments at once.

### ▶️ Run a Single Metric
Navigate to experiment/scripts/ and run any of the metric scripts. Example:

<pre> python ragas_ff.py --input_csv ../input/ff.csv --output_csv ../output/ragas_ff_output.csv </pre>

### ▶️ Reproduce All Experiments

Alternatively, you can use the provided run_exp.sh script to reproduce all experiments in one go:

<pre> bash run_exp.sh </pre>

This script sequentially executes all evaluation pipelines and saves their outputs under experiment/output/.

## 📌 Naming Conventions
The files in follow the naming pattern:

- **ff** – Faithfulness

- **ar** – Answer Relevance

- **cr** – Context Relevance

- **prompt** - DP-free + DP-token + G-eval

- **variants** - including variants of prompt

For example:

[`ragas_cr.py`](/experiment/scipts/ragas_cr.py) evaluates context relevance using RAGAS

[`prompt_ar_variants.py`](/experiment/scipts/prompt_ar_variants.py) evaluates answer relevance using DP-free, DP-token and G-eval, including prompt variants
