"""
This script evaluates prompt variants for LLM scoring across three dimensions 
(Answer Relevance, Faithfulness, Context Relevance). It compares model scores 
(explicit_score, weighted_score, top1_score) with human-annotated answers to 
find the best-performing mode (imperative or interrogative) and score range 
for each dimension. Results are saved as summary CSV files.
"""
import pandas as pd

# Dictionary to map dimensions to corresponding sheet names
map_dim_to_sheet = {
    "ar": "Answer Relevance",
    "ff": "Faithfulness",
    "cr": "Context Relevance"
}

score_columns = ["explicit_score", "weighted_score", "top1_score"]
modes = ["imperative", "interrogative"]

for dim in ["ar", "ff", "cr"]:
    # Load the data for the current dimension
    df = pd.read_csv(f"../../experiment/output/{dim}_prompt_variants.csv")
    final_df = pd.read_excel("final_annotated_with_final_answer.xlsx", sheet_name=map_dim_to_sheet[dim])

    results = {}
    table_rows = []  

    # Process each score column
    for score_col in score_columns:
        results[score_col] = {}

        # For each score column, find the best mode and score range combination
        best_accuracy = -1
        best_mode = None
        best_range_str = None

        for mode in modes:
            sub_df = df[df["mode"] == mode].sort_index().reset_index(drop=True)

            # Ensure that the number of rows is a multiple of 3
            if len(sub_df) % 3 != 0:
                print(f"[WARNING] Row count for mode={mode} is not a multiple of 3 ({len(sub_df)} rows)")

            num_samples = len(sub_df) // 3 // 2  # Half the number of variants
            results[score_col][mode] = {}

            for i in range(num_samples):
                start_a = i * 3
                start_b = (i + num_samples) * 3

                rows_a = sub_df.iloc[start_a:start_a + 3]
                rows_b = sub_df.iloc[start_b:start_b + 3]

                for idx in range(3):  # 3 variants per sample
                    row_a = rows_a.iloc[idx]
                    row_b = rows_b.iloc[idx]
                    range_str = row_a["score_range"]

                    if row_b["score_range"] != range_str:
                        print(f"[ERROR] score_range mismatch at sample={i}, mode={mode}, variant={idx}")
                        continue

                    try:
                        low, high = map(float, range_str.split("-"))
                        score_a = float(row_a[score_col])
                        score_b = float(row_b[score_col])
                    except Exception as e:
                        print(f"[ERROR] Failed to parse score/sample at i={i}, col={score_col}: {e}")
                        continue

                    if range_str not in results[score_col][mode]:
                        results[score_col][mode][range_str] = {
                            "correct": 0,
                            "total": 0,
                            "out_of_range": 0,
                        }

                    out_of_range = not (low <= score_a <= high) or not (low <= score_b <= high)
                    if out_of_range:
                        results[score_col][mode][range_str]["out_of_range"] += 1

                    model_choice = 1 if score_a > score_b else 2
                    human_label = final_df.iloc[i]["Final_answer"]

                    if model_choice == human_label:
                        results[score_col][mode][range_str]["correct"] += 1
                    results[score_col][mode][range_str]["total"] += 1

            # Now for this mode, find the best score range
            for range_str, stats in results[score_col][mode].items():
                accuracy = stats["correct"] / stats["total"] * 100
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_mode = mode
                    best_range_str = range_str

        # Store the best setting for the current score type
        table_rows.append({
            "dimension": dim,
            "score_type": score_col,
            "best_mode": best_mode,
            "best_score_range": best_range_str,
            "best_accuracy (%)": round(best_accuracy, 2)
        })

    # Create a table with the best settings for each dimension and score type
    result_table = pd.DataFrame(table_rows)
    result_table.to_csv(f"{dim}_best_prompt_results_summary.csv", index=False)

    print(f"\n=== Best Settings Summary for {dim} ===")
    print(result_table)
