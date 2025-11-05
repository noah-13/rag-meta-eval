import pandas as pd
from scipy.stats import spearmanr, kendalltau

DIMS = ["ar", "ff", "cr"]
MODES = ["imperative", "interrogative"]
SCORE_COLUMNS = ["explicit_score", "weighted_score", "top1_score"]

DIM_TO_SHEET = {
    "ar": "Answer Relevance",
    "ff": "Faithfulness",
    "cr": "Context Relevance"
}
DIM_TO_SCORE_COL = {
    "ar": "answer_relevance",
    "ff": "faithfulness",
    "cr": "context_relevance"
}

def evaluate_accuracy_by_prompt_variant(dim):
    df = pd.read_csv(f"../../experiment/output/{dim}_prompt_variants.csv")
    final_df = pd.read_excel("final_annotated_with_final_answer.xlsx", sheet_name=DIM_TO_SHEET[dim])

    results = {}
    best_configs = {}

    for score_col in SCORE_COLUMNS:
        results[score_col] = {}
        best_accuracy = 0
        best_mode = ""
        best_range = ""
        best_scores = []

        for mode in MODES:
            sub_df = df[df["mode"] == mode].sort_index().reset_index(drop=True)
            num_samples = len(sub_df) // 6
            results[score_col][mode] = {}

            for i in range(num_samples):
                start_a, start_b = i * 3, (i + num_samples) * 3
                rows_a = sub_df.iloc[start_a:start_a + 3]
                rows_b = sub_df.iloc[start_b:start_b + 3]

                for j in range(3):
                    row_a = rows_a.iloc[j]
                    row_b = rows_b.iloc[j]
                    range_str = row_a["score_range"]

                    if row_b["score_range"] != range_str:
                        continue
                    try:
                        low, high = map(float, range_str.split("-"))
                        score_a, score_b = float(row_a[score_col]), float(row_b[score_col])
                    except:
                        continue

                    stats = results[score_col][mode].setdefault(range_str, {"correct": 0, "total": 0})
                    model_choice = 1 if score_a > score_b else 2
                    human_label = final_df.iloc[i]["Final_answer"]

                    if model_choice == human_label:
                        stats["correct"] += 1
                    stats["total"] += 1

            # 选出 accuracy 最好的设定
            for range_str, stats in results[score_col][mode].items():
                acc = stats["correct"] / stats["total"] * 100
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_mode = mode
                    best_range = range_str
                    best_scores = sub_df[(sub_df["mode"] == mode) & (sub_df["score_range"] == range_str)][score_col].values

        best_configs[score_col] = {
            "best_accuracy": best_accuracy,
            "best_mode": best_mode,
            "best_range_str": best_range,
            "best_scores": best_scores
        }

    return best_configs
def evaluate_new_metric_best_setting(dim):
    final_df = pd.read_excel("final_annotated_with_final_answer.xlsx", sheet_name=DIM_TO_SHEET[dim])
    best_accuracy = 0
    best_mode = ""
    best_scores = []

    for mode in ["", "chat_"]:
        df = pd.read_csv(f"{dim}_ragas_{mode}output.csv")
        df["pair_id"] = df.index % 50

        correct = 0
        total = 0
        scores = []

        for i in range(50):
            row_a = df.iloc[i]
            row_b = df.iloc[i + 50]

            score_a = row_a[f"{DIM_TO_SCORE_COL[dim]}_score"]
            score_b = row_b[f"{DIM_TO_SCORE_COL[dim]}_score"]

            model_choice = 1 if score_a > score_b else 2
            human_label = final_df.iloc[i]["Final_answer"]

            scores.append(score_a)
            scores.append(score_b)

            if model_choice == human_label:
                correct += 1
            total += 1

        accuracy = correct / total * 100
        print(f"[{DIM_TO_SHEET[dim]}] Mode: {mode or 'plain'} Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mode = mode or "plain"
            best_scores = scores

    print(f"✅ Best new_metric mode for [{DIM_TO_SHEET[dim]}]: {best_mode} with {best_accuracy:.2f}%")
    return best_scores

def compute_correlations(dim, best_configs, new_metric_scores):
    score_data = pd.DataFrame({
        'explicit_score': best_configs["explicit_score"]["best_scores"],
        'weighted_score': best_configs["weighted_score"]["best_scores"],
        'top1_score': best_configs["top1_score"]["best_scores"],
        'new_metric': new_metric_scores
    })

    if score_data.nunique().min() <= 1:
        print(f"[WARNING] Skipping correlation for {dim}: insufficient variance")
        return

    print(f"\n=== Correlation Analysis for {DIM_TO_SHEET[dim]} ===")
    print("Spearman Correlation Matrix:")
    print(score_data.corr(method="spearman"))

    print("\nKendall's Tau:")
    for col1 in score_data.columns:
        for col2 in score_data.columns:
            if col1 < col2:  # Avoid duplicate pairs
                tau, _ = kendalltau(score_data[col1], score_data[col2])
                print(f"{col1} vs {col2}: {tau:.3f}")

if __name__ == "__main__":
    for dim in DIMS:
        print(f"\n=== Evaluating Dimension: {DIM_TO_SHEET[dim]} ===")
        
        best_config = evaluate_accuracy_by_prompt_variant(dim)
        new_metric_scores = evaluate_new_metric_best_setting(dim)  
        compute_correlations(dim, best_config, new_metric_scores)
