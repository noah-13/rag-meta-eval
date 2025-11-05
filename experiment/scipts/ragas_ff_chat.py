import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
import re


class FaithfulnessEvaluator:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()


    def _generate(self, messages, max_new_tokens):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                do_sample=False
            )

        generated_len = outputs.shape[-1] - input_ids.shape[-1]
        truncated = generated_len >= max_new_tokens

        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True), truncated
    

    @staticmethod
    def safe_parse_json_list_with_error(output: str) -> tuple[list[str] | None, str]:
        def attempt_parse(candidate: str):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return parsed, "ok"
                elif isinstance(parsed, dict) and "statements" in parsed and isinstance(parsed["statements"], list):
                    return parsed["statements"], "ok_dict"
                return None, "not_a_list_or_dict"
            except json.JSONDecodeError:
                return None, "json_parse_error"

        # 1. Try direct parse
        parsed, status = attempt_parse(output)
        if status.startswith("ok"):
            return parsed, status

        # 2. Try extracting middle well-formed JSON structure
        for pattern in [r'(\[\s*.*?\])', r'(\{\s*.*?\})']:
            matches = re.finditer(pattern, output, re.DOTALL)
            for match in matches:
                candidate = match.group(1)

                # Try truncating at bracket balance point
                open_brackets = {'[': ']', '{': '}'}
                stack = []
                for i, ch in enumerate(candidate):
                    if ch in '[{':
                        stack.append(open_brackets[ch])
                    elif ch in ']}' and stack:
                        if ch == stack[-1]:
                            stack.pop()
                        else:
                            break
                    if not stack:
                        candidate = candidate[:i + 1]
                        break

                parsed, status = attempt_parse(candidate)
                if status.startswith("ok"):
                    return parsed, "extracted_middle"

        # 3. Last-ditch: recover list starting from first [ (e.g. '["a", "b", "c')
        match = re.search(r'\[\s*.*', output, re.DOTALL)
        if match:
            candidate = match.group(0)

            # Fix unmatched closing bracket
            if candidate.count('[') > candidate.count(']'):
                candidate += ']'

            # Fix common trailing comma
            candidate = re.sub(r',\s*]', ']', candidate)

            # Try loading normally
            parsed, status = attempt_parse(candidate)
            if status.startswith("ok"):
                return parsed, "partial_fix_list"

            # Fallback: extract complete strings from list using regex
            string_items = re.findall(r'"([^"]*)"', candidate)
            if string_items:
                return string_items, "partial_fix_list_truncated"

        return None, "parse_failed"


    def evaluate(self, question: str, answer: str, context: str) -> dict:
        result = {
            "raw_statements_output": "",
            "statements_parse_success": 0,
            "parsed_statements": [],
            "raw_judge_outputs": [],
            "verdicts": [],
            "faithfulness_score": np.nan,
            "parse_status": "",
            "statements_truncated": 0,
            "any_judge_truncated": 0
        }

        # You must return a valid JSON array. Do not include any additional text.
        # 1. Generate factual statements
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""Given a question and its answer, rewrite the answer into multiple self-contained factual statements without using any pronouns. Return only the list of statements in JSON format.
            
            Example:

            Question: What is GPT-4?
            Answer: GPT-4 is a powerful AI model developed by OpenAI. It can process images and texts.
            Output:
            [
            "GPT-4 is a powerful AI model developed by OpenAI.",
            "GPT-4 can process images.",
            "GPT-4 can process texts."
            ]

            Now rewrite the following:
            
            Question: {question}
            Answer: {answer}
            Output:"""}
        ]
        decoded, truncated = self._generate(messages, max_new_tokens=512)
        statements, parse_status = self.safe_parse_json_list_with_error(decoded)
        result["statements_truncated"] = int(truncated)
        result["raw_statements_output"] = decoded
        result["parse_status"] = parse_status
        if parse_status not in {"ok", "ok_dict", "partial_fix_list", "extracted_middle", "partial_fix_list_truncated"}:
            print(f"[WARN] Failed to parse statements: {parse_status}")
            return result
        result["statements_parse_success"]= 1
        result["parsed_statements"] = statements


        # 2. Judge each statement
        any_trunc = False
        for stmt in statements:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""For each given context and statement, determine whether the statement can be directly inferred from the context. Respond with 'Verdict: 1 or 0'.
                
                Context:{context}
                Statement:{stmt}"""}
            ]
            decoded_judge, truncated = self._generate(messages, max_new_tokens=64)
            if truncated:
                any_trunc = True
            result["raw_judge_outputs"].append(decoded_judge)

            verdict = 0
            for line in decoded_judge.split("\n"):
                if re.search(r"Verdict\s*:\s*1", line):
                    verdict = 1


            result["verdicts"].append(verdict)

        result["any_judge_truncated"] = int(any_trunc)

        if result["verdicts"]:
            result["faithfulness_score"] = sum(result["verdicts"]) / len(result["verdicts"])

        return result


def main(args):
    df = pd.read_csv(args.input_csv)
    evaluator = FaithfulnessEvaluator(args.model_id)

    new_columns = {
        "faithfulness_score": [],
        "raw_statements_output": [],
        "statements_parse_success": [],
        "parsed_statements": [],
        "raw_judge_outputs": [],
        "verdicts": [],
        "parse_status": [],
        "statements_truncated": [],
        "any_judge_truncated": []
    }

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q, a, c = row['question'], row['answer'], row['context']
        try:
            result = evaluator.evaluate(q, a, c)
        except Exception as e:
            print(f"Error processing row: {e}")
            # fallback: ensure every field is present and properly typed
            result = {
                k: np.nan if k == "faithfulness_score"
                else "" if k == "parse_status"
                else 0 if k in ["statements_truncated", "any_judge_truncated"]
                else [] for k in new_columns
            }

        for k in new_columns:
            new_columns[k].append(result[k])

    for col, values in new_columns.items():
        if isinstance(values[0], list):
            df[col] = [json.dumps(v, ensure_ascii=False) for v in values]
        else:
            df[col] = values

    df.to_csv(args.output_csv, index=False)
    print(f"Saved output to {args.output_csv}")
    truncated_df = df[
        (df["statements_truncated"] == 1) | (df["any_judge_truncated"] == 1)
    ]
    truncated_path = args.output_csv.replace(".csv", "_truncated.csv")
    truncated_df.to_csv(truncated_path, index=False)
    print(f"Saved truncated samples to {truncated_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output CSV with scores")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name or path")
    args = parser.parse_args()

    main(args)
