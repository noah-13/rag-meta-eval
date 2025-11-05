import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer


class AnswerRelevanceEvaluator:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda", strictness=3):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        self.strictness = strictness
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def _generate(self, messages, max_new_tokens=256):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            )

        generated_len = outputs.shape[-1] - input_ids.shape[-1]
        truncated = generated_len >= max_new_tokens
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True), truncated

    @staticmethod
    def safe_parse_response(output: str) -> tuple[str, int, str]:
        try:
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if "question" in parsed and "noncommittal" in parsed:
                    return parsed["question"], int(parsed["noncommittal"]), "ok"
        except Exception as e:
            pass
        return "", 1, "parse_failed"

    def evaluate(self, question: str, answer: str) -> dict:
        result = {
            "answer_relevancy_score": np.nan,
            "raw_outputs": [],
            "gen_questions": [],
            "noncommittal_flags": [],
            "output_parse_success": 0,
            "output_parse_status": [],
            "any_truncated": 0
        }

        any_trunc = False
        gen_questions = []
        flags = []
        raw_outputs = []
        statuses = []

        for _ in range(self.strictness):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""Given an answer, generate a question that the answer likely responds to, and classify whether the answer is noncommittal (e.g., vague, evasive, or ambiguous).
                 
                 Return your output as a JSON object with keys:
                - `"question"`: the inferred question
                - `"noncommittal"`: 1 if the answer is noncommittal, otherwise 0

                Examples:

                Answer: I think it might work, but I can't guarantee it.  
                Output:  
                {{
                "question": "Will this method work?",
                "noncommittal": 1
                }}

                Answer: The capital of France is Paris.
                Output:
                {{
                "question": "What is the capital of France?",
                "noncommittal": 0
                }}
                
                Now do the same for the following:
                Answer:
                {answer}

                Output:"""}
            ]
            decoded, truncated = self._generate(messages, max_new_tokens=256)
            q, flag, status = self.safe_parse_response(decoded)

            raw_outputs.append(decoded)
            gen_questions.append(q)
            flags.append(flag)
            statuses.append(status)

            if truncated:
                any_trunc = True

        result["raw_outputs"] = raw_outputs
        result["gen_questions"] = gen_questions
        result["noncommittal_flags"] = flags
        result["output_parse_status"] = statuses
        result["any_truncated"] = int(any_trunc)
        result["output_parse_success"] = int(all(s == "ok" for s in statuses))

        if all(q == "" for q in gen_questions):
            return result

        question_vec = self.embedder.encode([question], normalize_embeddings=True)
        gen_vecs = self.embedder.encode(gen_questions, normalize_embeddings=True)
        sims = np.dot(gen_vecs, question_vec.T).reshape(-1)
        result["answer_relevancy_score"] = float(sims.mean() * (1 - int(any(flags))))

        return result


def main(args):
    df = pd.read_csv(args.input_csv)
    evaluator = AnswerRelevanceEvaluator(args.model_id, args.strictness)

    new_columns = {
        "answer_relevancy_score": [],
        "raw_outputs": [],
        "gen_questions": [],
        "noncommittal_flags": [],
        "output_parse_success": [],
        "output_parse_status": [],
        "any_truncated": []
    }

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            q, a = row["question"], row["answer"]
            result = evaluator.evaluate(q, a)
        except Exception as e:
            print(f"[Error] {e}")
            result = {k: np.nan if k == "answer_relevancy_score" else [] if isinstance(v, list) else 0 for k, v in new_columns.items()}

        for k in new_columns:
            new_columns[k].append(result[k])

    for k, values in new_columns.items():
        if isinstance(values[0], list):
            df[k] = [json.dumps(v, ensure_ascii=False) for v in values]
        else:
            df[k] = values

    df.to_csv(args.output_csv, index=False)
    print(f"Saved result to {args.output_csv}")

    truncated_df = df[df["any_truncated"] == 1]
    truncated_path = args.output_csv.replace(".csv", "_truncated.csv")
    truncated_df.to_csv(truncated_path, index=False)
    print(f"Saved truncated samples to {truncated_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output CSV with scores")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name or path")
    parser.add_argument("--strictness", type=int, default=3, help="Number of questions to generate per answer")
    args = parser.parse_args()

    main(args)
