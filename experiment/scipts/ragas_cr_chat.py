import argparse
import json
import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util


def safe_parse_json_list_with_error(output: str) -> tuple[list[str] | None, str]:
    def attempt_parse(candidate: str):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed, "ok"
            return None, "not_a_list"
        except json.JSONDecodeError:
            return None, "json_parse_error"

    parsed, status = attempt_parse(output)
    if status.startswith("ok"):
        return parsed, status

    for pattern in [r'(\[\s*.*?\])']:
        matches = re.finditer(pattern, output, re.DOTALL)
        for match in matches:
            candidate = match.group(1)
            stack = []
            for i, ch in enumerate(candidate):
                if ch == "[": stack.append("]")
                elif ch == "]" and stack: stack.pop()
                if not stack:
                    candidate = candidate[:i + 1]
                    break
            parsed, status = attempt_parse(candidate)
            if status.startswith("ok"):
                return parsed, "extracted_middle"

    match = re.search(r'\[\s*.*', output, re.DOTALL)
    if match:
        candidate = match.group(0)
        if candidate.count('[') > candidate.count(']'):
            candidate += ']'
        candidate = re.sub(r',\s*]', ']', candidate)
        parsed, status = attempt_parse(candidate)
        if status.startswith("ok"):
            return parsed, "partial_fix_list"
        string_items = re.findall(r'"([^"]*)"', candidate)
        if string_items:
            return string_items, "partial_fix_list_truncated"

    return None, "parse_failed"


class EmbeddingContextRelevance:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

        self.spacy_nlp = English()
        self.spacy_nlp.add_pipe("sentencizer", config={"overwrite": True})

        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
        self.device = torch.device(device)        
        self.cutoff = 0.5

    def sent_tokenize(self, text: str) -> list[str]:
        doc = self.spacy_nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _generate(self, question: str, context: str) -> tuple[str, bool]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""Given a question and a context, extract only the sentences from the context that help answer the question.

        The extracted sentences must:
        - Be copied exactly from the original context.
        - Be relevant to answering the question.
        - Be returned as a JSON list of strings.
        - Do not paraphrase or modify any sentences.
        - If no relevant sentence is found, return exactly: ["Insufficient Information"]

        Question: {question}
        Context:
        {context}

        Output:"""}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                do_sample=False
            )

        generated_len = outputs.shape[-1] - input_ids.shape[-1]
        truncated = generated_len >= 256
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True), truncated

    def evaluate(self, question: str, context: str) -> dict:
        sents = self.sent_tokenize(context)
        result = {
            "context_relevance_score": np.nan,
            "total_sentences": len(sents),
            "extracted_sentences": [],
            "raw_output": "",
            "parse_status": "",
            "statements_truncated": 0
        }

        if not sents or question.strip() == "":
            result["context_relevance_score"] = 0.0
            return result

        output, truncated = self._generate(question, context)
        result["raw_output"] = output
        result["statements_truncated"] = int(truncated)

        parsed_list, status = safe_parse_json_list_with_error(output)
        result["parse_status"] = status

        if parsed_list is None or parsed_list == ["Insufficient Information"]:
            result["context_relevance_score"] = 0.0
            return result

        try:
            question_emb = self.embedder.encode(question, convert_to_tensor=True, device=self.device)
            parsed_embs = self.embedder.encode(parsed_list, convert_to_tensor=True, device=self.device)
            sim_scores = util.cos_sim(question_emb, parsed_embs)[0]
            selected = [parsed_list[i] for i in range(len(parsed_list)) if sim_scores[i] >= self.cutoff]

            ctx_embs = self.embedder.encode(sents, convert_to_tensor=True, device=self.device)
            extracted = []
            for s in selected:
                s_emb = self.embedder.encode(s, convert_to_tensor=True, device=self.device)
                cos_sim = util.cos_sim(s_emb, ctx_embs)[0]
                best_idx = int(torch.argmax(cos_sim).item())
                extracted.append(sents[best_idx])
        except Exception as e:
            print(f"[Embedding error] {e}")
            extracted = []

        result["extracted_sentences"] = extracted
        result["context_relevance_score"] = len(extracted) / len(sents) if sents else 0.0
        return result


def main(args):
    df = pd.read_csv(args.input_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = EmbeddingContextRelevance(args.model_id, device=device)

    new_columns = {
        "context_relevance_score": [],
        "total_sentences": [],
        "extracted_sentences": [],
        "raw_output": [],
        "parse_status": [],
        "statements_truncated": []
    }

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            q = str(row["question"]).strip()
            ctx = str(row["context"]).strip()
            result = evaluator.evaluate(q, ctx)
        except Exception as e:
            print(f"[Error] {e}")
            result = {
                "context_relevance_score": np.nan,
                "total_sentences": 0,
                "extracted_sentences": [],
                "raw_output": "ERROR",
                "parse_status": "exception",
                "statements_truncated": 0
            }

        for k in new_columns:
            v = result[k]
            new_columns[k].append(json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)

    for k, values in new_columns.items():
        df[k] = values

    df.to_csv(args.output_csv, index=False)
    print(f"[✓] Saved to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()
    main(args)
