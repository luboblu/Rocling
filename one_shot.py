#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, pathlib, re, sys
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# ================== 基本設定 ==================
MODEL = "gpt-5"
TEMPERATURE = 1          # 若 Responses API 報 unsupported，就移除此參數
MAX_RETRIES = 3
SLEEP_S = 0.6

# 路徑
TEST_PATH = r"C:\Users\lubob\Desktop\Rockling\dataset\Japanese_test.json"
EXAMPLES_DIR = r"C:\Users\lubob\Desktop\Rockling\examples"  # 你的 1/3/5-shot 範例放這裡
SHOT = 1  # ← 設 1/3/5，就會讀 <task>_shot{SHOT}.json
OUT_DIR = r"C:\Users\lubob\Desktop\Rockling\results\1shot"

# ===========================================================
load_dotenv()
client = OpenAI()

# ───────────────────────────────────────────────────────────
# 任務設定
# ───────────────────────────────────────────────────────────
def get_task_config(task: str) -> Dict[str, Any]:
    if task == "promise_status":
        enums = ["Yes", "No"]
        guidance = "- promise_status: 'Yes' if there is a concrete or organizational-level commitment/stance; else 'No'."
    elif task == "evidence_status":
        enums = ["Yes", "No"]
        guidance = "- evidence_status: 'Yes' if there is verifiable action/evidence described; else 'No'."
    elif task == "evidence_quality":
        enums = ["Clear", "Not Clear", "Misleading", "N/A"]
        guidance = (
            "- evidence_quality:\n"
            "  * 'Clear' = evidence is specific, measurable, or auditable (dates, counts, rates, clear procedures).\n"
            "  * 'Not Clear' = evidence exists but is vague or lacks specifics.\n"
            "  * 'Misleading' = evidence contradicts targets or shows underperformance while being framed as positive.\n"
            "  * 'N/A' = when there is no evidence."
        )
    elif task == "verification_timeline":
        enums = ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"]
        guidance = (
            "- verification_timeline:\n"
            "  * 'already' for practices/plans already in place or institutionalized.\n"
            "  * 'within_2_years', 'between_2_and_5_years', 'more_than_5_years' based on explicit timing.\n"
            "  * 'N/A' if no timeline can be inferred."
        )
    else:
        raise ValueError(f"Unsupported TASK: {task}")

    json_schema = {
        "name": f"{task}_Output",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {task: {"type": "string", "enum": enums}},
            "required": [task],
        },
    }

    system_prompt = f"""You are an ESG report compliance inspector.
Classify the INPUT into a single field: {task}.
Return STRICT JSON that matches the provided JSON schema. No extra keys. No commentary.
- Read multilingual (JP/EN/ZH) text.
- Think briefly, but return ONLY the JSON object.

Definitions:
{guidance}

Return ONLY one JSON object with the single key "{task}". Do not add any commentary or code fences."""
    return {"enums": enums, "schema": json_schema, "system_prompt": system_prompt}

# ───────────────────────────────────────────────────────────
# Few-shot 載入 & Message 建構
# ───────────────────────────────────────────────────────────
def load_shot_examples(task: str, shot: int) -> List[Dict[str, Any]]:
    """讀 examples/<task>_shot{shot}.json；只接受 {data, <task>} 結構"""
    path = pathlib.Path(EXAMPLES_DIR) / f"{task}_shot{shot}.json"
    if not path.exists():
        raise FileNotFoundError(f"Few-shot file not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        ex = json.load(f)
    if not isinstance(ex, list):
        raise ValueError(f"Few-shot file must be a list: {path}")
    return ex

def build_messages_with_fewshot(task: str, system_prompt: str, examples: List[Dict[str, Any]], sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Messages: system → (few-shot user/assistant pairs) → current user
    - few-shot user:   只放 example['data']
    - few-shot assist: {"<task>": "<label>"}
    - current user:    只放 sample['data']
    """
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for ex in examples:
        ex_data = ex.get("data", "")
        ex_label = ex.get(task, None)
        if ex_label is None:
            # 容錯：若 key 不存在就跳過
            continue
        msgs.append({"role": "user", "content": ex_data})
        msgs.append({"role": "assistant", "content": json.dumps({task: ex_label}, ensure_ascii=False)})
    # 當前樣本
    msgs.append({"role": "user", "content": sample.get("data", "")})
    return msgs

# ───────────────────────────────────────────────────────────
# 解析/正規化/呼叫 API
# ───────────────────────────────────────────────────────────
def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))

def _normalize_pred_single(task: str, enums: List[str], d: Dict[str, Any]) -> Dict[str, str]:
    def norm(x): return str(x).strip()
    val = norm(d.get(task, ""))

    if task == "evidence_quality":
        mapping = {"clear": "Clear", "not clear": "Not Clear", "misleading": "Misleading", "n/a": "N/A"}
        val = mapping.get(val.lower(), val)
    if task == "verification_timeline":
        mapping = {
            "already": "already",
            "within_2_years": "within_2_years",
            "between_2_and_5_years": "between_2_and_5_years",
            "more_than_5_years": "more_than_5_years",
            "n/a": "N/A",
        }
        val = mapping.get(val.lower(), val)

    if val not in enums:
        val = "N/A" if "N/A" in enums else enums[-1]
    return {task: val}

def call_gpt(messages: List[Dict[str, str]], schema: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                temperature=TEMPERATURE,  # 若報 unsupported 就移除
                response_format={"type": "json_schema", "json_schema": schema},
                input=[{"role": m["role"], "content": m["content"]} for m in messages],
            )
            return json.loads(resp.output_text)
        except TypeError:
            try:
                resp = client.responses.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    input=[{"role": m["role"], "content": m["content"]} for m in messages],
                )
                return _extract_json(resp.output_text)
            except Exception:
                cc = client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                )
                return _extract_json(cc.choices[0].message.content)
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
        time.sleep(SLEEP_S * attempt)
    return {}

# ───────────────────────────────────────────────────────────
# I/O
# ───────────────────────────────────────────────────────────
def load_samples(path: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TEST_PATH not found: {path}")
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
    else:
        with open(p, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                samples = obj
            elif isinstance(obj, dict) and "data" in obj:
                samples = [obj]
            else:
                raise ValueError("Unsupported JSON structure. Expect list[dict] or jsonl.")
    return samples

def run_one_task(task: str, samples: List[Dict[str, Any]], out_dir: str, shot: int):
    cfg = get_task_config(task)
    system_prompt, schema, enums = cfg["system_prompt"], cfg["schema"], cfg["enums"]

    # 載入 few-shot
    examples = load_shot_examples(task, shot)

    out_path = pathlib.Path(out_dir) / f"1shot_{task}_predictions.jsonl" if shot == 1 \
               else pathlib.Path(out_dir) / f"{shot}shot_{task}_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail = 0, 0
    with open(out_path, "w", encoding="utf-8") as wf:
        for idx, sample in enumerate(samples):
            messages = build_messages_with_fewshot(task, system_prompt, examples, sample)
            try:
                raw_pred = call_gpt(messages, schema)
                pred = _normalize_pred_single(task, enums, raw_pred)
                out = {"idx": idx, "input": {"data": sample.get("data", "")}, "pred": pred}
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                wf.write(json.dumps({"idx": idx, "error": str(e)}, ensure_ascii=False) + "\n")
                n_fail += 1

            if (idx + 1) % 20 == 0:
                print(f"[{task} | {shot}-shot] {idx + 1}/{len(samples)} processed ...")
            time.sleep(SLEEP_S)

    print(f"[{task} | {shot}-shot] DONE success={n_ok}, fail={n_fail} -> {out_path}")

# ───────────────────────────────────────────────────────────
# 主流程
# ───────────────────────────────────────────────────────────
def main():
    print(f"[INFO] MODEL={MODEL}, TEST_PATH={TEST_PATH}, SHOT={SHOT}, EXAMPLES_DIR={EXAMPLES_DIR}")
    samples = load_samples(TEST_PATH)
    print(f"[INFO] Loaded {len(samples)} samples")

    tasks = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
    for task in tasks:
        run_one_task(task, samples, OUT_DIR, SHOT)

    print("[ALL TASKS DONE]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] by user", file=sys.stderr)
        sys.exit(130)
