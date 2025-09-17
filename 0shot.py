#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, pathlib, re, sys
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# ================== 基本設定（可自行修改） ==================
MODEL = "gpt-5"
TEMPERATURE = 1
MAX_RETRIES = 3
SLEEP_S = 0.6  # 簡易速率控制（可視情況放大）

# 測試資料與輸出
TEST_PATH = r"C:\Users\lubob\Desktop\Rockling\dataset\Japanese_test.json"
OUT_PRED_JSONL = r"C:\Users\lubob\Desktop\Rockling\results\0shot_Japanese_predictions.jsonl"
# ===========================================================

# 載入 .env（若系統環境已有 OPENAI_API_KEY，此步驟也不影響）
load_dotenv()
client = OpenAI()  # 會讀 OPENAI_API_KEY

# ---- 嚴格輸出 Schema（新 SDK 可走） ----
JSON_SCHEMA = {
    "name": "PromiseEvalOutput",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "promise_status": {"type": "string", "enum": ["Yes", "No"]},
            "evidence_status": {"type": "string", "enum": ["Yes", "No"]},
            "evidence_quality": {
                "type": "string",
                "enum": ["Clear", "Not Clear", "Misleading", "N/A"]
            },
            "verification_timeline": {
                "type": "string",
                "enum": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"]
            }
        },
        "required": ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
    }
}

SYSTEM_PROMPT = """You are an ESG report compliance inspector.
Classify each input sample into four fields with STRICT JSON that matches the provided JSON schema.
- Read multilingual (JP/EN/ZH) text.
- Think briefly, but return ONLY the JSON object (no extra keys, no commentary).
Definition:
- promise_status: 'Yes' if there is a concrete or organizational-level commitment/stance; else 'No'.
- evidence_status: 'Yes' if there is verifiable action/evidence described; else 'No'.
- evidence_quality:
  * 'Clear' = evidence is specific, measurable, or auditable (dates, counts, rates, clear procedures).
  * 'Not Clear' = evidence exists but is vague or lacks specifics.
  * 'Misleading' = evidence contradicts targets or shows underperformance while being framed as positive.
  * 'N/A' = when evidence_status is 'No'.
- verification_timeline:
  * 'already' for practices/plans already in place or institutionalized.
  * 'within_2_years', 'between_2_and_5_years', 'more_than_5_years' based on explicit timing.
  * 'N/A' if no timeline can be inferred.
Return ONLY a single JSON object that matches the schema. Do not add any commentary or code fences."""

# ================== 工具函式 ==================
def build_messages_for(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """0-shot：只有 system + 當前樣本（無任何示範）"""
    current_user = {
        "role": "user",
        "content": json.dumps({
            "data": sample.get("data", ""),
            "URL": sample.get("URL", ""),
            "page_number": sample.get("page_number", ""),
            "ESG_type": sample.get("ESG_type", "")
        }, ensure_ascii=False)
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        current_user
    ]

def _extract_json(text: str) -> Dict[str, Any]:
    """從模型回覆擷取第一個 {...} JSON 片段"""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))

def _normalize_pred(d: Dict[str, Any]) -> Dict[str, str]:
    """欄位齊全 + 枚舉正規化 + 規則兜底"""
    def norm(x): 
        return str(x).strip()

    p = norm(d.get("promise_status", "No"))
    e = norm(d.get("evidence_status", "No"))
    q = norm(d.get("evidence_quality", "N/A"))
    t = norm(d.get("verification_timeline", "N/A"))

    q_map = {"clear":"Clear","not clear":"Not Clear","misleading":"Misleading","n/a":"N/A"}
    t_map = {
        "already":"already",
        "within_2_years":"within_2_years",
        "between_2_and_5_years":"between_2_and_5_years",
        "more_than_5_years":"more_than_5_years",
        "n/a":"N/A"
    }
    q = q_map.get(q.lower(), q)
    t = t_map.get(t.lower(), t)

    if e == "No":
        q = "N/A"

    if p not in ["Yes","No"]: p = "No"
    if e not in ["Yes","No"]: e = "No"
    if q not in ["Clear","Not Clear","Misleading","N/A"]: q = "N/A"
    if t not in ["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]: t = "N/A"

    return {
        "promise_status": p,
        "evidence_status": e,
        "evidence_quality": q,
        "verification_timeline": t
    }

def call_gpt(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    先嘗試 Responses API + json_schema（新 SDK）
    若出現不支援 'response_format' 的 TypeError：
      A) Responses API（不帶 schema）→ 擷取/正規化
      B) 再失敗 → Chat Completions → 擷取/正規化
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # ---- 新路徑：responses + response_format（若 SDK 支援）----
            resp = client.responses.create(
                model=MODEL,
                temperature=TEMPERATURE,  # 若報 unsupported，就刪掉這一行
                response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
                input=[{"role": m["role"], "content": m["content"]} for m in messages],
            )
            return json.loads(resp.output_text)

        except TypeError as e:
            if "response_format" in str(e):
                # ---- Fallback A：responses 不帶 schema ----
                try:
                    resp = client.responses.create(
                        model=MODEL,
                        temperature=TEMPERATURE,  # 若報 unsupported，就刪掉
                        input=[{"role": m["role"], "content": m["content"]} for m in messages],
                    )
                    text = resp.output_text
                    return _normalize_pred(_extract_json(text))
                except Exception:
                    # ---- Fallback B：chat.completions ----
                    cc = client.chat.completions.create(
                        model=MODEL,
                        temperature=TEMPERATURE,
                        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                    )
                    text = cc.choices[0].message.content
                    return _normalize_pred(_extract_json(text))
            else:
                if attempt >= MAX_RETRIES:
                    raise
        except Exception as e:
            if attempt >= MAX_RETRIES:
                raise
        time.sleep(SLEEP_S * attempt)
    return {}

# ================== 主流程 ==================
def load_samples(path: str) -> List[Dict[str, Any]]:
    """支援 .json（list[dict] 或單一 dict）與 .jsonl；皆用 utf-8-sig（支援 BOM）"""
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

def main():
    print(f"[INFO] MODEL={MODEL}, TEST_PATH={TEST_PATH}")
    samples = load_samples(TEST_PATH)
    print(f"[INFO] Loaded {len(samples)} samples")

    out_dir = pathlib.Path(OUT_PRED_JSONL).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail = 0, 0
    with open(OUT_PRED_JSONL, "w", encoding="utf-8") as wf:
        for idx, sample in enumerate(samples):
            messages = build_messages_for(sample)
            try:
                pred = call_gpt(messages)
                pred = _normalize_pred(pred)
                out = {
                    "idx": idx,
                    "input": {
                        "data": sample.get("data", ""),
                        "URL": sample.get("URL", ""),
                        "page_number": sample.get("page_number", ""),
                        "ESG_type": sample.get("ESG_type", "")
                    },
                    "pred": pred
                }
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                wf.write(json.dumps({"idx": idx, "error": str(e)}, ensure_ascii=False) + "\n")
                n_fail += 1

            if (idx + 1) % 10 == 0:
                print(f"[PROGRESS] {idx + 1}/{len(samples)} processed ...")
            time.sleep(SLEEP_S)

    print(f"[DONE] success={n_ok}, fail={n_fail}")
    print(f"[SAVED] {OUT_PRED_JSONL}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] by user", file=sys.stderr)
        sys.exit(130)
