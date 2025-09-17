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

# 測試資料與輸出（預設為你的 Windows 路徑；如用上傳檔改成 /mnt/data/...）
TEST_PATH = r"C:\Users\lubob\Desktop\Rockling\dataset\Japanese_test.json"
OUT_PRED_JSONL = r"C:\Users\lubob\Desktop\Rockling\results\Japanese_predictions.jsonl"
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

FEWSHOTS = [
    ({"data": "ガバナンス 当社は取締役会において、気候関連リスクと機会の認知および対応策の検討、目標の進捗状況のモニタリングと監督を通じて気候変動への対応を強化し、経営戦略に取り込んでいくことを意思決定しています。社長執行役員は、カーボンニュートラル・環境経営推進会議の議長を務め、気候変動への対応を含む環境活動の統括を担っています。気候関連問題への対応については、環境マネジメントの枠組みにおいて進捗状況を集約しカーボンニュートラル・環境経営推進会議へ報告した後、サステナビリティ重要課題（マテリアリティ）の一つとして、サステナビリティ推進会議を経て、毎年取締役会に報告することとしています。なお、サステナビリティ担当役員の諮問機関として外部有識者で構成されるアドバイザリーボードを設置し、サステナビリティ潮流やステークホルダー目線での助言を受け、サステナビリティ推進会議の審議へ織り込んでおります。", 
      "URL": "https://www.tohoku-epco.co.jp/ir/report/integrated/pdf/tohoku_sustainabilityreport2023_jp_14-34.pdf", 
      "page_number": "8", "ESG_type": "E"}, 
     {"promise_status": "Yes", "evidence_status": "Yes", "evidence_quality": "Not Clear", "verification_timeline": "already"}),

    ({"data": "移行リスクの分析結果 移行リスクの増大が想定される1.5℃シナリオにおいては、短中長期いずれの期間においても政治・政策的リスク（カーボンプライシング導入等）または経済・市場的リスク（従来型電源の市場価格低下等）が想定され、これにより、炭素排出コストの負担がより大きくなることで、石炭などの化石燃料由来の火力発電の競争力が低下するリスクがあります。中長期においては、熱効率の改善・電気自動車用蓄電池コストの低下など脱炭素技術が進展することが見込まれます。これに伴うリスクとしては、新規設備投資額の増加や省エネ技術が進展することによる電力需要の減少が挙げられます。一方で、1.5℃シナリオにおいては、脱炭素製品・サービスの市場シェアの拡大や電化率の上昇などが当社にとっての事業機会と想定されます。", 
      "URL": "https://www.tohoku-epco.co.jp/ir/report/integrated/pdf/tohoku_sustainabilityreport2023_jp_14-34.pdf", 
      "page_number": "10", "ESG_type": "E"}, 
     {"promise_status": "No", "evidence_status": "N/A", "evidence_quality": "N/A", "verification_timeline": "N/A"}),

    ({"data": "物理リスクの分析結果 物理的リスクの大きい4°Cシナリオにおいては、気候変動の影響が顕著となり、気象災害の激甚化・降水パターンの変化が想定されます。急性リスクとして気象災害の頻発化・激甚化による当社設備被害・供給支障の増加が想定されるため電力レジリエンスの重要性が高まります。また、慢性リスクとして降水パターンの変化による水力発電等への影響が想定されます。当社は、頻発化・激甚化する気象災害に備え、設備の強靱化と復旧対応力を高め、電力レジリエンスの向上を図っていきます。", 
      "URL": "https://www.tohoku-epco.co.jp/ir/report/integrated/pdf/tohoku_sustainabilityreport2023_jp_14-34.pdf", 
      "page_number": "11", "ESG_type": "E"}, 
     {"promise_status": "Yes", "evidence_status": "No", "evidence_quality": "N/A", "verification_timeline": "between_2_and_5_years"}),

    ({"data": "丸紅行動憲章 丸紅は、公正なる競争を通じて利潤を追求する企業体であると同時に、世界経済の発展に貢献し、社会にとって価値のある企業であることを目指します。これを踏まえて、以下の6項目を行動の基本原則とします。(a)公正、透明な企業活動の徹底法律を遵守し、公正な取引を励行する。内外の政治や行政との健全な関係を保ち、自由競争による営業活動を徹底する。反社会的な活動や勢力に対しては毅然とした態度で臨む。(b)グローバル・ネットワーク企業としての発展各国、各地域の文化を尊重し、企業活動を通じて地域経済の繁栄に貢献していく。グローバルに理解が得られる経営システムを通じて、各地域社会と調和のとれた発展を目指す。(c)新しい価値の創造市場や産業の変化に対応するだけでなく、変化を自ら創造し、市場や顧客に対して新しい商品やサービスを提供していく。既存の常識や枠組みにとらわれることなく、常に新たな可能性にチャレンジする。(d)個性の尊重と独創性の発揮一人一人の個性を尊重し、独創性が存分に発揮できる、自由で活力のある企業風土を醸成する。自己管理の下、自らが課題達成に向けて主体的に行動する。(e)コーポレート・ガバナンスの推進株主や社会に対して積極的な情報開示を行い、経営の透明度を高める。経営の改善等に係る提案を尊重し、株主や社会に対してオープンな経営を目指す。(f)社会貢献や地球環境への積極的な関与国際社会における企業市民としての責任を自覚し、積極的な社会貢献活動を行う。", 
      "URL": "https://marubeni.disclosure.site/ja/sustainability/pdf/report/sdr2024_jp_06.pdf", 
      "page_number": "2", "ESG_type": "G"}, 
     {"promise_status": "Yes", "evidence_status": "Yes", "evidence_quality": "Clear", "verification_timeline": "more_than_5_years"}),

    ({"data": "「伊藤忠ユニダス株式会社」での取組み 伊藤忠ユニダス（株）は、障がい者と健常者が共に支え合いながら一体となってクリーニング、印刷、書類電子化、写真サービス、メール集配、ランドリー・清掃等の事業を展開しています。2015年11月には、事業の拡大に加え、障がいのある従業員にとってより働きやすい職場環境を実現するため、ユニバーサルデザインで、最新の機器を有する横浜市都筑区の新社屋へ移転しました。現在、横浜市都筑区の本社に加え、青山事業所、日吉事業所、及びクリーニングサービスの店舗「よつ葉クリーニング」（横浜市旭区）の4拠点で事業を展開しています。今後も引続き、障がいのある人々の社会参画を積極的に促し、仕事を通して社会に価値を提供することで、働く喜びを実感できる職場環境を目指して参ります。", 
      "URL": "https://www.itochu.co.jp/ja/csr/pdf/23fullj100-168.pdf", 
      "page_number": "13", "ESG_type": "S"}, 
     {"promise_status": "Yes", "evidence_status": "Yes", "evidence_quality": "Misleading", "verification_timeline": "within_2_years"})
]
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
    """建立 multi-shot 對話（多組示範）+ 當前樣本"""
    current_user = {
        "role": "user",
        "content": json.dumps({
            "data": sample.get("data", ""),
            "URL": sample.get("URL", ""),
            "page_number": sample.get("page_number", ""),
            "ESG_type": sample.get("ESG_type", "")
        }, ensure_ascii=False)
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in FEWSHOTS:
        messages.append({"role": "user", "content": json.dumps(u, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})
    messages.append(current_user)
    return messages

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

    # 統一大小寫/拼字
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

    # 規則：若 evidence_status=No -> quality=N/A
    if e == "No":
        q = "N/A"

    # 保證在枚舉集合內
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
                temperature=TEMPERATURE,
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
                        temperature=TEMPERATURE,
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
                pred = _normalize_pred(pred)  # 再保險一次
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
