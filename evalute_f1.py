#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
評分：pred(JSONL) + gold(JSON)
- 任務：promise_status / evidence_status / evidence_quality / verification_timeline
- 指標：Macro / Micro / Accuracy
- 輸出：metrics_summary.csv（總表）與 metrics_breakdown.csv（逐類別）
"""

import json, csv
from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, classification_report
)

# ===== 路徑：請改成你的實際位置 =====
PRED_JSONL        = Path(r"C:\Users\lubob\Desktop\Rockling\results\1shot_Japanese_predictions.jsonl")
GOLD_JSON         = Path(r"C:\Users\lubob\Desktop\Rockling\dataset\Japanese_test.json")
OUT_SUMMARY_CSV   = Path(r"C:\Users\lubob\Desktop\Rockling\results\1shot_metrics_summary.csv")
OUT_BREAKDOWN_CSV = Path(r"C:\Users\lubob\Desktop\Rockling\results\1shot_metrics_breakdown.csv")

TASKS = ["promise_status","evidence_status","evidence_quality","verification_timeline"]

# ===== 正規化：對齊精確標籤集合 =====
def norm_yesno(v: str) -> str:
    v = (v or "").strip().lower()
    return "Yes" if v in ["yes","y","true","是"] else "No"

def norm_eq(v: str) -> str:
    v = (v or "").strip()
    allowed = {"Clear","Not Clear","Misleading","N/A"}
    t = v.replace("_"," ").title().replace("  "," ").strip()
    mapping = {"":"N/A","Na":"N/A","N\\A":"N/A"}
    t = mapping.get(t, t)
    return t if t in allowed else "N/A"

def norm_vt(v: str) -> str:
    v = (v or "").strip().lower()
    mapping = {
        "already":"already",
        "within_2_years":"within_2_years",
        "between_2_and_5_years":"between_2_and_5_years",
        "more_than_5_years":"more_than_5_years",
        "n/a":"N/A", "na":"N/A", "":"N/A",
        # 兼容英文長句
        "less than 2 years":"within_2_years",
        "2 to 5 years":"between_2_and_5_years",
        "more than 5 years":"more_than_5_years",
    }
    return mapping.get(v, "N/A")

def normalize_record(rec: dict) -> dict:
    return {
        "promise_status":       norm_yesno(rec.get("promise_status")),
        "evidence_status":      norm_yesno(rec.get("evidence_status")),
        "evidence_quality":     norm_eq(rec.get("evidence_quality")),
        "verification_timeline":norm_vt(rec.get("verification_timeline")),
    }

# ===== 載入 =====
def load_pred_jsonl(path: Path):
    preds = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            j = json.loads(line)
            pid = j.get("id")
            preds[pid if pid is not None else idx] = normalize_record(j.get("pred", {}))
    return preds

def load_gold_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    gold = OrderedDict()
    for idx, item in enumerate(data):
        gid = item.get("id")
        g = {
            "promise_status": item.get("promise_status"),
            "evidence_status": item.get("evidence_status"),
            "evidence_quality": item.get("evidence_quality"),
            "verification_timeline": item.get("verification_timeline"),
        }
        gold[gid if gid is not None else idx] = normalize_record(g)
    return gold

def align_by_id(preds: OrderedDict, gold: OrderedDict):
    if any(k is None for k in preds.keys()) or any(k is None for k in gold.keys()):
        if len(preds) != len(gold):
            raise ValueError("❌ pred 與 gold 筆數不同，且至少一方缺 id，無法順序對齊。")
        keys = list(range(len(preds)))
    else:
        keys = [k for k in preds if k in gold]
        if not keys:
            raise ValueError("❌ pred 與 gold 的 id 無交集。")
    y_pred = {t: [] for t in TASKS}
    y_true = {t: [] for t in TASKS}
    for k in keys:
        p = preds[k]; g = gold[k]
        for t in TASKS:
            y_pred[t].append(p[t]); y_true[t].append(g[t])
    return y_pred, y_true, keys

# ===== 主流程 =====
preds = load_pred_jsonl(PRED_JSONL)
gold  = load_gold_json(GOLD_JSON)
y_pred, y_true, used_keys = align_by_id(preds, gold)
print(f"=== 評分樣本數 === {len(used_keys)}\n")

# 1) 總表：Macro / Micro / Accuracy
rows = []
for t in TASKS:
    p_mac = precision_score(y_true[t], y_pred[t], average="macro", zero_division=0)
    r_mac = recall_score( y_true[t], y_pred[t], average="macro", zero_division=0)
    f_mac = f1_score(     y_true[t], y_pred[t], average="macro", zero_division=0)

    p_mic = precision_score(y_true[t], y_pred[t], average="micro", zero_division=0)
    r_mic = recall_score( y_true[t], y_pred[t], average="micro", zero_division=0)
    f_mic = f1_score(     y_true[t], y_pred[t], average="micro", zero_division=0)

    acc   = accuracy_score(y_true[t], y_pred[t])

    rows.append([t, p_mac, r_mac, f_mac, p_mic, r_mic, f_mic, acc])
    print(f"{t}: Macro(P/R/F1)=({p_mac:.4f}/{r_mac:.4f}/{f_mac:.4f})  "
          f"Micro(P/R/F1)=({p_mic:.4f}/{r_mic:.4f}/{f_mic:.4f})  Acc={acc:.4f}")

avg_mac = [sum(r[i] for r in rows)/len(rows) for i in (1,2,3)]
avg_mic = [sum(r[i] for r in rows)/len(rows) for i in (4,5,6)]
avg_acc = sum(r[7] for r in rows)/len(rows)
print(f"\nAverages →  Macro=({avg_mac[0]:.4f}/{avg_mac[1]:.4f}/{avg_mac[2]:.4f})  "
      f"Micro=({avg_mic[0]:.4f}/{avg_mic[1]:.4f}/{avg_mic[2]:.4f})  Acc={avg_acc:.4f}\n")

# 2) 輸出 CSV：總表
OUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Task","Macro-Precision","Macro-Recall","Macro-F1",
                "Micro-Precision","Micro-Recall","Micro-F1","Accuracy"])
    for r in rows:
        w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
                          f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.4f}", f"{r[7]:.4f}"])
    w.writerow(["Average", f"{avg_mac[0]:.4f}", f"{avg_mac[1]:.4f}", f"{avg_mac[2]:.4f}",
                           f"{avg_mic[0]:.4f}", f"{avg_mic[1]:.4f}", f"{avg_mic[2]:.4f}",
                           f"{avg_acc:.4f}"])
print(f"✅ 指標總表：{OUT_SUMMARY_CSV}")

# 3) 逐類別明細
OUT_BREAKDOWN_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_BREAKDOWN_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Task","Label","Precision","Recall","F1","Support"])
    for t in TASKS:
        rpt = classification_report(y_true[t], y_pred[t], output_dict=True, zero_division=0)
        for label, stats in rpt.items():
            if label in ["accuracy","macro avg","weighted avg"]:
                continue
            w.writerow([t, label,
                        f"{stats.get('precision',0):.4f}",
                        f"{stats.get('recall',0):.4f}",
                        f"{stats.get('f1-score',0):.4f}",
                        int(stats.get('support',0))])
print(f"✅ 逐類別明細：{OUT_BREAKDOWN_CSV}")
