# -*- coding: utf-8 -*-
"""benchmark_clova.csv → v3 vs v5 vs CLOVA 3-모델 복원율 막대그래프(PNG). 발표용.
   상용 CLOVA를 비교 기준으로 얹어 '오픈소스 v5의 상용 대비 경쟁력'을 시각화."""
import re, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ocr_results"

for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(f.lower() in fn.name.lower() for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

rows = list(csv.DictReader(open(OUT / "benchmark_clova.csv", encoding="utf-8-sig")))


def frac(s):
    m = re.match(r"\s*(\d+)/(\d+)", str(s))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 1)


rows.sort(key=lambda r: int(re.sub(r"\D", "", r["image"]) or "0"))
labels = [r["image"].rsplit(".", 1)[0] for r in rows]
cols = [("PP-OCRv3", "#9aa5b1"), ("PP-OCRv5", "#2f6fed"), ("CLOVA", "#14a06e")]
data = {c: [frac(r[c])[0] / frac(r[c])[1] * 100 for r in rows] for c, _ in cols}

# 전체 합계 막대
labels.append("전체")
for c, _ in cols:
    n = sum(frac(r[c])[0] for r in rows); d = sum(frac(r[c])[1] for r in rows)
    data[c].append(n / d * 100)

legend = {"PP-OCRv3": "PP-OCRv3 (베이스라인) 65%",
          "PP-OCRv5": "PP-OCRv5 (개선) 84%",
          "CLOVA": "CLOVA OCR (상용 기준) 87%"}

x = range(len(labels)); w = 0.27
fig, ax = plt.subplots(figsize=(18, 6))
for i, (c, color) in enumerate(cols):
    off = (i - 1) * w
    bars = ax.bar([j + off for j in x], data[c], w, label=legend[c], color=color)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                ha="center", va="bottom", fontsize=5.2)
ax.set_ylabel("메뉴명 복원율 (%)"); ax.set_ylim(0, 108)
ax.set_title("한식 메뉴명 복원율: PP-OCRv3 vs PP-OCRv5 vs 상용 CLOVA OCR")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.axvline(len(labels) - 1.5, color="#ccc", ls="--", lw=1)
ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path = OUT / "benchmark_chart_clova.png"
plt.savefig(path, dpi=150)
print("저장:", path)
