# -*- coding: utf-8 -*-
"""benchmark.csv → v3 vs v5 메뉴명 복원율 막대그래프(PNG). 발표용."""
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

rows = list(csv.DictReader(open(OUT / "benchmark.csv", encoding="utf-8-sig")))


def frac(s):
    m = re.match(r"\s*(\d+)/(\d+)", s)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 1)


rows.sort(key=lambda r: int(re.sub(r"\D", "", r["image"]) or "0"))   # 숫자순 정렬(1,2,…,25)
labels = [r["image"].rsplit(".", 1)[0] for r in rows]
v3 = [frac(r["PP-OCRv3"])[0] / frac(r["PP-OCRv3"])[1] * 100 for r in rows]
v5 = [frac(r["PP-OCRv5"])[0] / frac(r["PP-OCRv5"])[1] * 100 for r in rows]
# 전체 합계 막대 추가
n3 = sum(frac(r["PP-OCRv3"])[0] for r in rows); d3 = sum(frac(r["PP-OCRv3"])[1] for r in rows)
n5 = sum(frac(r["PP-OCRv5"])[0] for r in rows); d5 = sum(frac(r["PP-OCRv5"])[1] for r in rows)
labels.append("전체"); v3.append(n3 / d3 * 100); v5.append(n5 / d5 * 100)

x = range(len(labels)); w = 0.42
fig, ax = plt.subplots(figsize=(18, 6))
b3 = ax.bar([i - w / 2 for i in x], v3, w, label="PP-OCRv3 (베이스라인)", color="#9aa5b1")
b5 = ax.bar([i + w / 2 for i in x], v5, w, label="PP-OCRv5 (개선)", color="#2f6fed")
for bars in (b3, b5):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                ha="center", va="bottom", fontsize=6)
ax.set_ylabel("메뉴명 복원율 (%)"); ax.set_ylim(0, 108)
ax.set_title("OCR 모델별 한식 메뉴명 복원율: PP-OCRv3 vs PP-OCRv5")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.axvline(len(labels) - 1.5, color="#ccc", ls="--", lw=1)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path = OUT / "benchmark_chart.png"
plt.savefig(path, dpi=150)
print("저장:", path)
