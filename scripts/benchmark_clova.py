# -*- coding: utf-8 -*-
"""
benchmark_clova.py — CLOVA General OCR를 PP-OCRv3/v5와 '같은 잣대'로 비교.
=========================================================================
기존 benchmark.csv(v3/v5, Colab 산출)에 **CLOVA 열만 추가**한다.
CLOVA 출력을 v3/v5와 동일한 후처리(merge_line_fragments)·채점(compare_menu 유사도)에
그대로 태워 '모델 효과'만 순수 비교 → benchmark_clova.csv 저장.

실행:  & "C:\\Users\\smhrd1\\anaconda3\\python.exe" scripts\\benchmark_clova.py
"""
from __future__ import annotations
import sys, re, csv, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_clova
import menu_to_prompt as M
import compare_menu as C

KR_DIR = ROOT / "data" / "menu_images" / "menu" / "한국"
OUT = ROOT / "data" / "ocr_results"
THRESHOLD = 0.6
_SPACE = re.compile(r"(?<=[가-힣])\s+(?=[가-힣])")


def extract_names_clova(path: Path):
    """CLOVA(text,score,box) → benchmark_ocr.extract_names와 동일 경로로 메뉴명 후보."""
    items = run_clova.clova_ocr_items(str(path))
    raw = M.merge_line_fragments(items)
    names, seen = [], set()
    for t in raw:
        t = _SPACE.sub("", (t or "").strip())
        if not M.looks_like_name(t) or t.lower() in seen:
            continue
        seen.add(t.lower()); names.append(t)
    return names


def recall(truth, names):
    n = 0
    for tr in truth:
        best = max((C.similarity(tr, x) for x in names), default=0.0)
        if best >= THRESHOLD:
            n += 1
    return n


def find_image(num: str):
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = KR_DIR / f"{num}{ext}"
        if p.exists():
            return p
    return None


def load_prev():
    """기존 benchmark.csv(v3/v5) → {image: {truth, PP-OCRv3, PP-OCRv5}}."""
    f = OUT / "benchmark.csv"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8-sig", newline="") as fh:
        return {r["image"]: r for r in csv.DictReader(fh)}


def main():
    truths = sorted((ROOT / "data").glob("truth_*.txt"),
                    key=lambda p: int(re.sub(r"\D", "", p.stem.split("_")[-1]) or 0))
    if not truths:
        print("truth_<N>.txt 없음."); return 2
    prev = load_prev()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    tot_hit = tot_truth = 0
    hdr = f"{'이미지':<9}{'정답':>4}{'v3':>10}{'v5':>10}{'CLOVA':>10}"
    print(hdr); print("-" * len(hdr))
    for tf in truths:
        num = tf.stem.split("_")[-1]
        img = find_image(num)
        if img is None:
            print(f"[skip] {num}: 이미지 없음"); continue
        truth = [l.strip() for l in tf.read_text(encoding="utf-8").splitlines() if l.strip()]
        try:
            n = recall(truth, extract_names_clova(img))
            clova_cell = f"{n}/{len(truth)} ({n/len(truth):.0%})"
            tot_hit += n; tot_truth += len(truth)
        except Exception as e:
            clova_cell = "ERR"
            print(f"  [!] {img.name} CLOVA 실패: {str(e)[:120]}")
        pv = prev.get(img.name, {})
        v3 = pv.get("PP-OCRv3", "-"); v5 = pv.get("PP-OCRv5", "-")
        rows.append({"image": img.name, "truth": len(truth),
                     "PP-OCRv3": v3, "PP-OCRv5": v5, "CLOVA": clova_cell})
        print(f"{img.name:<9}{len(truth):>4}{v3.split(' ')[-1] if v3!='-' else '-':>10}"
              f"{v5.split(' ')[-1] if v5!='-' else '-':>10}{clova_cell.split(' ')[-1]:>10}")
        time.sleep(0.4)   # 레이트리밋 여유

    print("-" * len(hdr))
    print(f"{'CLOVA 합계':<9}{tot_truth:>4}{'':>20}{tot_hit}/{tot_truth} ({tot_hit/max(1,tot_truth):.0%})")

    with open(OUT / "benchmark_clova.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "truth", "PP-OCRv3", "PP-OCRv5", "CLOVA"])
        w.writeheader(); w.writerows(rows)
    print(f"\n저장: {OUT / 'benchmark_clova.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
