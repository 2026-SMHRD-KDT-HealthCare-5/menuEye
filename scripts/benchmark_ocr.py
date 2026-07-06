# -*- coding: utf-8 -*-
"""
OCR 모델 비교 벤치마크 (PP-OCRv3 vs PP-OCRv5)
=============================================
data/truth_<N>.txt (원어민 정답)이 있는 이미지에 대해, 두 모델로
OCR→병합→필터한 '메뉴명 복원율(recall)'을 정량 비교한다. LLM 불필요(무비용).

- recall = (정답 메뉴 중 추출 이름과 유사도 THRESHOLD 이상으로 매칭된 개수) / 정답 수
- 같은 후처리(merge_line_fragments)를 두 모델에 동일 적용 → '모델 효과'만 순수 비교.

실행:  python scripts/benchmark_ocr.py
출력:  콘솔 표 + data/ocr_results/benchmark.csv
"""
from __future__ import annotations
import os
os.environ.setdefault("FLAGS_enable_pir_api", "0")
import sys, re, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_ocr
import menu_to_prompt as M
import compare_menu as C

KR_DIR = ROOT / "data" / "menu_images" / "menu" / "한국"
OUT = ROOT / "data" / "ocr_results"
VERSIONS = ["PP-OCRv3", "PP-OCRv5"]
THRESHOLD = 0.6                       # 이 이상이면 '복원됨'으로 판정
_SPACE = re.compile(r"(?<=[가-힣])\s+(?=[가-힣])")


def extract_names(path: Path, version: str):
    """menu_to_prompt와 동일한 OCR→병합→필터 경로로 메뉴명 후보 추출."""
    eng = run_ocr.get_engine("korean", version)
    lines = run_ocr.run_image(eng, str(path))
    raw = M.merge_line_fragments(lines)
    names, seen = [], set()
    for t in raw:
        t = _SPACE.sub("", (t or "").strip())
        if not M.looks_like_name(t) or t.lower() in seen:
            continue
        seen.add(t.lower()); names.append(t)
    return names


def recall(truth, names):
    """각 정답에 대해 최고 유사도 + 복원 여부."""
    out = []
    for tr in truth:
        best = max((C.similarity(tr, n) for n in names), default=0.0)
        out.append((tr, best >= THRESHOLD, round(best, 2)))
    return out


def find_image(num: str):
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = KR_DIR / f"{num}{ext}"
        if p.exists():
            return p
    return None


def main():
    truths = sorted((ROOT / "data").glob("truth_*.txt"),
                    key=lambda p: p.stem.split("_")[-1])
    if not truths:
        print("truth_<N>.txt 파일이 없습니다."); return 2

    OUT.mkdir(parents=True, exist_ok=True)
    rows, agg = [], {v: [0, 0] for v in VERSIONS}
    misses = {v: {} for v in VERSIONS}

    print(f"{'이미지':<10}{'정답수':>6}   " + "".join(f"{v+' recall':>16}" for v in VERSIONS))
    print("-" * 60)
    for tf in truths:
        num = tf.stem.split("_")[-1]
        img = find_image(num)
        if img is None:
            print(f"[skip] {num}: 이미지 없음"); continue
        truth = [l.strip() for l in tf.read_text(encoding="utf-8").splitlines() if l.strip()]
        cells, row = [], {"image": img.name, "truth": len(truth)}
        for v in VERSIONS:
            hits = recall(truth, extract_names(img, v))
            n = sum(1 for _, ok, _ in hits if ok)
            agg[v][0] += n; agg[v][1] += len(truth)
            misses[v][img.name] = [tr for tr, ok, _ in hits if not ok]
            row[v] = f"{n}/{len(truth)} ({n/len(truth):.0%})"
            cells.append(f"{n}/{len(truth)} ({n/len(truth):.0%})")
        rows.append(row)
        print(f"{img.name:<10}{len(truth):>6}   " + "".join(f"{c:>16}" for c in cells))

    print("-" * 60)

    def pct(v):
        h, t = agg[v]
        return f"{h}/{t} ({h / max(1, t):.0%})"

    print(f"{'합계':<10}{sum(r['truth'] for r in rows):>6}   " +
          "".join(pct(v).rjust(16) for v in VERSIONS))

    with open(OUT / "benchmark.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "truth"] + VERSIONS)
        w.writeheader(); w.writerows(rows)
    print(f"\n저장: {OUT / 'benchmark.csv'}")

    print("\n[v5가 아직 놓친 정답 — 오류사례]")
    for name, ms in misses["PP-OCRv5"].items():
        if ms:
            print(f"  {name}: {ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
