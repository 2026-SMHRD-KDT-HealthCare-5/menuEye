# -*- coding: utf-8 -*-
"""
메뉴판 이미지 OCR 실행 (MenuEye 베이스라인)
==========================================
data/menu_images/ 아래 메뉴판 이미지를 PaddleOCR로 인식해 결과를 저장한다.
PaddleOCR 2.x / 3.x API를 모두 방어적으로 처리한다.

실행:
    python scripts/run_ocr.py
    python scripts/run_ocr.py --lang latin     # 독/프/스/이 = 라틴문자

출력 (data/ocr_results/):
    ocr_lines.csv    : 인식된 텍스트 줄 단위 (country, image, text, score, box크기, is_price)
    ocr_summary.csv  : 이미지별 요약 (줄수, 평균신뢰도, 저신뢰비율, 가격검출수)
    ocr_results.json : 전체 원본 결과
필요: paddlepaddle, paddleocr  (pip install paddlepaddle paddleocr)
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# PaddlePaddle 3.x CPU 백엔드 버그 회피: 신규 PIR 실행기 비활성화 (핵심 수정)
# (ConvertPirAttribute2RuntimeAttribute ... onednn_instruction.cc 오류 대응)
os.environ.setdefault("FLAGS_enable_pir_api", "0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / "data" / "menu_images"
OUT_DIR = PROJECT_ROOT / "data" / "ocr_results"

# 가격 토큰 정규식 (menu_translator/grouping.py와 동일 취지)
PRICE_RE = re.compile(
    r"(?:[€$£₩¥]\s?\d[\d.,]*)|(?:\d[\d.,]*\s?(?:원|円|元|EUR|USD))|(?:\d{1,3}(?:[.,]\d{3})+)|(?:\d+[.,]\d{2})"
)


# 나라 폴더명 → PaddleOCR 언어코드 (v2: 한식 메뉴판 → 한국어 주력)
COUNTRY_LANG = {"한국": "korean"}
FALLBACK_LANG = "korean"   # 폴더 매핑이 없으면 한국어로 폴백

# 스캔할 이미지 확장자 (jpg/jpeg/png/webp 모두)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

_engine_cache: dict = {}


def _make_engine(lang: str, version: str):
    """PaddleOCR 2.x/3.x 어느 쪽이든 생성되도록 kwargs를 순차 시도."""
    from paddleocr import PaddleOCR
    # 핵심: enable_mkldnn=False (oneDNN 끔 → paddle 3.x 버그 회피, 검증됨)
    # + 무거운 문서 방향/휨 보정 모델 끄기(메뉴 사진엔 불필요 → 속도)
    fast = dict(enable_mkldnn=False,
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False)
    attempts = [
        dict(lang=lang, ocr_version=version, **fast),                         # 3.x, 빠름+안전
        dict(lang=lang, ocr_version=version, use_textline_orientation=True, enable_mkldnn=False),
        dict(lang=lang, ocr_version=version),
        dict(lang=lang, use_angle_cls=True, show_log=False),                  # 2.x
        dict(lang=lang),
    ]
    last = None
    for kw in attempts:
        try:
            return PaddleOCR(**kw)
        except Exception as e:
            last = e
    raise last


def get_engine(lang: str, version: str):
    """언어별 엔진을 캐시. 해당 언어 모델이 없으면 라틴 폴백."""
    key = (lang, version)
    if key in _engine_cache:
        return _engine_cache[key]
    try:
        eng = _make_engine(lang, version)
        print(f"[engine] lang={lang} ({version}) 생성 성공")
    except Exception as e:
        print(f"[engine] lang={lang} 실패({str(e)[:70]}) → 폴백 {FALLBACK_LANG}")
        eng = _make_engine(FALLBACK_LANG, version)
    _engine_cache[key] = eng
    return eng


def _box_dims(box):
    """box(4점 좌표 또는 numpy) → (w, h, 직렬화가능 box)."""
    if box is None:
        return None, None, None
    try:
        pts = [[float(p[0]), float(p[1])] for p in box]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return max(xs) - min(xs), max(ys) - min(ys), pts
    except Exception:
        return None, None, None


def parse_result(raw):
    """OCR 원본 → [(text, score, box), ...] (2.x/3.x 자동 판별)."""
    lines = []
    if not raw:
        return lines
    first = raw[0]
    # 3.x: predict() → dict 유사 객체(OCRResult) 리스트
    if hasattr(first, "get") and ("rec_texts" in first or "rec_scores" in first):
        for res in raw:
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            polys = res.get("rec_polys")
            if polys is None:
                polys = res.get("dt_polys")
            if polys is None:
                polys = res.get("rec_boxes")
            for i, t in enumerate(texts):
                s = float(scores[i]) if i < len(scores) else None
                box = polys[i] if (polys is not None and i < len(polys)) else None
                lines.append((str(t), s, box))
        return lines
    # 2.x: ocr() → [page] / page = [[box,(text,score)], ...]
    for page in raw:
        if not page:
            continue
        for item in page:
            try:
                box, (text, score) = item[0], item[1]
            except Exception:
                continue
            lines.append((str(text), float(score), box))
    return lines


def run_image(ocr, path: str):
    raw = None
    if hasattr(ocr, "predict"):
        try:
            raw = ocr.predict(path)
        except Exception as e:
            print(f"  [predict 실패→ocr() 시도] {e}")
            raw = None
    if raw is None:
        raw = ocr.ocr(path)
    return parse_result(raw)


def main():
    ap = argparse.ArgumentParser(description="메뉴판 이미지 OCR 실행")
    ap.add_argument("--lang", default="auto",
                    help="PaddleOCR 언어코드. 기본 auto=나라 폴더별 언어 자동 적용")
    ap.add_argument("--version", default="PP-OCRv5", help="OCR 모델 버전 (한국어는 PP-OCRv5 권장: 저해상도·정확도 우수)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="저신뢰 판정 임계값")
    args = ap.parse_args()

    images = sorted(p for p in IMG_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[오류] 이미지가 없습니다: {IMG_DIR}", file=sys.stderr)
        return 2
    print(f"이미지 {len(images)}장, lang={args.lang}, version={args.version}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    line_rows, summary_rows, full = [], [], []
    t0 = time.time()
    for idx, img in enumerate(images, 1):
        country = img.parent.name
        rel = img.relative_to(IMG_DIR).as_posix()
        lang = COUNTRY_LANG.get(country, FALLBACK_LANG) if args.lang == "auto" else args.lang
        ocr = get_engine(lang, args.version)
        try:
            lines = run_image(ocr, str(img))
        except Exception as e:
            print(f"[{idx}/{len(images)}] {rel}  OCR 실패: {e}")
            lines = []
        scores = [s for (_, s, _) in lines if s is not None]
        n_price = 0
        img_lines = []
        for (text, score, box) in lines:
            w, h, pts = _box_dims(box)
            is_price = bool(PRICE_RE.search(text))
            n_price += int(is_price)
            line_rows.append({
                "country": country, "image": rel, "text": text,
                "score": round(score, 4) if score is not None else "",
                "box_w": round(w, 1) if w else "", "box_h": round(h, 1) if h else "",
                "is_price": is_price,
            })
            img_lines.append({"text": text, "score": score, "box": pts, "is_price": is_price})
        mean_s = sum(scores) / len(scores) if scores else 0.0
        low_ratio = (sum(1 for s in scores if s < args.min_conf) / len(scores)) if scores else 0.0
        summary_rows.append({
            "country": country, "lang": lang, "image": rel, "n_lines": len(lines),
            "mean_score": round(mean_s, 4), "low_conf_ratio": round(low_ratio, 4),
            "n_price": n_price,
        })
        full.append({"country": country, "image": rel, "lines": img_lines})
        print(f"[{idx}/{len(images)}] {rel:24s} 줄={len(lines):3d} 평균신뢰도={mean_s:.3f} 가격검출={n_price}")

    # 저장
    with open(OUT_DIR / "ocr_lines.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["country", "image", "text", "score", "box_w", "box_h", "is_price"])
        w.writeheader(); w.writerows(line_rows)
    with open(OUT_DIR / "ocr_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["country", "lang", "image", "n_lines", "mean_score", "low_conf_ratio", "n_price"])
        w.writeheader(); w.writerows(summary_rows)
    with open(OUT_DIR / "ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    # 콘솔 요약
    total_lines = sum(r["n_lines"] for r in summary_rows)
    all_scores = [float(r["score"]) for r in line_rows if r["score"] != ""]
    print("\n" + "=" * 56)
    print(f"완료: {len(images)}장, 총 {total_lines}줄, {time.time()-t0:.1f}s")
    if all_scores:
        print(f"전체 평균 신뢰도: {sum(all_scores)/len(all_scores):.3f} "
              f"| 저신뢰(<{args.min_conf}) 비율: {sum(1 for s in all_scores if s < args.min_conf)/len(all_scores):.1%}")
    print("\n나라별 평균 신뢰도 / 줄수:")
    by_country = {}
    for r in summary_rows:
        by_country.setdefault(r["country"], []).append(r)
    for c, rs in by_country.items():
        ms = sum(x["mean_score"] for x in rs) / len(rs)
        nl = sum(x["n_lines"] for x in rs)
        npx = sum(x["n_price"] for x in rs)
        print(f"  {c:6s}  평균신뢰도={ms:.3f}  줄={nl:4d}  가격검출={npx}")
    print(f"\n저장 위치: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
