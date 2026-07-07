# -*- coding: utf-8 -*-
"""
run_clova.py — 네이버 CLOVA General OCR(V2)로 메뉴판 이미지를 읽어 텍스트 줄 리스트를 반환.
v3/v5(PaddleOCR)와 같은 잣대로 비교하기 위해 인식 결과 텍스트만 뽑는다.

자격증명(둘 중 하나):
  1) 환경변수  CLOVA_OCR_URL / CLOVA_OCR_SECRET
  2) 프로젝트 루트 파일  clova_url.txt / clova_secret.txt   (.gitignore 처리됨)
  ※ 시크릿은 코드/깃/로그에 절대 노출하지 않는다.

사용:
  & "C:\\Users\\smhrd1\\anaconda3\\python.exe" scripts\\run_clova.py "data\\menu_images\\menu\\한국\\17.jpg"

핵심(중요):
  파일 확장자를 믿지 않는다. 실제 바이트를 cv2로 디코드해 **PNG로 재인코딩**해서 전송한다.
  → 확장자만 .jpg인 WebP, .webp, 프로그레시브/CMYK 등 CLOVA 디코더가 거부하는 케이스를 모두 정규화.
"""
import os, sys, io, json, time, base64, uuid
from pathlib import Path
import requests
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_cred(env_name, fallback_file):
    """환경변수 우선, 없으면 gitignore된 파일. BOM 안전(utf-8-sig)."""
    v = os.getenv(env_name)
    if v and v.strip():
        return v.strip()
    p = ROOT / fallback_file
    if p.exists():
        t = p.read_text(encoding="utf-8-sig").strip()
        if t:
            return t
    return None


CLOVA_URL = _load_cred("CLOVA_OCR_URL", "clova_url.txt")
CLOVA_SECRET = _load_cred("CLOVA_OCR_SECRET", "clova_secret.txt")


def _mask(secret):
    return f"<len={len(secret)} tail=...{secret[-3:]}>" if secret else "<없음>"


def _image_to_png_b64(image_path):
    """확장자 무관: 실제 바이트를 cv2로 디코드 → PNG 재인코딩 → base64.
       (CLOVA는 webp 미지원, 또한 일부 jpg 인코딩을 거부 → 정규화가 정답)"""
    raw = np.fromfile(image_path, dtype=np.uint8)  # 한글 경로 안전(np.fromfile)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"이미지 디코드 실패(지원 안 되는 형식?): {image_path}")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"PNG 재인코딩 실패: {image_path}")
    return base64.b64encode(buf.tobytes()).decode(), img.shape[:2]


def _clova_request(image_path, verbose=False):
    """CLOVA에 1장 전송 → images[0].fields 리스트 반환(원본)."""
    if not CLOVA_URL or not CLOVA_SECRET:
        raise RuntimeError(
            "자격증명 없음: 환경변수 CLOVA_OCR_URL/CLOVA_OCR_SECRET 또는 "
            "루트의 clova_url.txt/clova_secret.txt 를 준비하세요."
        )

    data_b64, (h, w) = _image_to_png_b64(image_path)

    payload = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "images": [{"format": "png", "name": "menu", "data": data_b64}],
    }
    headers = {"X-OCR-SECRET": CLOVA_SECRET, "Content-Type": "application/json"}

    if verbose:
        skel = {**payload, "images": [{"format": "png", "name": "menu",
                                       "data": f"<base64 len={len(data_b64)}>"}]}
        print(f"[요청] URL=...{CLOVA_URL[-14:]}  X-OCR-SECRET={_mask(CLOVA_SECRET)}")
        print(f"[요청] 정규화: → PNG {w}x{h}  payload={json.dumps(skel, ensure_ascii=False)}")

    r = requests.post(CLOVA_URL, headers=headers, data=json.dumps(payload), timeout=60)
    if r.status_code != 200:
        # 실패 시 서버 원문 전체 출력(본문에 시크릿은 포함되지 않음)
        raise RuntimeError(f"CLOVA {r.status_code}: {r.text}")
    return r.json()["images"][0].get("fields", [])


def _poly(field):
    """boundingPoly.vertices → [[x,y]*4] (run_ocr와 동일 형식). 없으면 None."""
    try:
        vs = field["boundingPoly"]["vertices"]
        return [[float(v.get("x", 0.0)), float(v.get("y", 0.0))] for v in vs]
    except Exception:
        return None


def clova_ocr_items(image_path, verbose=False):
    """[(text, score, box)] — PaddleOCR(run_image)와 동일 구조.
       → 같은 merge_line_fragments/채점 파이프라인을 그대로 재사용 가능."""
    items = []
    for f in _clova_request(image_path, verbose):
        t = (f.get("inferText") or "").strip()
        if not t:
            continue
        s = f.get("inferConfidence")
        items.append((t, float(s) if s is not None else None, _poly(f)))
    return items


def clova_ocr_lines(image_path, verbose=False):
    """이미지 1장 → CLOVA OCR → 인식된 텍스트 줄 리스트(단순)."""
    return [t for t, _, _ in clova_ocr_items(image_path, verbose)]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용: python scripts\\run_clova.py <이미지경로> [--verbose]")
        sys.exit(2)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    path = sys.argv[1]
    verbose = "--verbose" in sys.argv[2:] or "-v" in sys.argv[2:]
    print(f"CLOVA OCR 실행: {path}")
    try:
        lines = clova_ocr_lines(path, verbose=verbose)
    except Exception as e:
        print("에러 발생:", str(e)[:1500])
        sys.exit(1)
    print(f"인식된 조각: {len(lines)}개")
    for t in lines:
        print("  ", t)
