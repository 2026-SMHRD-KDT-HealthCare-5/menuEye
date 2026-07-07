# -*- coding: utf-8 -*-
"""MenuEye — 한식 메뉴판 다국어 해석기 (Streamlit 데모)
실행:  streamlit run app.py
파이프라인(OCR→병합→LLM)은 scripts/menu_to_prompt.py 를 그대로 재사용한다.
"""
import os
os.environ.setdefault("FLAGS_enable_pir_api", "0")
import sys, re, json, tempfile, hashlib, threading, time, pickle
from pathlib import Path
import numpy as np
import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_ocr
import menu_to_prompt as M

st.set_page_config(page_title="MenuEye · 한식 메뉴판 해석기", page_icon="🍚", layout="wide")
KR_DIR = ROOT / "data" / "menu_images" / "menu" / "한국"

# 결과 카드 라벨 — 번역 목표 언어에 맞춰 표시 (값은 LLM이 이미 목표 언어로 생성)
LABELS = {
    "English": {"ingredients": "Ingredients", "cooking": "Cooking", "taste": "Taste",
                "spicy": "Spiciness", "spicy_none": "None", "allergen": "⚠️ Possible allergens",
                "tail": "— please check with the restaurant to be sure.",
                "candidates": "Detected menu items", "done": "dishes"},
    "한국어": {"ingredients": "재료", "cooking": "조리법", "taste": "맛",
              "spicy": "매운맛", "spicy_none": "없음", "allergen": "⚠️ 의심 알레르겐",
              "tail": "— 정확한 건 식당에 직접 문의하세요.",
              "candidates": "인식된 메뉴 후보", "done": "개 요리"},
    "日本語": {"ingredients": "材料", "cooking": "調理法", "taste": "味",
             "spicy": "辛さ", "spicy_none": "なし", "allergen": "⚠️ アレルギー注意",
             "tail": "— 正確な情報はお店にご確認ください。",
             "candidates": "認識されたメニュー候補", "done": "品"},
    "中文": {"ingredients": "食材", "cooking": "烹饪方式", "taste": "口味",
            "spicy": "辣度", "spicy_none": "无", "allergen": "⚠️ 可能致敏原",
            "tail": "— 详情请咨询餐厅。",
            "candidates": "识别的菜单项", "done": "道菜"},
}


@st.cache_resource(show_spinner="OCR 엔진 로딩 중... (최초 1회, 20초 내외)")
def load_engine():
    return run_ocr.get_engine("korean", "PP-OCRv5")


def read_key(provider):
    fname = {"openai": "openai_key.txt", "gemini": "gemini_key.txt"}[provider]
    p = ROOT / fname
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return os.getenv("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY")


def imdecode(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def draw_boxes(bgr, lines):
    out = bgr.copy()
    for t, s, box in lines:
        if box is None:
            continue
        try:
            pts = np.array([[int(p[0]), int(p[1])] for p in box], np.int32)
            cv2.polylines(out, [pts], True, (0, 0, 255), 2)
        except Exception:
            pass
    return out


def show_img(col, bgr, caption, max_w=580, max_h=420):
    """긴 세로 사진이 화면을 밀어 '해석하기' 버튼이 스크롤되지 않도록
       미리보기 크기를 (최대 폭·높이)로 제한해 표시한다. OCR은 원본을 쓰므로 무관."""
    h, w = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    col.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption=caption)


def extract_names(lines):
    raw = M.merge_line_fragments(lines)
    names, seen = [], set()
    for t in raw:
        t = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", (t or "").strip())
        if M.looks_like_name(t) and t.lower() not in seen:
            seen.add(t.lower()); names.append(t)
    return names


FALLBACKS = {
    "gemini": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-flash-lite-latest"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
}


def check_and_pick(provider, model):
    """LLM 사용 가능 확인. 선택 모델이 안 되면 대체 모델을 자동 탐색.
    반환: (ok, 사용할_모델, 메시지)."""
    cands = [model] + [m for m in FALLBACKS.get(provider, []) if m != model]
    last = ""
    for m in cands:
        try:
            M.ping_llm(provider, m)
            return True, m, ("" if m == model else f"'{model}' 사용 불가 → '{m}'로 자동 대체")
        except Exception as e:
            last = str(e)
            low = last.lower()
            if "insufficient_quota" in low or "api key" in low or "api_key" in low or "permission" in low:
                break   # 제공자/키 문제 → 다른 모델 시도 무의미
    return False, None, last[:200]


# OCR 결과 영구 캐시(파일). 앱을 껐다 켜도 유지 → 시연 때 미리 한 번 데워두면
# 동일 이미지는 재시작해도 즉시 표시(박스 시각화=인식 과정은 그대로 렌더된다).
_OCR_CACHE_PATH = ROOT / "data" / "ocr_cache.pkl"
_OCR_CACHE = {}   # 이미지 해시 → OCR 결과 (프로세스 내 + 파일 유지)
if _OCR_CACHE_PATH.exists():
    try:
        _OCR_CACHE = pickle.loads(_OCR_CACHE_PATH.read_bytes())
    except Exception:
        _OCR_CACHE = {}


def _ocr_raw(img_bytes, eng):
    """OCR 실행(파일 캐시 포함). Streamlit 호출이 없어 워커 스레드에서 안전.
       동일 이미지는 파일 캐시에서 즉시 반환 → 시연 시 재시작해도 빠르다."""
    h = hashlib.sha256(img_bytes).hexdigest()
    if h in _OCR_CACHE:
        return _OCR_CACHE[h]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf.write(cv2.imencode(".png", imdecode(img_bytes))[1].tobytes())
        tmp = tf.name
    lines = run_ocr.run_image(eng, tmp)
    _OCR_CACHE[h] = lines
    try:
        _OCR_CACHE_PATH.write_bytes(pickle.dumps(_OCR_CACHE))   # 파일로 영구 저장
    except Exception:
        pass
    return lines


# ── UI ──
st.title("🍚 MenuEye — 한식 메뉴판 다국어 해석기")
st.caption("사진 한 장 → 글자 인식(PaddleOCR v5) → LLM이 소개·재료·맛·알레르기 주의를 다국어로 안내. "
           "방한 외국인을 위한 메뉴 가이드. (추천, 가격 미포함)")

eng = load_engine()   # 앱 시작 시 OCR 엔진 예열(최초 1회 ~20초, 이후 모든 해석이 빨라짐)

with st.sidebar:
    st.header("⚙️ 설정")
    provider = st.selectbox("LLM 제공자", ["gemini", "openai"], index=0)
    model = st.selectbox("모델", {
        "gemini": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
        "openai": ["gpt-4o-mini", "gpt-4o"],
    }[provider])
    target = st.selectbox("번역 언어", ["English", "한국어", "日本語", "中文"], index=0)
    st.divider()
    st.caption("샘플 또는 직접 업로드")
    _imgs = sorted([p for p in KR_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")],
                   key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))   # 1,2,…,25 숫자순
    _opts = ["— 선택하세요 —"] + [p.name for p in _imgs]
    sample = st.selectbox("샘플 이미지", _opts, index=0)   # 첫 화면 = 깨끗한 랜딩(자동 선택 없음)

up = st.file_uploader("메뉴판 이미지 업로드", type=["jpg", "jpeg", "png", "webp"])

img_bytes, img_name = None, None
if up is not None:
    img_bytes, img_name = up.read(), up.name
elif sample != "— 선택하세요 —":
    p = KR_DIR / sample
    img_bytes, img_name = p.read_bytes(), sample

if img_bytes is None:
    st.info("👈 왼쪽 사이드바에서 **샘플 이미지**를 고르거나 메뉴판 사진을 **업로드**하세요.")
    st.markdown("**사용 순서**  ①  이미지 선택 · 업로드  →  ②  🔍 해석하기  →  ③  요리별 다국어 카드")
    st.stop()

bgr = imdecode(img_bytes)
c1, c2 = st.columns(2)
show_img(c1, bgr, f"원본 · {img_name}")

if not st.button("🔍 해석하기", type="primary"):
    st.stop()

key = read_key(provider)
if not key:
    st.error(f"{provider} API 키가 없습니다. 프로젝트 폴더에 `{provider}_key.txt`를 두거나 환경변수를 설정하세요.")
    st.stop()
os.environ["OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"] = key
L = LABELS.get(target, LABELS["English"])

prog = st.progress(0, text="시작...")

# 전체 작업(LLM 확인 → OCR → 해석)을 워커 스레드로 돌리고,
# 메인 스레드는 진행바를 95%까지 '점근식'으로 서서히 채운다(멈춤 없이 부드럽게).
R = {"phase": "① LLM 사용 가능 확인 중...", "done": False}


def _pipeline():
    try:
        ok, um, pmsg = check_and_pick(provider, model)   # OCR 전에 LLM 먼저 확인
        if not ok:
            R["error"] = f"지금 {provider} LLM을 사용할 수 없습니다.\n{pmsg}"
            return
        R["use_model"] = um
        R["pick_msg"] = "" if um == model else pmsg
        R["phase"] = "② 글자 인식 중 (OCR)..."
        R["lines"] = _ocr_raw(img_bytes, eng)
        R["phase"] = "③ 메뉴 후보 정리 중..."
        R["names"] = extract_names(R["lines"])
        R["phase"] = f"④ {provider} · {um} 해석 중..."
        prompt = M.build_prompt(R["names"], target)
        cache = M._load_cache()
        ckey = M._cache_key(provider, um, prompt)
        if ckey in cache:
            R["data"], R["cached"] = cache[ckey], True
        else:
            data = M.parse_json(M.call_llm(prompt, provider, um))
            cache[ckey] = data
            M.LLM_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            R["data"] = data
    except Exception as e:
        R["error"] = str(e)
    finally:
        R["done"] = True


th = threading.Thread(target=_pipeline, daemon=True)
th.start()

p = 0.0
while not R["done"]:
    p += (95 - p) * 0.025           # 95%에 점근 → 멈추지 않고 서서히 채워짐
    prog.progress(int(p), text=R["phase"])
    time.sleep(0.1)
th.join()

if R.get("error"):
    prog.empty()
    st.error(f"❌ {R['error']}\n\n다른 모델/제공자를 선택해 다시 시도하세요.")
    st.stop()

for q in range(int(p), 101, 2):     # 마지막 100%까지 부드럽게 마무리
    prog.progress(q, text="✅ 완료!")
    time.sleep(0.008)
prog.empty()

use_model, lines, names, data = R["use_model"], R["lines"], R["names"], R["data"]
if R.get("pick_msg"):
    st.info(f"ℹ️ {R['pick_msg']}")
if R.get("cached"):
    st.caption("↺ 동일 입력 — 캐시된 해석 사용 (API 호출 생략)")

show_img(c2, draw_boxes(bgr, lines), f"인식 박스 {len(lines)}개")
st.write(f"**{L['candidates']}: {len(names)}** — {', '.join(names[:20])}{' …' if len(names) > 20 else ''}")

items = data.get("items", [])
st.success(f"✅ {len(items)} {L['done']}")
for it in items:
    with st.container(border=True):
        title = it.get("original", "")
        tr = it.get("translation", "")
        st.subheader(f"{title}  ·  {tr}" if tr else title)
        if it.get("romanization"):
            st.caption(it["romanization"])
        if it.get("intro"):
            st.write(it["intro"])
        a, b = st.columns(2)
        ing = it.get("ingredients") or []
        if ing:
            a.markdown(f"**{L['ingredients']}** · " + ", ".join(ing))
        if it.get("cooking_method"):
            a.markdown(f"**{L['cooking']}** · " + it["cooking_method"])
        if it.get("taste"):
            b.markdown(f"**{L['taste']}** · " + it["taste"])
        sp = int(it.get("spicy_level", 0) or 0)
        b.markdown(f"**{L['spicy']}** · " + ("🌶️" * sp if sp else L["spicy_none"]) + f"  ({sp}/5)")
        cau = it.get("allergen_caution") or []
        if cau:
            st.warning(f"{L['allergen']}: " + ", ".join(cau) + " " + L["tail"])

if data.get("notice"):
    st.info("ℹ️ " + data["notice"])
