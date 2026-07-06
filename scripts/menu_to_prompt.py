# -*- coding: utf-8 -*-
"""
메뉴판 이미지 1장 → OCR → LLM 해석·가이드 (동적)
================================================
어떤 유럽 메뉴판 사진이든 경로만 주면, OCR 결과로 프롬프트를 만들어
LLM(gpt-4o 등)에 넘겨 메뉴별 소개·재료·조리법·맛·알레르기 주의를 얻는다.

실행:
    python scripts/menu_to_prompt.py "경로/메뉴판.jpg"                      # 프롬프트만 출력
    python scripts/menu_to_prompt.py "경로/메뉴판.jpg" --llm openai --model gpt-4o   # 해석까지

※ 지원 범위(정확도 최적): 가격이 우측에 있는 '표준 아라카르트 메뉴'.
   다단 컬럼·가격 컬럼·코스(prix-fixe) 등 복잡한 레이아웃은 정확도 한계(향후 개선: 레이아웃 분석/비전모델).
"""
from __future__ import annotations
import os
os.environ.setdefault("FLAGS_enable_pir_api", "0")   # 이 PC paddle CPU 버그 회피
import sys, re, json, argparse, hashlib
from pathlib import Path

# 콘솔 UTF-8 고정 (Windows cp949에서 한글 출력 크래시 방지)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_ocr   # 우리가 쓰던 OCR 엔진 재사용

# llm.py와 동일한 프롬프트 구조 (실제 시스템이 쓰는 것과 같은 템플릿)
SCHEMA = '''{
  "items": [
    {
      "original": "원문 메뉴명 (한국어 그대로)",
      "romanization": "메뉴명 로마자 표기 (예: Gonggi-bap)",
      "translation": "목표언어로 번역한 메뉴명 (낯선 한국어 용어는 괄호로 짧게 풀이)",
      "intro": "이 요리가 '일반적으로' 어떤 음식인지 3~4문장 소개 (목표언어). 처음 나오는 한국어 용어는 괄호로 풀이. 이 식당의 실제 레시피를 단정하지 말 것",
      "ingredients": ["'일반적으로 흔히' 쓰이는 대표 재료 3~8개, 목표언어 (구체 명칭). 이 그릇에 확실히 들었다고 단정하지 말 것"],
      "cooking_method": "일반적 조리법, 목표언어로 (예: stir-fried / grilled / stewed / steamed)",
      "taste": "'일반적인' 맛/식감 3~4문장, 목표언어. 왕초보도 이해할 쉽고 명확한 표현. 'earthy/흙내' 같은 모호·부정적 풍미어 금지",
      "spicy_level": 0,
      "allergen_caution": ["의심 알레르겐만, 목표언어로 (예: soy, sesame, egg). 단정 아님"]
    }
  ],
  "notice": "정확한 재료·알레르기 여부는 식당에 직접 문의하세요. (목표언어로 작성)"
}'''


def build_prompt(names, target="English"):
    block = "\n".join(f"{i+1}. {n}" for i, n in enumerate(names))
    return f"""당신은 한식에 정통한 셰프이자, 한국어와 한식을 '전혀 모르는' 외국인에게 메뉴를 풀어 설명하는 해설가입니다.
아래는 한국 식당 메뉴판에서 OCR로 인식한 텍스트 줄들입니다.

[가장 중요한 정직성 규칙 — 반드시 지킬 것]
- 당신은 '이 식당의 실제 레시피'를 알지 못합니다. 메뉴 '이름'으로 알 수 있는 '일반적 상식'만 설명하세요.
- 재료·조리법·맛은 "일반적으로/보통/흔히"(typically, usually, often) 식으로 헤징하고, "이 그릇에 반드시 ~가 들어간다"처럼 단정하지 마세요.
- 이름에서 드러나지 않는 특정 재료(예: 된장, 고추장, 특정 향신료)를 임의로 넣지 마세요. 그 요리에 일반적으로 흔히 쓰일 때만 "often seasoned with ..." 정도로만 언급하고, 아니면 아예 언급하지 마세요.
- 원문에 없는 요리·재료를 지어내지 마세요(환각 금지).

[출력 언어 규칙]
- 번역 목표 언어는 "{target}"입니다.
- "original"만 한국어 원문 그대로 두고, 나머지(translation, intro, ingredients, cooking_method, taste, allergen_caution, notice)는 모두 "{target}"로 작성하세요.

목적: 한국어를 못 읽는 사람이 각 메뉴를 '이해'하도록 소개·맛·식재료·알레르기 주의를 안내합니다. (추천/필터링은 하지 않습니다.)
다음 JSON 스키마로만 응답(순수 JSON만):
{SCHEMA}

규칙:
- 다음은 결과에서 반드시 제외한다(요리가 아님): 섹션 헤더, 영업시간·주소·안내문, 원산지 표시, 가격.
- 가게 상호·간판 문구는 요리가 아니므로 제외한다: 메뉴판에서 가장 크게 강조된 제목, '24시·원조·전통·명가·맛집·본점' 같은 홍보 수식어가 붙은 가게 이름, 브랜드명. 특히 어떤 항목이 실제 메뉴명(예: 뼈해장국)에 홍보 수식어(얼큰한·24시·원조 등)를 덧붙인 '간판 제목'(예: 24시 얼큰한 뼈 해장국)으로 보이면, 그것은 상호이므로 별도 요리로 만들지 말고 제외한다. 실제 주문하는 메뉴명만 남긴다.
- 메뉴명 앞뒤에 섹션 라벨(곁들임·세트·음료·주류 등)이나 안내 문구(포장가능·포장·별도·1인분·2인이상 등)가 붙어 인식될 수 있다. 이럴 땐 그 부분만 떼어내고 '순수 요리명'을 original로 삼되, 라벨·문구가 붙었다는 이유로 그 요리를 절대 누락하지 마라(예: "곁한우육회김밥포장가능" → 한우육회김밥, "트간장찜닭" → 간장찜닭).
- 주류와 음료는 제외한다: 소주, 맥주, 막걸리, 청하, 하이볼, 와인, 사케 등 술 종류와,
  콜라·사이다·주스·음료수 등 마시는 음료는 items에 넣지 않는다.
- 한국어 OCR은 음절이 쪼개지거나(예: "공 기 밥"→"공기밥") 합쳐지거나 글자가 빠질 수 있다(예: 된장찌게→된장찌개). 문맥상 명백한 한식이면 연속된 조각을 한 메뉴명으로 합치고 "original"을 올바른 한글 철자로 복원한다. 불명확하면 추측하지 않는다(환각 금지).
- 글자가 잘리거나 일부만 인식되어 이름이 불완전하면 추측으로 완성하지 말고 제외한다(환각 금지).
- 확실히 식별되는 실제 요리는 하나도 빠짐없이 모두 items에 포함한다(개수를 임의로 줄이지 말 것).
- 각 요리는 보통 [짧은 요리명] + [그 아래 재료·소스·조리 설명 줄]로 구성된다. 재료 나열이나 소스·조리법을 풀어 쓴 '설명 문장'은 독립 요리가 아니라 바로 앞 요리명의 설명이므로 그 요리(intro/taste/ingredients)에 합친다.
- 단, 사리·공기밥·볶음밥·추가·세트처럼 그 자체로 '주문 가능한 항목'(대개 옆에 가격이 붙어 있음)은 곁들임처럼 보여도 독립 items로 반드시 포함한다. 예: 수제비사리, 만두사리, 라면사리, 공기밥, 볶음밥, 뼈추가, ○○세트. 이런 사리·추가류는 intro에 "무엇에 곁들여/추가로 넣어 먹는지"를 한 줄로 설명한다(예: 라면사리 = 전골·찌개 국물에 추가로 넣어 먹는 인스턴트 면).
- items는 메뉴판에 적힌 순서(위→아래)대로 배열한다.
- [왕초보 배려] 읽는 사람은 한국어·한식을 전혀 모른다. 처음 나오는 한국어 음식 용어는 반드시 목표언어로 짧게 풀이한다 — 재료뿐 아니라 요리 유형 용어도:
    · 칼국수 = knife-cut noodles, 국수 = noodles, 냉면 = cold noodles, 만두 = dumplings
    · 찌개 = stew, 전골 = hot pot, 탕/국 = soup, 볶음 = stir-fry, 무침 = seasoned/tossed dish, 구이 = grilled dish, 조림 = braised dish, 사리 = add-in noodles for soup, 정식/백반 = set meal
- 한국 전통 장(醬)이 나오면 그 장이 무엇인지 '한 줄'로 반드시 풀이한다(등장할 때마다):
    · 된장(doenjang) = fermented soybean paste (savory, hearty)
    · 고추장(gochujang) = fermented red chili paste (spicy, slightly sweet)
    · 간장(ganjang) = fermented soy sauce (salty; contains soy)
    · 청국장(cheonggukjang) = strong-smelling fast-fermented soybean paste
    · kimchi 등 낯선 발효식품도 처음 나오면 간단히 풀이한다.
- intro와 taste는 각각 3~4문장으로, 그 요리가 '일반적으로' 어떤 형태(국물/볶음/구이/전골 등)로 나오고 어떤 식감·간(짠맛·단맛·매운맛)·먹는 방법인지 명확하고 쉽게 묘사한다. 왕초보도 식욕이 생기도록 정확하고 담백하게 쓰되, 'earthy/흙내', 'muddy' 같은 모호하거나 부정적으로 들리는 풍미어는 쓰지 않는다.
- 위 예시(용어·장류)는 설명을 돕기 위한 것일 뿐, 특정 메뉴판이 아니라 '어떤 한식 메뉴판'에도 동일하게 적용한다.
- 알레르기는 단정하지 않는다. 반응 가능성이 있는 '의심 재료'만 allergen_caution에 짚는다. 특히 한식의 '숨은 알레르겐'에 주의: 간장·된장 = 대두(soy), 참기름 = 참깨(sesame), 그 외 계란·갑각류·밀·견과·메밀이 자주 쓰인다.
- notice에는 항상 "정확한 재료·알레르기 여부는 식당에 직접 문의하세요." 취지의 안내를 목표언어로 넣는다.
- 특정 조건으로 메뉴를 고르거나 추천하지 않는다(해석·안내만). 가격은 다루지 않는다.
- ingredients는 그 요리에 '일반적으로 흔히' 쓰이는 대표 재료를 구체적 명칭으로 3~8개 나열한다("vegetables"·"sauce" 같은 뭉뚱그림 금지, 예: napa cabbage, bean sprouts, sesame oil). 단 이 그릇에 확실히 들었다고 단정하지 말고, 이름에서 드러나지 않는 재료는 넣지 않는다.
- spicy_level은 0~5 사이 정수이며, 아래 한식 기준으로 판단한다:
    0 = 안 매움 (예: 잡채, 계란찜, 고추장 없는 비빔밥)
    1 = 아주 약함 (예: 김치볶음밥, 순한 떡볶이)
    2 = 약간 매움 (예: 순두부찌개 순한맛, 순한 제육)
    3 = 보통 매움 (예: 김치찌개, 일반 떡볶이)
    4 = 매움 (예: 낙지볶음, 순한 불닭)
    5 = 아주 매움 (예: 불닭, 아주 매운 찜닭, 청양고추 많은 요리)

메뉴 목록:
{block}
"""


def looks_like_name(t):
    t = t.strip()
    # 한글(가-힣) 또는 라틴 문자가 있으면 메뉴명 후보. 한글은 2글자(소주·전복 등)도 허용.
    return len(t) >= 2 and bool(re.search(r"[A-Za-zÀ-ÿ가-힣]", t))


_DIGIT_RE = re.compile(r"\d")
_CONTENT_RE = re.compile(r"[0-9A-Za-z가-힣]")
_PRICE_TAIL_RE = re.compile(r"[\s()大中]*\d[\d.,]*\s*원?\s*\)?$")   # 박스 끝의 가격 토큰


def _name_part(t):
    """박스 텍스트에서 끝의 가격을 떼고 메뉴명 부분만 남긴다 (예: '대()30,000' → '대')."""
    t = _PRICE_TAIL_RE.sub("", t).strip()
    t = re.sub(r"(포장\s*가능|포장|별도)$", "", t).strip()   # 안내 문구 제거(…김밥포장가능 → …김밥)
    return t.strip("·-,.， \t")   # 괄호는 보존(뼈추가(1인분) 같은 이름 깨짐 방지)


def merge_line_fragments(lines, gap_ratio=1.2):
    """OCR (text, score, box) 목록을 '줄' 단위로 묶고, 한 줄에서 가로로 붙은
    한글 조각(예: 공/기/밥 → 공기밥)을 한 메뉴명으로 병합한다.

    - 메뉴판 디자인상 글자 사이 띄어쓰기로 인해 한 메뉴가 여러 박스로 쪼개지는 문제 대응.
    - 숫자(가격) 포함 박스는 '병합 장벽'으로 두어 이름과 가격이 붙지 않게 한다
      (예: '뼈전골'과 바로 옆 '*38000'이 오병합되는 것 방지).
    - 순수 기호(-, · 등) 박스는 제외. 박스 좌표가 없는 항목은 그대로 통과.
    """
    boxed, plain = [], []
    for t, s, b in lines:
        t = (t or "").strip()
        if not t or not _CONTENT_RE.search(t):     # 순수 기호/빈 줄 제외
            continue
        if b is None:
            plain.append(t)
            continue
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        boxed.append({"t": t, "x0": min(xs), "x1": max(xs),
                      "y0": min(ys), "y1": max(ys), "h": max(ys) - min(ys)})

    # 1) 같은 '줄' 묶기 = 세로(y) 구간이 실제로 겹치는 박스만.
    #    (yc 거리 방식은 세로로 쌓인 다른 메뉴 - 예: 해삼/멍게 - 를 한 줄로 오인해 붙이므로,
    #     줄의 기준 박스와의 y겹침 비율로 판정하고 누적 드리프트를 막는다.)
    boxed.sort(key=lambda d: (d["y0"], d["x0"]))
    rows = []
    for d in boxed:
        for row in rows:
            a = row["anchor"]
            overlap = min(d["y1"], a["y1"]) - max(d["y0"], a["y0"])
            # 큰 박스(제목) 기준 절반 이상 겹쳐야 같은 줄 → 키 큰 간판이 작은 메뉴를 흡수하지 않음
            if overlap >= 0.5 * max(1, d["h"], a["h"]):
                row["items"].append(d)
                break
        else:
            rows.append({"anchor": d, "items": [d]})

    # 2) 간판/상호(맨 위의 유독 큰 글자) 줄 제외 준비: 높이 중앙값 + 상단 영역
    all_h = sorted(d["h"] for d in boxed) or [1]
    med_h = all_h[len(all_h) // 2]
    y_top = min((d["y0"] for d in boxed), default=0)
    y_bot = max((d["y1"] for d in boxed), default=1)
    y_span = max(1, y_bot - y_top)

    # 3) 각 줄에서 x 정렬 후, 숫자 아닌 한글 조각을 가로 간격이 좁으면 병합
    names = []
    for row in sorted(rows, key=lambda r: r["anchor"]["y0"]):
        row_h = max(d["h"] for d in row["items"])
        row_y0 = min(d["y0"] for d in row["items"])
        # 상단 25% 안 + 글자 높이가 중앙값의 1.7배↑ = 가게 상호/간판 → 제외
        # (6.jpg처럼 큰 제목이 없어 높이가 고른 메뉴판은 아무것도 안 지움)
        if row_h >= 1.7 * med_h and (row_y0 - y_top) <= 0.25 * y_span:
            continue
        cur, cur_x1, cur_h = "", None, None
        for d in sorted(row["items"], key=lambda d: d["x0"]):
            np = _name_part(d["t"])
            if _DIGIT_RE.search(d["t"]):
                # 가격(숫자) 박스 = 그 메뉴명의 '종료 지점'. 가격 앞에 붙은 한글(대/듬/기)은
                # 바로 앞까지 모은 이름에 흡수하고, 이름을 확정(flush)한다.
                if np:
                    cur = cur + np
                if cur:
                    names.append(cur)
                cur, cur_x1, cur_h = "", None, None
                continue
            if not np:                              # 순수 기호/괄호 등
                continue
            if not cur:
                cur, cur_x1, cur_h = np, d["x1"], d["h"]
            elif d["x0"] - cur_x1 <= gap_ratio * max(cur_h, d["h"], 1):
                cur += np; cur_x1 = d["x1"]; cur_h = max(cur_h, d["h"])
            else:                                   # 같은 줄이지만 멀리 떨어진 별개 항목
                names.append(cur)
                cur, cur_x1, cur_h = np, d["x1"], d["h"]
        if cur:
            names.append(cur)
    return names + plain


def _imread_kr(p):
    import numpy as np, cv2
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def tesseract_ocr(image_path, lang="fra"):
    """CLAHE 전처리(그레이+업스케일+대비) 후 Tesseract 인식 → 텍스트 줄 리스트.
    실험 결과: 원본 대비 인식단어 +47%, 악센트 +62% (PaddleOCR보다 악센트 우수)."""
    import cv2, pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    bgr = _imread_kr(image_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 2000:                                    # 작으면 업스케일
        s = 2000 / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)   # 대비 향상
    txt = pytesseract.image_to_string(Image.fromarray(gray), lang=lang)
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]


def call_llm(prompt, provider, model, temperature=0.3, retries=3):
    """일시적 서버 오류(503/과부하)는 짧게 대기 후 자동 재시도. 하드 쿼터(429)는 즉시 실패."""
    import time
    last = None
    for i in range(retries):
        try:
            return _call_llm_once(prompt, provider, model, temperature)
        except Exception as e:
            last, msg = e, str(e)
            transient = any(k in msg for k in ("503", "UNAVAILABLE", "overloaded", "high demand"))
            if transient and i < retries - 1:
                time.sleep(1.5 * (i + 1))
                continue
            raise
    raise last


def _call_llm_once(prompt, provider, model, temperature=0.3):
    """LLM API 호출 → 응답 텍스트(JSON 문자열) 반환. 키는 환경변수에서 읽음."""
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()                       # OPENAI_API_KEY 환경변수 사용
        r = client.chat.completions.create(
            model=model, temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "You output only valid JSON."},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()          # ANTHROPIC_API_KEY 환경변수 사용
        r = client.messages.create(model=model, max_tokens=4096, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    else:  # gemini
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        r = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, response_mime_type="application/json"))
        return r.text


def ping_llm(provider, model):
    """모델 사용 가능 여부 확인용 최소 호출. 성공하면 True, 실패하면 예외를 던진다.
    (OCR 같은 무거운 단계 전에 LLM 가용성을 먼저 확인해 헛대기를 막는 용도)"""
    _call_llm_once("Return {}", provider, model, 0.0)
    return True


LLM_CACHE_PATH = ROOT / "data" / "llm_cache.json"


def _load_cache():
    try:
        return json.loads(LLM_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_key(provider, model, prompt):
    # 프롬프트(=OCR결과+규칙+목표언어)+모델이 같으면 같은 해석 → 재호출 불필요
    return hashlib.sha256(f"{provider}|{model}|{prompt}".encode("utf-8")).hexdigest()


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`"); text = text[text.find("{"):]
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1])


def print_items(data):
    for i, it in enumerate(data.get("items", []), 1):
        print(f"\n[{i}] {it.get('original','')}  →  {it.get('translation','')}")
        if it.get("intro"): print(f"    소개 : {it['intro']}")
        ing = ", ".join(it.get("ingredients", []) or [])
        if ing: print(f"    재료 : {ing}")
        if it.get("cooking_method"): print(f"    조리법 : {it['cooking_method']}")
        if it.get("taste"): print(f"    맛   : {it['taste']}  (맵기 {it.get('spicy_level', 0)}/5)")
        cau = ", ".join(it.get("allergen_caution", []) or [])
        if cau: print(f"    알레르기 주의 : {cau} (의심 재료)")
    if data.get("notice"): print(f"\n※ {data['notice']}")


def main():
    ap = argparse.ArgumentParser(description="메뉴판 이미지 → LLM 해석·가이드")
    ap.add_argument("image", help="메뉴판 이미지 경로")
    ap.add_argument("--lang", default="korean", help="OCR 언어. 기본 korean")
    ap.add_argument("--target", default="English", help="번역 목표 언어 (나중에 확장)")
    ap.add_argument("--tlang", default="kor", help="Tesseract 언어 (kor/eng)")
    ap.add_argument("--llm", choices=["openai", "anthropic", "gemini"], help="지정하면 API로 해석까지 자동 실행")
    ap.add_argument("--model", help="LLM 모델명 (기본: openai=gpt-4o-mini / anthropic=claude-3-5-sonnet-latest / gemini=gemini-2.5-flash)")
    ap.add_argument("--ocr", choices=["paddle", "tesseract"], default="paddle",
                    help="OCR 엔진. tesseract = CLAHE 전처리+Tesseract (악센트·인식 개선)")
    ap.add_argument("--refresh", action="store_true", help="캐시 무시하고 LLM을 강제로 다시 호출")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print("이미지를 찾을 수 없습니다:", args.image); return 2

    print("[촬영 안내] 메뉴판은 '한 면(페이지)'만 프레임에 가득 담아 촬영하세요.")
    print("           옆 페이지가 함께 찍히면 잘린 글자를 잘못 인식할 수 있습니다.\n")

    print(f"OCR 실행 중... (engine={args.ocr}, {args.image})")
    if args.ocr == "tesseract":
        raw = tesseract_ocr(args.image, args.tlang)                    # CLAHE 전처리 + Tesseract
    else:
        ocr = run_ocr.get_engine(args.lang, "PP-OCRv5")               # v5 = 한국어 인식 정확도 우수
        lines = run_ocr.run_image(ocr, args.image)                     # [(text, score, box), ...]
        raw = merge_line_fragments(lines)                              # 쪼개진 한글 조각 병합

    names, seen = [], set()
    for t in raw:
        t = (t or "").strip()
        t = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", t)   # 한글 사이 띄어쓰기 제거(공 기 밥→공기밥)
        if not looks_like_name(t) or t.lower() in seen:
            continue
        seen.add(t.lower()); names.append(t)
    print(f"인식된 텍스트: {len(names)}개\n")

    prompt = build_prompt(names, args.target)
    (ROOT / "data" / "generated_prompt.txt").write_text(prompt, encoding="utf-8")

    # --llm 미지정: 프롬프트만 출력(붙여넣기용)
    if not args.llm:
        print("=" * 70); print(prompt); print("=" * 70)
        print("저장: data/generated_prompt.txt  → 복사해 ChatGPT/Claude에 붙여넣기")
        print("(자동 해석까지 하려면:  --llm openai --model gpt-4o)")
        return 0

    # --llm 지정: API로 해석까지 실행 (단, 같은 입력이면 캐시 사용)
    defaults = {"openai": "gpt-4o-mini", "anthropic": "claude-3-5-sonnet-latest", "gemini": "gemini-2.5-flash"}
    model = args.model or defaults[args.llm]
    cache = _load_cache()
    key = _cache_key(args.llm, model, prompt)
    if key in cache and not args.refresh:
        print("캐시된 해석 사용 (동일 입력 → API 호출 생략). 강제 재호출: --refresh")
        data = cache[key]
    else:
        print(f"LLM 해석 중... (provider={args.llm}, model={model})")
        try:
            data = parse_json(call_llm(prompt, args.llm, model))
        except Exception as e:
            print("LLM 호출 실패:", e)
            print("→ API 키 환경변수와 패키지 설치를 확인하세요.")
            return 1
        cache[key] = data
        LLM_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print_items(data)
    (ROOT / "data" / "explained.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장: data/explained.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
