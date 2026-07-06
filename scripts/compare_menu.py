# -*- coding: utf-8 -*-
"""
compare_menu.py  —  메뉴 인식 결과 '진단' 도구 (오류사례 분석용)

정답 목록(사람이 원본 메뉴판에서 직접 적은 것)과
파이프라인 출력(explained.json)을 대조해서
누락 / 오인식 / 합쳐짐 / 과분할을 표로 짚어준다.

※ 이 도구는 '진단'만 한다. 자동으로 고쳐주지 않는다.
   (어디가 틀렸는지 찾는 것까지 = 해커톤 '오류사례 분석' 재료)

사용법:
  python compare_menu.py --truth truth.txt --output data/explained.json
  python compare_menu.py --truth truth.txt --output data/explained.json --name-key original --out-csv result.csv

- truth.txt : 한 줄에 메뉴 하나 (원본 메뉴판의 실제 메뉴명)
- explained.json : 파이프라인이 저장한 결과 (메뉴 리스트)
"""

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher

# ── 판정 기준값 (필요하면 조정) ──
MATCH_TH = 0.85   # 이 이상이면 '같은 메뉴'로 봄
LOW_TH   = 0.55   # 이 미만이면 '매칭 실패'(누락 후보)


# ── 문자열 정규화: 대소문자/악센트/기호 차이를 없애 비교를 안정화 ──
def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def norm(s: str) -> str:
    s = strip_accents(str(s)).lower()
    s = unicodedata.normalize("NFC", s)          # 한글 자모 재결합(strip_accents의 NFKD 분해 복원)
    s = re.sub(r"[^a-z0-9가-힣]+", " ", s)        # 알파벳/숫자/한글 외에는 공백으로
    return re.sub(r"\s+", " ", s).strip()

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()

def word_diff(truth: str, out: str):
    """매칭된 두 메뉴명에서 서로 다른 단어를 찾아준다 (오인식 탐지용)."""
    t_words = norm(truth).split()
    o_words = norm(out).split()
    only_truth = [w for w in t_words if w not in o_words]   # 정답엔 있는데 출력엔 없음
    only_out   = [w for w in o_words if w not in t_words]   # 출력엔 있는데 정답엔 없음
    return only_truth, only_out


# ── 입력 읽기 ──
def load_truth(path: str):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def get_name(item, name_key=None):
    """출력 JSON의 각 항목에서 '메뉴 원문명'을 최대한 찾아낸다."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if name_key and name_key in item:
            return str(item[name_key])
        for k in ("original", "name", "menu", "원문", "메뉴", "title", "source", "fr"):
            if k in item and item[k]:
                return str(item[k])
        # 마지막 수단: 가장 긴 문자열 값
        strs = [v for v in item.values() if isinstance(v, str)]
        if strs:
            return max(strs, key=len)
    return str(item)

def load_output(path: str, name_key=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # 리스트이거나, {"items":[...]} / {"menus":[...]} 형태 모두 대응
    if isinstance(data, dict):
        for k in ("items", "menus", "results", "data", "메뉴"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    if not isinstance(data, list):
        raise ValueError("출력 JSON에서 메뉴 리스트를 찾지 못했습니다. --name-key 로 지정해 보세요.")
    return [get_name(it, name_key) for it in data]


# ── 핵심 대조 로직 ──
def compare(truth_list, out_list):
    n_out = len(out_list)

    # 각 정답 메뉴에 대해 가장 비슷한 출력 항목 찾기
    rows = []
    best_out_for_truth = []       # (out_idx, score)
    for ti, t in enumerate(truth_list):
        best_i, best_s = -1, 0.0
        for oi, o in enumerate(out_list):
            s = similarity(t, o)
            if s > best_s:
                best_i, best_s = oi, s
        best_out_for_truth.append((best_i, best_s))

    # 하나의 출력 항목에 몰린 정답이 2개 이상이면 '합쳐짐' 후보
    from collections import defaultdict
    claim = defaultdict(list)
    for ti, (oi, s) in enumerate(best_out_for_truth):
        if s >= LOW_TH:
            claim[oi].append(ti)
    merged_out = {oi for oi, tis in claim.items() if len(tis) >= 2}

    matched_out_idx = set()
    for ti, t in enumerate(truth_list):
        oi, s = best_out_for_truth[ti]
        if s < LOW_TH:
            status = "❌ 누락"
            matched = ""
            detail = "정답에 있으나 출력에서 찾지 못함"
        else:
            matched_out_idx.add(oi)
            matched = out_list[oi]
            only_t, only_o = word_diff(t, matched)
            if oi in merged_out:
                status = "🔀 합쳐짐"
                others = [truth_list[x] for x in claim[oi] if x != ti]
                detail = "다음과 한 항목으로 합쳐진 듯: " + " / ".join(others)
            elif s >= MATCH_TH and not only_t and not only_o:
                status = "✅ 일치"
                detail = ""
            else:
                status = "⚠️ 오인식/부분오류"
                bits = []
                if only_t:
                    bits.append("빠진 단어: " + ", ".join(only_t))
                if only_o:
                    bits.append("잘못 들어간 단어: " + ", ".join(only_o))
                detail = " · ".join(bits) if bits else f"유사도 {s:.2f}"
        rows.append({
            "번호": ti + 1,
            "정답 메뉴": t,
            "상태": status,
            "매칭된 출력": matched,
            "유사도": round(s, 2),
            "세부": detail,
        })

    # 아무 정답과도 매칭 안 된 출력 = 과분할/추가(허위) 후보
    extra_rows = []
    for oi, o in enumerate(out_list):
        if oi not in matched_out_idx:
            extra_rows.append({
                "출력번호": oi + 1,
                "출력 메뉴": o,
                "상태": "➕ 추가/과분할",
                "세부": "정답에 대응 항목 없음 (한 메뉴가 쪼개졌거나 잘못 생성)",
            })

    return rows, extra_rows


def print_table(rows, columns):
    """pandas 있으면 예쁘게, 없으면 단순 출력."""
    try:
        import pandas as pd
        pd.set_option("display.max_colwidth", 40)
        pd.set_option("display.width", 200)
        df = pd.DataFrame(rows, columns=columns)
        print(df.to_string(index=False))
        return df
    except Exception:
        for r in rows:
            print(" | ".join(f"{c}:{r.get(c,'')}" for c in columns))
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="정답 목록 파일 (한 줄에 메뉴 하나)")
    ap.add_argument("--output", required=True, help="파이프라인 출력 JSON (explained.json)")
    ap.add_argument("--name-key", default=None, help="JSON에서 메뉴명이 담긴 키 이름 (예: original)")
    ap.add_argument("--out-csv", default=None, help="결과를 CSV로 저장할 경로")
    args = ap.parse_args()

    truth_list = load_truth(args.truth)
    out_list = load_output(args.output, args.name_key)

    rows, extra_rows = compare(truth_list, out_list)

    # ── 요약 ──
    from collections import Counter
    cnt = Counter(r["상태"] for r in rows)
    print("=" * 70)
    print(f"정답 메뉴 수 : {len(truth_list)}    /    출력 메뉴 수 : {len(out_list)}")
    print("-" * 70)
    print(f"  ✅ 일치            : {cnt.get('✅ 일치', 0)}")
    print(f"  ⚠️ 오인식/부분오류 : {cnt.get('⚠️ 오인식/부분오류', 0)}")
    print(f"  ❌ 누락            : {cnt.get('❌ 누락', 0)}")
    print(f"  🔀 합쳐짐          : {cnt.get('🔀 합쳐짐', 0)}")
    print(f"  ➕ 추가/과분할     : {len(extra_rows)}")
    print("=" * 70)

    print("\n[정답 기준 대조표]")
    df = print_table(rows, ["번호", "정답 메뉴", "상태", "매칭된 출력", "유사도", "세부"])

    if extra_rows:
        print("\n[출력에만 있는 항목 (추가/과분할)]")
        print_table(extra_rows, ["출력번호", "출력 메뉴", "상태", "세부"])

    if args.out_csv:
        try:
            import pandas as pd
            cols = ["번호", "정답 메뉴", "상태", "매칭된 출력", "유사도", "세부"]
            all_rows = list(rows)
            for er in extra_rows:                       # ➕ 추가/과분할(허위)도 같은 표에 포함
                all_rows.append({
                    "번호": f"+{er['출력번호']}",
                    "정답 메뉴": "",
                    "상태": er["상태"],
                    "매칭된 출력": er["출력 메뉴"],
                    "유사도": "",
                    "세부": er["세부"],
                })
            pd.DataFrame(all_rows, columns=cols).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
            print(f"\n결과 저장: {args.out_csv}  (허위/과분할 {len(extra_rows)}건 포함)")
        except Exception as e:
            print(f"\nCSV 저장 실패: {e}")


if __name__ == "__main__":
    main()
