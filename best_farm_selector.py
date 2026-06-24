"""
best_farm_selector.py
생장 단계별 우수 농가 선별 모듈

- 1~4단계: 화방개수, 잎면적, 줄기면적을 min-max 정규화 후 평균 -> 종합점수 최고 농가
- 5단계: 수확량(yield_amount) 기준 최고 농가
- 결과: 단계별 최적 환경 조건(temperature, humidity, co2) 딕셔너리
"""

from typing import List, Dict, Any

GROWTH_INDICATOR_FIELDS = ["flower_count", "leaf_area", "stem_area"]
ENV_FIELDS = ["temperature", "humidity", "co2"]
HARVEST_STAGE = 5


def _min_max_normalize(values: List[float]) -> List[float]:
    """min-max 정규화. 값이 모두 같으면 0.5로 통일(분모 0 방지)."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _group_by_stage(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["growth_stage"], []).append(rec)
    return grouped


def select_best_farm_growth_stage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    1~4단계용: 화방개수/잎면적/줄기면적을 정규화 후 평균낸 종합점수가
    가장 높은 레코드(농가)를 선택한다.

    records: 동일 growth_stage에 속한 레코드 리스트 (여러 농가)
    return: 선택된 best record + composite_score 필드 추가
    """
    if not records:
        raise ValueError("선별할 레코드가 없습니다.")

    normalized_cols = {}
    for field in GROWTH_INDICATOR_FIELDS:
        values = [r[field] for r in records]
        normalized_cols[field] = _min_max_normalize(values)

    best_idx = None
    best_score = float("-inf")

    for i, rec in enumerate(records):
        score = sum(normalized_cols[field][i] for field in GROWTH_INDICATOR_FIELDS) / len(
            GROWTH_INDICATOR_FIELDS
        )
        if score > best_score:
            best_score = score
            best_idx = i

    best_record = dict(records[best_idx])
    best_record["composite_score"] = round(best_score, 4)
    return best_record


def select_best_farm_harvest_stage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """5단계(수확기)용: yield_amount(수확량)가 가장 높은 레코드(농가)를 선택한다."""
    if not records:
        raise ValueError("선별할 레코드가 없습니다.")
    return max(records, key=lambda r: r["yield_amount"])


def build_optimal_condition_table(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    전체 전처리된 레코드를 입력받아,
    생장단계(1~5)별 최적 환경 조건 테이블을 생성한다.

    return:
        {
          1: {"best_farm_id": ..., "temperature": ..., "humidity": ..., "co2": ..., "score_info": {...}},
          2: {...},
          ...
          5: {"best_farm_id": ..., "temperature": ..., ..., "yield_amount": ...},
        }
    """
    grouped = _group_by_stage(records)
    optimal_table: Dict[int, Dict[str, Any]] = {}

    for stage in sorted(grouped.keys()):
        stage_records = grouped[stage]

        if stage == HARVEST_STAGE:
            best = select_best_farm_harvest_stage(stage_records)
            optimal_table[stage] = {
                "best_farm_id": best["farm_id"],
                "temperature": best["temperature"],
                "humidity": best["humidity"],
                "co2": best["co2"],
                "selection_basis": "yield_amount",
                "yield_amount": best["yield_amount"],
            }
        else:
            best = select_best_farm_growth_stage(stage_records)
            optimal_table[stage] = {
                "best_farm_id": best["farm_id"],
                "temperature": best["temperature"],
                "humidity": best["humidity"],
                "co2": best["co2"],
                "selection_basis": "composite_score(flower_count, leaf_area, stem_area)",
                "composite_score": best["composite_score"],
                "flower_count": best["flower_count"],
                "leaf_area": best["leaf_area"],
                "stem_area": best["stem_area"],
            }

    return optimal_table


def print_optimal_table(optimal_table: Dict[int, Dict[str, Any]]) -> None:
    print("\n=== 생장단계별 최적 환경 조건 (우수 농가 기준) ===")
    for stage, info in optimal_table.items():
        print(f"\n[단계 {stage}] 최우수 농가: {info['best_farm_id']} (선별기준: {info['selection_basis']})")
        print(f"  - 온도: {info['temperature']}℃ / 습도: {info['humidity']}% / CO2: {info['co2']}ppm")
        if stage == HARVEST_STAGE:
            print(f"  - 수확량: {info['yield_amount']}g")
        else:
            print(f"  - 종합점수: {info['composite_score']} "
                  f"(화방개수: {info['flower_count']}, 잎면적: {info['leaf_area']}, 줄기면적: {info['stem_area']})")


if __name__ == "__main__":
    from preprocessing import preprocess_pipeline

    cleaned_records = preprocess_pipeline("/home/claude/strawberry_smartfarm/sample_farm_data.json")
    table = build_optimal_condition_table(cleaned_records)
    print_optimal_table(table)
