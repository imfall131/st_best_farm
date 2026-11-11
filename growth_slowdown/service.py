"""Local API and JSON command-line entry point for indoor environment advice."""

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from math import isfinite
from pathlib import Path
import sys

if __package__:
    from .advice import CONTROL_BANDS, SENSOR_LIMITS, advise_after_observation
else:
    from advice import CONTROL_BANDS, SENSOR_LIMITS, advise_after_observation


LEGACY = Path('C:/Users/김가을/Documents/Codex/2026-11-02/files-mentioned-by-the-user-demo')
BASE_MODEL = LEGACY / 'outputs/conservative_candidates_20261103_v1/raw_blended.joblib'
FORECAST_SOURCE = LEGACY / 'work/advisory_prototype/core.py'
KST = timezone(timedelta(hours=9))
SENSOR_ALIASES = {'ei_value': 'nutrient_ec', 'pl_value': 'nutrient_ph'}
# The 80th percentile of signed H3-S20 gaps on the 116 development cases.
GROWTH_ALERT_PP = 5.404442266470099


def load_forecast_core():
    spec = spec_from_file_location('strawberry_forecast_core', FORECAST_SOURCE)
    core = module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def environment_records(payload, group_id):
    if 'environment_csv' in payload and 'environment_rows' in payload:
        raise ValueError('센서 CSV 업로드와 직접 입력 중 하나를 선택해 주세요.')
    if 'environment_csv' in payload:
        raw = csv.DictReader(io.StringIO(payload['environment_csv'].lstrip('\ufeff')))
        fields = set(raw.fieldnames or [])
        if 'timestamp' not in fields or not fields.intersection(SENSOR_LIMITS.keys() | SENSOR_ALIASES.keys()):
            raise ValueError('CSV에 timestamp와 온도·습도·CO₂·양액 EC·pH 센서 열 중 하나 이상이 필요합니다.')
    else:
        raw = payload.get('environment_rows', [])

    parsed = {}
    for row in raw:
        if row.get('group_id') not in (None, '', group_id):
            raise ValueError('선택한 재배구역의 센서 데이터만 입력해 주세요.')
        timestamp = datetime.fromisoformat(str(row['timestamp']).strip())
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=KST)
        timestamp = timestamp.astimezone(KST)
        phase = str(row.get('light_phase') or '').strip().lower()
        phase = {'명기': 'light', '암기': 'dark', '': 'unknown'}.get(phase, phase)
        if phase not in ('light', 'dark', 'unknown'):
            raise ValueError('점등 상태는 명기·암기 또는 공란으로 입력해 주세요.')
        normalized = {'light_phase': phase if phase != 'unknown' else None}
        for field, value in row.items():
            sensor = SENSOR_ALIASES.get(field, field)
            if sensor not in SENSOR_LIMITS or value is None or not str(value).strip():
                continue
            lo, hi = SENSOR_LIMITS[sensor]
            number = float(value)
            if not isfinite(number) or not lo <= number <= hi:
                continue
            if normalized.setdefault(sensor, number) != number:
                raise ValueError('한 행에 같은 센서의 서로 다른 값이 있습니다.')
        record = parsed.setdefault(timestamp, {'timestamp': timestamp})
        for key, value in normalized.items():
            if value is None:
                continue
            if key in record and record[key] != value:
                raise ValueError('같은 시각에 서로 다른 센서값 또는 점등 상태가 있습니다.')
            record[key] = value
    return list(parsed.values())


def resolve_profile(payload, context, approved_profiles):
    profile_id = payload.get('profile_id')
    targets = payload.get('operating_targets')
    if 'reviewed_profile' in payload or (profile_id and targets is not None):
        raise ValueError('등록된 관리 기준과 농장 운영 목표 중 하나를 선택해 주세요.')
    if profile_id:
        profile = approved_profiles.get(profile_id)
        if profile is None:
            raise ValueError('등록되지 않은 관리 기준입니다.')
        return profile
    if targets is None:
        return None
    if targets.get('confirmed') is not True:
        raise ValueError('농장 운영 목표값을 확인해 주세요.')
    targets = dict(targets)
    if 'dark_temperature_lower_c' in targets:
        lower = targets.pop('dark_temperature_lower_c')
        if targets.setdefault('night_temperature_lower_c', lower) != lower:
            raise ValueError('암기 온도 하한을 한 가지 값으로 입력해 주세요.')
    keys = {key for *_, lower, upper in CONTROL_BANDS.values() for key in (lower, upper)} | {'nutrient_ec_unit'}
    unknown_keys = targets.keys() - keys - {'confirmed'}
    if unknown_keys:
        raise ValueError(f'알 수 없는 목표 항목입니다. {", ".join(sorted(unknown_keys))}')
    profile = {key: context.get(key) for key in ('facility', 'cultivar', 'growth_stage')}
    return dict(profile, review_status='reviewed', evidence_kind='farm_operating_target',
                source_url='local:farm-operating-target', **{key: targets[key] for key in targets.keys() & keys})


def recommend(payload, approved_profiles=None):
    context = payload['context']
    group_id = context.get('group_id')
    if not group_id or context.get('facility') != 'indoor_vertical':
        raise ValueError('실내 수직농장의 농가·재배구역을 선택해 주세요.')
    start_date = date.fromisoformat(payload['forecast_as_of'])
    actual_date = date.fromisoformat(payload['observation_date'])
    if (actual_date - start_date).days != 7:
        raise ValueError('예측 기준일에서 정확히 7일 뒤의 관측을 사용해 주세요.')
    unit = payload['unit']
    measurements = payload['measurements']
    paired = {start_date: {}, actual_date: {}}
    for row in measurements:
        if row.get('group_id', group_id) != group_id or row.get('unit', unit) != unit:
            raise ValueError('같은 재배구역과 측정 단위의 생육 기록을 사용해 주세요.')
        day = date.fromisoformat(row['measured_date'])
        if day in paired:
            panel = paired[day]
            sid, value = str(row['series_id']), float(row['crown_diameter'])
            if sid in panel and panel[sid] != value:
                raise ValueError('같은 날짜·관리번호에 서로 다른 관부직경이 있습니다.')
            panel[sid] = value

    profile = resolve_profile(payload, context, approved_profiles or {})
    records = environment_records(payload, group_id)
    forecast = load_forecast_core().forecast_growth(measurements, start_date.isoformat(), unit=unit)
    result = advise_after_observation(forecast, paired[start_date], paired[actual_date],
                                     observation_date=actual_date.isoformat(), unit=unit, context=context,
                                     sensor_records=records, profile=profile, threshold_pct=GROWTH_ALERT_PP)
    return dict(group_id=group_id, forecast=forecast, observation_date=actual_date.isoformat(),
                **result, automatic_control=False, growth_gain_estimate=None)


def main():
    parser = argparse.ArgumentParser(description='관부직경 생육 경고 후 온도·습도·CO₂·양액 EC·pH 조정 권고')
    parser.add_argument('payload', help='입력 JSON 경로 또는 표준 입력을 뜻하는 -')
    parser.add_argument('--environment-csv', type=Path, help='해당 재배구역의 센서 CSV 경로')
    args = parser.parse_args()
    try:
        payload = (json.load(sys.stdin.buffer) if args.payload == '-'
                   else json.loads(Path(args.payload).read_text(encoding='utf-8-sig')))
        if args.environment_csv:
            if 'environment_csv' in payload:
                raise ValueError('센서 CSV는 한 곳에서만 입력해 주세요.')
            payload['environment_csv'] = args.environment_csv.read_text(encoding='utf-8-sig')
        result = json.dumps(recommend(payload), ensure_ascii=False, indent=2)
        sys.stdout.buffer.write((result + '\n').encode('utf-8'))
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
