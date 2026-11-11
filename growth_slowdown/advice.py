"""Offer manual environment adjustments after a completed H3-S20 forecast."""

from datetime import date
from math import isclose, isfinite
from statistics import mean, median


MIN_SENSOR_DAYS = 5
MIN_EXCURSION_DAYS = 2
NUTRIENT_SENSORS = ('nutrient_ec', 'nutrient_ph')
SENSOR_LIMITS = {'temperature_c': (-10, 55), 'humidity_pct': (0, 100), 'co2_ppm': (100, 5000),
                 'nutrient_ec': (0, float('inf')), 'nutrient_ph': (0, 14)}
CONTROL_BANDS = {
    'light_temperature': ('명기 온도', '°C', 'temperature_c', 'light_temperature_lower_c', 'light_temperature_upper_c'),
    'dark_temperature': ('암기 온도', '°C', 'temperature_c', 'night_temperature_lower_c', 'night_temperature_upper_c'),
    'humidity': ('상대습도', '%', 'humidity_pct', 'humidity_lower_pct', 'humidity_upper_pct'),
    'light_co2': ('명기 CO₂', 'ppm', 'co2_ppm', 'co2_light_lower_ppm', 'co2_light_upper_ppm'),
    'nutrient_ec': ('양액 EC', None, 'nutrient_ec', 'nutrient_ec_lower', 'nutrient_ec_upper'),
    'nutrient_ph': ('양액 pH', 'pH', 'nutrient_ph', 'nutrient_ph_lower', 'nutrient_ph_upper'),
}


def normalize_ec_unit(unit):
    if not isinstance(unit, str):
        return None
    unit = unit.strip().replace('μ', 'µ')
    if unit == 'uS/cm':
        unit = 'µS/cm'
    return unit if unit in ('µS/cm', 'mS/cm', 'dS/m') else None


def validated_bands(context, profile):
    if profile is None:
        return {}
    if (profile.get('review_status') != 'reviewed' or not profile.get('source_url')
            or profile.get('evidence_kind') not in {'management_guideline', 'farm_operating_target'}
            or any(not context.get(key) or profile.get(key) != context[key]
                   for key in ('facility', 'cultivar', 'growth_stage'))):
        raise ValueError('관리 기준의 출처와 시설·품종·생육단계를 확인해 주세요.')
    bands = {}
    for name, (label, _, sensor, lower_key, upper_key) in CONTROL_BANDS.items():
        lo, hi = profile.get(lower_key), profile.get(upper_key)
        limits = SENSOR_LIMITS[sensor]
        if any(value is not None and (type(value) not in (int, float) or not isfinite(value)
                                      or not limits[0] <= value <= limits[1])
               for value in (lo, hi)) or (lo is not None and hi is not None and lo > hi):
            raise ValueError(f'{label} 관리 기준의 범위와 단위를 확인해 주세요.')
        bands[name] = (lo, hi)
    return bands


def compare_band(name, daily, bounds, profile, *, ec_unit=None):
    label, unit, _, _, _ = CONTROL_BANDS[name]
    lower, upper = bounds
    if lower is None and upper is None:
        return 'needs_target', None
    if len(daily) < MIN_SENSOR_DAYS:
        return 'insufficient_sensor_days', None
    if name == 'nutrient_ec':
        if ec_unit is None or ec_unit != normalize_ec_unit(profile.get('nutrient_ec_unit')):
            return 'needs_matching_ec_unit', None
        unit = ec_unit
    low_stat, high_stat = (mean, mean) if name == 'light_co2' else (min, max)
    below = {day.isoformat(): low_stat(values) for day, values in sorted(daily.items())
             if lower is not None and low_stat(values) < lower}
    above = {day.isoformat(): high_stat(values) for day, values in sorted(daily.items())
             if upper is not None and high_stat(values) > upper}
    needs_raise = len(below) >= MIN_EXCURSION_DAYS
    needs_lower = len(above) >= MIN_EXCURSION_DAYS
    if not needs_raise and not needs_lower:
        return ('isolated_excursion' if below or above else 'within_profile'), None
    if name == 'light_co2' and needs_raise and upper is None:
        return 'needs_upper_target', None
    if name == 'light_co2' and needs_raise and above:
        needs_lower = True

    if needs_raise and needs_lower:
        direction, status = 'stabilize', 'varies_across_profile'
        message = f'{label}이 상·하한을 모두 벗어난 시간대의 센서와 제어 설정을 점검하고 변동 폭을 줄이세요.'
    elif needs_raise:
        direction, status = 'increase', 'below_profile'
        message = f'현재도 낮으면 {label}을 등록한 하한 쪽으로 올리는 방향을 검토하세요.'
    else:
        direction, status = 'decrease', 'above_profile'
        message = f'현재도 높으면 {label}을 등록한 상한 쪽으로 낮추는 방향을 검토하세요.'

    if name == 'humidity':
        equipment_check = '가습·제습 설정과 공기순환을 확인하세요.'
    elif name == 'light_co2':
        equipment_check = '점등 중 센서·공급 기록과 시설의 작업자 안전 기준을 확인한 뒤 승인된 운전 범위에서 조정하세요.'
    elif name in NUTRIENT_SENSORS:
        equipment_check = '해당 재배구역에 공급되는 양액을 다시 측정하고, 센서 보정 상태와 양액 혼합 설정을 확인하세요.'
    else:
        equipment_check = '해당 점등 상태에서 냉난방 설정과 작물 높이의 센서값을 확인하세요.'
    return status, dict(type=f'{name}_{status}', variable=name, label=label, unit=unit,
                        direction=direction, target=dict(lower=lower, upper=upper),
                        observed_days=len(daily), below_by_date=below, above_by_date=above,
                        statistic='light_phase_daily_mean' if name == 'light_co2' else 'daily_observed_extremes',
                        actions=[equipment_check, message], basis=profile['evidence_kind'],
                        source_url=profile['source_url'])


def assess_shortfall(start_values, actual_values, predicted, threshold_pct):
    """Hold the forecast growth rate fixed for the leave-one-series-out check."""
    if len(start_values) != len(actual_values) or len(start_values) < 3:
        raise ValueError('같은 관리번호 3개 이상의 시작·종료 측정값이 필요합니다.')
    if not all(isfinite(x) and x > 0 for x in (*start_values, *actual_values, predicted, threshold_pct)):
        raise ValueError('예측값·측정값·경고 기준은 양수여야 합니다.')
    expected_pct = 100 * (predicted / median(start_values) - 1)
    actual_pct = 100 * (median(actual_values) / median(start_values) - 1)
    shortfall = expected_pct - actual_pct
    gaps_without_one = [
        expected_pct - 100 * (median(actual_values[:i] + actual_values[i + 1:])
                              / median(start_values[:i] + start_values[i + 1:]) - 1)
        for i in range(len(start_values))
    ]
    if shortfall <= threshold_pct:
        status = 'within_expected_variation'
    elif min(gaps_without_one) <= threshold_pct:
        status = 'check_measurement_first'
    else:
        status = 'review_environment'
    return dict(status=status, expected_growth_pct=expected_pct, actual_growth_pct=actual_pct,
                shortfall_pp=shortfall, threshold_pp=threshold_pct,
                min_gap_without_one_pp=min(gaps_without_one))


def advise_after_observation(forecast, start_by_id, actual_by_id, *, observation_date,
                             unit, context, sensor_records, profile, threshold_pct):
    """Inputs are for the same zone and sensor timestamps are parsed in KST.

    The profile comes from the existing reviewed-profile or confirmed-farm-target
    path. Optional CO2 targets refer to mean light-phase readings per day.
    """
    if forecast.get('model') != 'H3-S20' or unit != forecast.get('unit') or not unit:
        raise ValueError('환경 보정 전 HGB 예측과 동일한 측정 단위를 사용해 주세요.')
    if context.get('facility') != 'indoor_vertical':
        raise ValueError('실내 수직농장용 점검 기능입니다.')
    bands = validated_bands(context, profile)
    start_day = date.fromisoformat(forecast['as_of'])
    end_day = date.fromisoformat(forecast['forecast_date'])
    if (end_day - start_day).days != 7 or date.fromisoformat(observation_date) != end_day:
        raise ValueError('예측 대상일의 7일 후속 관측을 사용해 주세요.')
    panel = forecast['panel_ids']
    if len(set(panel)) != len(panel) or set(panel) != set(start_by_id):
        raise ValueError('기준 예측을 만들 때 사용한 관리번호와 시작값을 확인해 주세요.')
    if not all(key in actual_by_id for key in panel):
        return dict(status='missing_followup_panel', recommendations=[])
    start_values = [float(start_by_id[key]) for key in panel]
    if not isclose(median(start_values), forecast['current_median'], rel_tol=0, abs_tol=1e-10):
        raise ValueError('예측 당시의 시작값을 사용해 주세요.')
    result = assess_shortfall(start_values, [float(actual_by_id[key]) for key in panel],
                              forecast['predicted_median'], threshold_pct)
    result['recommendations'] = []
    if result['status'] != 'review_environment':
        return result

    daily = {key: {} for key in SENSOR_LIMITS}
    by_control = {key: {} for key in CONTROL_BANDS}
    for row in sensor_records:
        day = row['timestamp'].date()
        if not start_day < day < end_day:
            continue
        phase = row.get('light_phase')
        for key, (lower, upper) in SENSOR_LIMITS.items():
            value = row.get(key)
            if value is not None and isfinite(value) and lower <= value <= upper:
                daily[key].setdefault(day, []).append(value)
                if key == 'temperature_c' and phase in ('light', 'dark'):
                    by_control[f'{phase}_temperature'].setdefault(day, []).append(value)
                if key == 'co2_ppm' and phase == 'light':
                    by_control['light_co2'].setdefault(day, []).append(value)

    by_control['humidity'] = daily['humidity_pct']
    for name in NUTRIENT_SENSORS:
        if daily[name] or any(value is not None for value in bands.get(name, ())):
            by_control[name] = daily[name]
        else:
            del by_control[name]
            del daily[name]
    ec_unit = normalize_ec_unit(context.get('nutrient_ec_unit'))
    if 'nutrient_ec' in by_control:
        result['nutrient_ec_unit'] = ec_unit
    days = {key: len(values) for key, values in daily.items()}
    result['sensor_days'] = days | {key: len(values) for key, values in by_control.items()}
    result['sensor_means'] = {key: mean(map(mean, values.values()))
                              for key, values in daily.items() if days[key] >= MIN_SENSOR_DAYS}
    result['checks'] = {}
    for name, values in by_control.items():
        status, card = compare_band(name, values, bands.get(name, (None, None)), profile, ec_unit=ec_unit)
        result['checks'][name] = status
        if card is not None:
            result['recommendations'].append(card)
    result['co2_check'] = result['checks']['light_co2']
    missing_input = {'needs_target': '관리 기준', 'needs_upper_target': '관리 상한',
                     'needs_matching_ec_unit': '센서와 관리 기준의 동일 단위 확인',
                     'insufficient_sensor_days': f'센서 기록 {MIN_SENSOR_DAYS}일 이상'}
    result['needed'] = [f'{CONTROL_BANDS[name][0]} {missing_input[status]}'
                        for name, status in result['checks'].items()
                        if status in missing_input]
    return result
