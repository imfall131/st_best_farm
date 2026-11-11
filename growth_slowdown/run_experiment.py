"""Replay observed seven-day gaps and test whether sensor values explain them."""

import hashlib
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from advice import MIN_SENSOR_DAYS, advise_after_observation
from service import BASE_MODEL, GROWTH_ALERT_PP, LEGACY, recommend


HERE = Path(__file__).resolve().parent
WORK = LEGACY / 'work'
SOURCE = LEGACY / 'outputs/kept_candidates_20261103_v1/source'
GROUP = ['farm_id', 'start_date', 'end_date']
FARM_AND_HISTORY = ['doy', 'expected_growth_pct', 'farm_id']
ENV = [f'inner_{name}_value_mean' for name in ('ti', 'hi', 'ci')]
ENV_COVERAGE = [name.replace('_mean', '_coverage') for name in ENV]
CALIBRATION_QUANTILE = 0.80


def read_evaluation():
    columns = ['row_id', 'farm_id', 'crops_id', 'kind_type', 'start_date', 'end_date', 'days',
               'doy', 'start_panel_ids', 'start', 'target', *ENV, *ENV_COVERAGE]
    cohort = pd.read_csv(WORK / 'concurrent_environment_analysis/features.csv', usecols=columns,
                         parse_dates=['start_date', 'end_date']).query('days == 7').copy()
    existing = WORK / 'shrink_normalize_trials'
    baseline_filter = 'algorithm == "HGB" and candidate_id == 3 and not normalized'
    oof = pd.read_csv(existing / 'development_oof_deltas.csv').query(baseline_filter).copy()
    oof['forecast'] = oof.start + 0.20 * oof.predicted_delta
    held = pd.read_csv(existing / 'old_test_predictions.csv').query(baseline_filter + ' and blended').copy()
    np.testing.assert_allclose(held.alpha, 0.20)
    held = held.assign(fold=4, forecast=held.prediction)
    predictions = pd.concat([oof, held])[['row_id', 'farm_id', 'fold', 'actual', 'start', 'forecast']]
    rows = cohort.merge(predictions, on=['row_id', 'farm_id', 'start'], validate='one_to_one')
    np.testing.assert_allclose(rows.actual, rows.target, atol=1e-12, rtol=0)
    rows['expected_growth_pct'] = 100 * (rows.forecast / rows.start - 1)
    rows['gap'] = 100 * (rows.forecast - rows.actual) / rows.start
    rows['sensor_complete'] = rows[ENV_COVERAGE].ge(MIN_SENSOR_DAYS / 6).all(axis=1)
    assert rows.loc[rows.sensor_complete, ENV].notna().all().all()
    assert rows.groupby('fold').size().tolist() == [23, 55, 38, 109]
    return rows


def environmental_comparison(train, test, threshold):
    """Identical rows in each arm, weight each farm/time window equally."""
    train = train.loc[train.sensor_complete].copy()
    test = test.loc[test.sensor_complete].copy()
    assert train.end_date.max() < test.start_date.min()
    weights = lambda frame: 1 / frame.groupby(GROUP).gap.transform('size')
    train_weights = weights(train)
    test_weights = weights(test)
    alert = test.gap > threshold
    result = dict(train_n=len(train), n=len(test), farm_windows=int(test_weights.sum().round()),
                  alert_n=int(alert.sum()), weighted_alert_rate=float(np.average(alert, weights=test_weights)),
                  scores={})
    for label, sensors in [('history_and_farm', []), ('plus_3_sensors', ENV)]:
        columns = FARM_AND_HISTORY + sensors
        transform = ColumnTransformer([
            ('scale', StandardScaler(), ['doy', 'expected_growth_pct', *sensors]),
            ('farm', OneHotEncoder(handle_unknown='ignore'), ['farm_id']),
        ])
        model = make_pipeline(transform, Ridge(alpha=10))
        model.fit(train[columns], train.gap, ridge__sample_weight=train_weights)
        prediction = model.predict(test[columns])
        result['scores'][label] = dict(
            gap_mae_pp=mean_absolute_error(test.gap, prediction, sample_weight=test_weights),
            alert_average_precision=average_precision_score(alert, prediction, sample_weight=test_weights),
            gap_mae_for_alerts_pp=mean_absolute_error(test.gap[alert], prediction[alert],
                                                    sample_weight=test_weights[alert]),
        )
    return result


def replay_held_alerts(held, threshold):
    sys.path.insert(0, str(WORK / 'advisory_prototype'))
    from core import forecast_growth
    from manual_inputs import KST

    series = pd.read_csv(SOURCE / 'series_day_values.csv', parse_dates=['date'], dtype={'series_id': str})
    series = series.loc[~series.ambiguous].copy()
    series['series_id'] = series.series_id.str.zfill(3)
    farm_group = ['farm_id', 'crops_id', 'kind_type']
    by_group = dict(tuple(series.groupby(farm_group)))
    source = LEGACY / 'outputs/corrected_recheck/environment_observations.csv'
    fields = {'ti_value': 'temperature_c', 'hi_value': 'humidity_pct', 'ci_value': 'co2_ppm'}
    sensors = pd.read_csv(source, usecols=['farm_id', 'receive_date', *fields], parse_dates=['receive_date'])
    sensors = sensors.rename(columns=fields)
    for field, (lo, hi) in zip(fields.values(), [(-10, 55), (0, 100), (100, 5000)]):
        values = pd.to_numeric(sensors[field], errors='coerce')
        sensors[field] = values.where(values.between(lo, hi))
    sensors = sensors.groupby(['farm_id', 'receive_date'], as_index=False)[list(fields.values())].median()
    sensors['timestamp'] = sensors.receive_date.dt.tz_localize(KST)
    by_farm = dict(tuple(sensors.groupby('farm_id')))
    counts = dict(initial_alert_n=0, check_measurement_first_n=0, review_environment_n=0,
                  complete_sensor_review_n=0, action_cards_without_profiles=0,
                  expected_positive_n=0, review_environment_expected_positive_n=0,
                  actual_increase_below_forecast_n=0, actual_increase_initial_alert_n=0, actual_increase_review_n=0,
                  actual_decrease_review_n=0)
    reviews = []
    demo = None
    for row in held.itertuples():
        panel = row.start_panel_ids.split('+')
        z = by_group[(row.farm_id, row.crops_id, row.kind_type)]
        unit = 'same recorded farm scale'
        start = z[z.date.eq(row.start_date) & z.series_id.isin(panel)].set_index('series_id').stem_value.to_dict()
        end = z[z.date.eq(row.end_date) & z.series_id.isin(panel)].set_index('series_id').stem_value.to_dict()
        history = z[z.date.le(row.start_date) & z.series_id.isin(panel)]
        measurements = [dict(measured_date=x.date.date().isoformat(), series_id=x.series_id,
                             crown_diameter=x.stem_value) for x in history.itertuples()]
        forecast = forecast_growth(measurements, row.start_date.date().isoformat(), unit=unit)
        np.testing.assert_allclose(forecast['predicted_median'], row.forecast, atol=1e-12, rtol=0)
        context = dict(facility='indoor_vertical', cultivar=row.kind_type)
        at = row.end_date.date().isoformat()
        farm = by_farm[row.farm_id]
        recs = farm[farm.receive_date.ge(row.start_date + pd.Timedelta(days=1))
                    & farm.receive_date.lt(row.end_date)][['timestamp', *fields.values()]]
        diagnosis = advise_after_observation(forecast, start, end, observation_date=at, unit=unit,
                                            context=context, sensor_records=recs.to_dict('records'),
                                            profile=None, threshold_pct=threshold)
        np.testing.assert_allclose(diagnosis['shortfall_pp'], row.gap, atol=1e-10, rtol=0)
        counts['initial_alert_n'] += int(row.gap > threshold)
        counts['actual_increase_below_forecast_n'] += int(row.actual > row.start and row.gap > 0)
        counts['actual_increase_initial_alert_n'] += int(row.actual > row.start and row.gap > threshold)
        counts['expected_positive_n'] += int(row.expected_growth_pct > 0)
        counts['check_measurement_first_n'] += int(diagnosis['status'] == 'check_measurement_first')
        if diagnosis['status'] != 'review_environment':
            continue
        counts['review_environment_n'] += 1
        counts['review_environment_expected_positive_n'] += int(row.expected_growth_pct > 0)
        counts['actual_increase_review_n'] += int(row.actual > row.start)
        counts['actual_decrease_review_n'] += int(row.actual < row.start)
        counts['complete_sensor_review_n'] += int(row.sensor_complete)
        assert (len(diagnosis['sensor_means']) == 3) == row.sensor_complete
        counts['action_cards_without_profiles'] += len(diagnosis['recommendations'])
        for actual, name in zip(fields.values(), ENV):
            if actual in diagnosis['sensor_means']:
                np.testing.assert_allclose(diagnosis['sensor_means'][actual], getattr(row, name), atol=1e-10, rtol=0)
        reviews.append(dict(farm_id=row.farm_id, row_id=row.row_id, end_date=at, **diagnosis))
        if demo is None:
            followup = [dict(measured_date=at, series_id=sid, crown_diameter=value) for sid, value in end.items()]
            demo = (measurements + followup, row.start_date, at, unit, context)

    assert counts['initial_alert_n'] == counts['check_measurement_first_n'] + counts['review_environment_n']
    assert counts['action_cards_without_profiles'] == 0 and demo is not None
    verification = verify_service(*demo)
    return dict(**counts, review_windows=len({(row['farm_id'], row['end_date']) for row in reviews}),
                reviews=reviews, synthetic_dispatch_verified=True, service_verification=verification)


def verify_service(measurements, start_date, at, unit, context):
    context = context | dict(group_id='test-only', growth_stage='test_only')
    targets = dict(confirmed=True, humidity_lower_pct=50, humidity_upper_pct=80,
                   night_temperature_lower_c=15, night_temperature_upper_c=23,
                   light_temperature_lower_c=18, light_temperature_upper_c=26,
                   co2_light_lower_ppm=600, co2_light_upper_ppm=850)
    # Values here exercise the rule dispatch and are not operating targets.
    raw = [dict(timestamp=(start_date + pd.Timedelta(days=offset, hours=hour)).isoformat(),
                temperature_c=temperature, humidity_pct=90, co2_ppm=500, light_phase=phase)
           for offset in range(1, 6) for hour, temperature, phase in [(2, 12, 'dark'), (12, 22, 'light')]]
    base = dict(forecast_as_of=start_date.date().isoformat(), observation_date=at, unit=unit,
                context=context, operating_targets=targets, measurements=measurements)

    def directions(result):
        return {card['variable']: card['direction'] for card in result['recommendations']}

    example = recommend(base | {'environment_rows': raw})
    assert directions(example) == {'humidity': 'decrease', 'dark_temperature': 'increase', 'light_co2': 'increase'}
    assert not example['automatic_control'] and example['growth_gain_estimate'] is None
    csv_text = io.StringIO(newline='')
    writer = csv.DictWriter(csv_text, fieldnames=raw[0].keys())
    writer.writeheader()
    writer.writerows(raw)
    assert recommend(base | {'environment_csv': csv_text.getvalue()}) == example

    reverse = [row | dict(temperature_c=29 if row['light_phase'] == 'light' else 19,
                          humidity_pct=45, co2_ppm=950) for row in raw]
    assert directions(recommend(base | {'environment_rows': reverse})) == {
        'light_temperature': 'decrease', 'humidity': 'increase', 'light_co2': 'decrease'}
    mixed = [row | {'humidity_pct': 45 if i < 4 else 90} for i, row in enumerate(raw)]
    assert directions(recommend(base | {'environment_rows': mixed}))['humidity'] == 'stabilize'
    spike = [row | {'humidity_pct': 90 if i == 0 else 70} for i, row in enumerate(raw)]
    result = recommend(base | {'environment_rows': spike})
    assert 'humidity' not in directions(result) and result['checks']['humidity'] == 'isolated_excursion'
    sparse = recommend(base | {'environment_rows': raw[:-2]})
    assert not sparse['recommendations'] and sparse['checks']['humidity'] == 'insufficient_sensor_days'
    missing_co2 = recommend(base | {'environment_rows': [row | {'co2_ppm': ''} for row in raw]})
    assert directions(missing_co2) == {'dark_temperature': 'increase', 'humidity': 'decrease'}
    no_co2_upper = {key: value for key, value in targets.items() if key != 'co2_light_upper_ppm'}
    assert recommend(base | {'environment_rows': raw, 'operating_targets': no_co2_upper})['co2_check'] == 'needs_upper_target'

    boundary = [raw[0] | {'timestamp': base['forecast_as_of'] + 'T12:00:00', 'humidity_pct': 45},
                raw[0] | {'timestamp': at + 'T02:00:00', 'humidity_pct': 45}]
    assert recommend(base | {'environment_rows': raw + boundary}) == example
    assert not recommend(base | {'environment_rows': raw, 'operating_targets': None})['recommendations']
    history = [row for row in measurements if row['measured_date'] != at]
    start = [row for row in history if row['measured_date'] == base['forecast_as_of']]
    followup = [row | {'measured_date': at} for row in start]
    same = recommend(base | {'measurements': history + followup, 'environment_rows': raw})
    assert same['status'] == 'within_expected_variation' and not same['recommendations']
    missing = recommend(base | {'measurements': history, 'environment_rows': raw})
    assert missing['status'] == 'missing_followup_panel' and not missing['recommendations']

    # Synthetic targets exercise unit checks and rule routing only.
    nutrient_targets = targets | dict(nutrient_ec_lower=0.8, nutrient_ec_upper=1.4,
                                      nutrient_ec_unit='mS/cm', nutrient_ph_lower=5.5, nutrient_ph_upper=6.5)
    nutrient_rows = [row | dict(nutrient_ec=1.7, nutrient_ph=5.2) for row in raw]
    nutrient_case = base | dict(operating_targets=nutrient_targets, environment_rows=nutrient_rows,
                                context=context | {'nutrient_ec_unit': 'mS/cm'})
    nutrient_result = recommend(nutrient_case)
    assert nutrient_result['forecast'] == example['forecast']
    assert directions(nutrient_result) == directions(example) | {'nutrient_ec': 'decrease', 'nutrient_ph': 'increase'}
    assert next(c for c in nutrient_result['recommendations'] if c['variable'] == 'nutrient_ec')['unit'] == 'mS/cm'
    aliases = [row | dict(ei_value=1.7, pl_value=5.2, tl_value=42) for row in raw]
    csv_text = io.StringIO(newline='')
    writer = csv.DictWriter(csv_text, fieldnames=aliases[0].keys())
    writer.writeheader()
    writer.writerows(aliases)
    csv_case = {key: value for key, value in nutrient_case.items() if key != 'environment_rows'}
    assert recommend(csv_case | {'environment_csv': csv_text.getvalue()}) == nutrient_result
    for ec_context in (context, context | {'nutrient_ec_unit': 'µS/cm'}):
        no_ec_unit = recommend(nutrient_case | {'context': ec_context})
        assert no_ec_unit['checks']['nutrient_ec'] == 'needs_matching_ec_unit'
        assert directions(no_ec_unit) == directions(example) | {'nutrient_ph': 'increase'}
    sparse_ph = [row | {'nutrient_ph': ''} if i >= 8 else row for i, row in enumerate(nutrient_rows)]
    no_ph = recommend(nutrient_case | {'environment_rows': sparse_ph})
    assert no_ph['checks']['nutrient_ph'] == 'insufficient_sensor_days'
    assert directions(no_ph) == directions(example) | {'nutrient_ec': 'decrease'}
    return dict(csv_matches_manual=True, increasing_and_decreasing_directions_verified=True,
                mixed_and_isolated_excursions_verified=True, insufficient_readings_verified=True,
                profile_missing_and_co2_limit_verified=True, measurement_dates_verified=True,
                missing_or_normal_followup_verified=True, nutrient_aliases_and_units_verified=True)


def main():
    before = hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest()
    rows = read_evaluation()
    evaluation = {}
    for fold in [2, 3, 4]:
        train, test = rows[rows.fold < fold], rows[rows.fold == fold]
        threshold = float(train.gap.quantile(CALIBRATION_QUANTILE))
        evaluation[fold] = dict(threshold_pp=threshold, **environmental_comparison(train, test, threshold))
    dev, held = rows[rows.fold < 4], rows[rows.fold == 4]
    threshold = evaluation[4]['threshold_pp']
    np.testing.assert_allclose(threshold, GROWTH_ALERT_PP, atol=1e-12, rtol=0)
    replay = replay_held_alerts(held, threshold)
    assert before == hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest()
    output = dict(calibration_n=len(dev), calibration_quantile=CALIBRATION_QUANTILE,
                  evaluated_n=len(held), evaluated_farm_windows=len(held.groupby(GROUP)),
                  evaluated_farms=int(held.farm_id.nunique()), gap_threshold_pp=threshold,
                  baseline_forecast_mae=float((held.forecast - held.actual).abs().mean()),
                  baseline_forecast_r2=float(r2_score(held.actual, held.forecast)),
                  retrospective_comparisons=evaluation, historical_replay=replay,
                  holdout_already_used_in_prior_work=True, intervention_outcomes_available=False)
    (HERE / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output | {'historical_replay': {k: v for k, v in replay.items() if k != 'reviews'}}, indent=2))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
