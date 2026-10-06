from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from factory import Twin, demo_observation, validate_observation, plant_kpis, forecast, business_effect
from simulator import initial_line, load_config, spawn_vehicles, advance_vehicles
from prediction import train_models, score_line, get_model_metrics

@pytest.fixture(scope='session')
def config():
    return load_config()

@pytest.fixture(scope='session')
def models():
    return train_models()

def test_topology_and_sensor_gaps(config):
    line = initial_line(config=config)
    assert len(line) == 30
    assert line.station_type.nunique() == 5
    assert not line.station_id.duplicated().any()
    assert line.loc[line.sensor_level.eq('manual'), ['temperature','vibration','torque']].isna().all().all()

def test_missing_config_has_usable_demo_defaults(tmp_path, models):
    config = load_config(str(tmp_path / 'missing.yaml'))
    assert Twin(config, models).kpis['plan'] == 500

def test_kpis_use_final_station_not_sum(config):
    line = demo_observation(initial_line(config=config))
    line['total_count'] = 60
    line['good_count'] = 57
    line['downtime_min'] = 6
    kpi = plant_kpis(line, config)
    assert kpi['throughput'] == 60
    assert kpi['quality'] == .95
    assert kpi['equipment_downtime'] == 180
    assert kpi['oee'] == pytest.approx(60/72*.95)
    assert kpi['shift_forecast'] == 480

def test_no_production_is_not_perfect_quality(config):
    line = demo_observation(initial_line(config=config))
    line[['total_count','good_count','throughput']] = 0
    line['downtime_min'] = 60
    kpi = plant_kpis(line, config)
    assert kpi['quality'] is None and kpi['oee'] is None
    assert kpi['utilization'] == 0

def test_csv_round_trip(config, models):
    twin = Twin(config, models)
    csv = twin.line.to_csv(index=False).encode('utf-8-sig')
    twin.import_csv(csv)
    assert twin.source == 'CSV' and len(twin.scored) == 30
    with pytest.raises(ValueError): twin.advance('Normal')

@pytest.mark.parametrize('bad', ['missing','duplicate','negative','infinite','counts','window','fraction','zone','sensor','manual','missing_process'])
def test_invalid_csv_rejected(config, bad):
    line = demo_observation(initial_line(config=config))
    if bad == 'missing': line = line.drop(columns='cycle_time')
    elif bad == 'duplicate': line.loc[1,'station_id'] = line.loc[0,'station_id']
    elif bad == 'negative': line.loc[0,'cycle_time'] = -1
    elif bad == 'infinite': line.loc[0,'throughput'] = np.inf
    elif bad == 'counts': line.loc[0,'good_count'] = 999
    elif bad == 'window': line.loc[0,'downtime_min'] = 61
    elif bad == 'fraction': line['total_count'] = line.total_count.astype(float); line.loc[0,'total_count'] = 2.5
    elif bad == 'zone': line.loc[0,'station_type'] = 'Paint'
    elif bad == 'sensor': line.loc[0,'sensor_level'] = 'unknown'
    elif bad == 'manual': line.loc[0,'sensor_level'] = 'manual'; line.loc[0,'temperature'] = 50
    elif bad == 'missing_process': line.loc[0,'queue_length'] = np.nan
    with pytest.raises((ValueError, TypeError)): validate_observation(line, config)

def test_import_failure_preserves_state(config, models):
    twin = Twin(config, models)
    before = twin.line.copy()
    with pytest.raises(ValueError): twin.import_csv(b'station_id\nS01\n')
    pd.testing.assert_frame_equal(before, twin.line)
    assert twin.source == 'Симуляция' and twin.step == 0

def test_bottleneck_and_recovery(config, models):
    twin = Twin(config, models)
    start = twin.kpis['throughput']
    for _ in range(10): twin.advance('Developing bottleneck')
    row = twin.scored.set_index('station_id').loc['S14']
    assert row.bottleneck_risk >= .75
    assert twin.kpis['throughput'] < start
    incident = next(i for i in twin.incidents if i['station_id']=='S14' and i['kind']=='risk' and i['state']!='Устранён')
    identity = incident['id']
    twin.acknowledge(identity)
    twin.advance('Developing bottleneck')
    assert incident['id'] == identity and incident['state'] == 'Принят в работу'
    twin.advance('Recovery')
    assert incident['state'] == 'Устранён'
    assert twin.kpis['throughput'] == start

def test_stop_propagates_to_output_and_reopens_incident(config, models):
    twin = Twin(config, models)
    twin.advance('Stopped')
    assert twin.kpis['throughput'] == 0
    incident = next(i for i in twin.incidents if i['kind']=='stop')
    twin.advance('Stopped')
    assert sum(i['kind']=='stop' for i in twin.incidents) == 1
    twin.advance('Recovery')
    assert incident['state'] == 'Устранён'
    twin.advance('Stopped')
    assert sum(i['kind']=='stop' for i in twin.incidents) == 2

def test_defect_scenario_reaches_quality_gate(config, models):
    twin = Twin(config, models)
    before = twin.kpis['quality']
    for _ in range(10): twin.advance('High defect risk')
    row = twin.scored.set_index('station_id').loc['S07']
    assert row.defect_risk >= .55
    assert twin.kpis['quality'] < before

def test_missing_sensors_keep_predictions(config, models):
    twin = Twin(config, models)
    twin.advance('Sensor-poor station')
    row = twin.scored.set_index('station_id').loc['S20']
    assert row.confidence < .75
    assert np.isfinite(row.bottleneck_risk)
    assert any(i['station_id']=='S20' and i['kind']=='data' for i in twin.incidents)

def test_prediction_bounds_and_config_thresholds(config, models):
    line = initial_line(config=config)
    scored = score_line(line, *models)
    assert scored[['bottleneck_risk','defect_risk','anomaly_score','confidence']].map(lambda v: 0<=v<=1).all().all()
    custom = deepcopy(config)
    custom['thresholds']['critical'] = {'bottleneck_risk':0, 'defect_risk':0}
    assert score_line(line, *models, config=custom).status.eq('Critical').all()
    for metrics in get_model_metrics().values():
        assert metrics['n_test'] == 875
        assert .75 <= metrics['auc'] <= 1

def test_forecast_warmup_rising_and_flat():
    assert forecast([], 'S14')['projected'] is None
    history = [{'risks':{'S14':v}} for v in [.4,.5,.6]]
    assert forecast(history,'S14')['eta_steps'] == 2
    assert forecast(history,'S14')['projected'] == 1
    assert forecast([{'risks':{'S14':.4}}]*3,'S14')['eta_steps'] is None

def test_business_effect():
    assert business_effect(40,15,200000,100,10,30000)['kzt'] == 1500000
    with pytest.raises(ValueError): business_effect(40,101,1,1,1,1)

def test_vehicle_completion_at_final_station(config, models):
    line = score_line(initial_line(config=config), *models)
    vehicles = spawn_vehicles(line.station_id.tolist(), config=config)
    vehicles, log = advance_vehicles(vehicles,line,step=1,stations_per_step=30)
    assert vehicles.completed.all()
    assert len(log) == 40*30
