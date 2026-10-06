from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_ui_load_and_scenario():
    app = AppTest.from_file(str(Path(__file__).parents[1]/'app.py'), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) >= 4
    assert len(app.tabs) == 5
    next(s for s in app.selectbox if s.label=='Сценарий демонстрации').select('Узкое место S14').run()
    for _ in range(10):
        next(b for b in app.button if b.label=='Следующий шаг').click().run()
        assert not app.exception
    assert app.session_state['twin'].step == 10
    assert app.session_state['twin'].scored.set_index('station_id').loc['S14','bottleneck_risk'] >= .75
    incident = next(i for i in app.session_state['twin'].incidents if i['station_id']=='S14' and i['kind']=='risk' and i['state']!='Устранён')
    incident_id = incident['id']
    next(s for s in app.selectbox if s.label=='ID инцидента для принятия в работу').select(incident_id).run()
    next(b for b in app.button if b.label=='Принять в работу').click().run()
    assert not app.exception
    assert next(i for i in app.session_state['twin'].incidents if i['id']==incident_id)['state'] == 'Принят в работу'
    next(s for s in app.selectbox if s.label=='Сценарий демонстрации').select('Восстановление').run()
    next(b for b in app.button if b.label=='Следующий шаг').click().run()
    assert not app.exception
    assert next(i for i in app.session_state['twin'].incidents if i['id']==incident_id)['state'] == 'Устранён'
    next(b for b in app.button if b.label=='Сброс').click().run()
    assert not app.exception
    assert app.session_state['twin'].step == 0
