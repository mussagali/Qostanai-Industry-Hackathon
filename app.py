from __future__ import annotations
import html
import pandas as pd
import plotly.express as px
import streamlit as st
from factory import Twin, ZONES, SCENARIOS, STATUS, forecast, business_effect, demo_observation
from simulator import load_config, initial_line
from prediction import train_models, get_model_metrics

st.set_page_config(page_title='АЛЛЮР • Цифровой двойник', page_icon='🏭', layout='wide')
st.markdown('''<style>
.stApp {background:#091320;color:#e7eef7} [data-testid="stSidebar"]{background:#101e30}
.block-container{padding-top:2rem} h1{letter-spacing:-1.5px} h2,h3{letter-spacing:-.5px}
.tag{display:inline-block;border:1px solid #27415d;border-radius:20px;padding:5px 12px;color:#7ae0db;font-size:12px;margin:0 8px 8px 0}
.node{border:1px solid #29415c;border-left:4px solid var(--c);border-radius:12px;padding:14px;background:#122237;margin:7px 0;min-height:145px}
.node b{color:#f0f6ff}.node small{color:#a1b2c8}.node .risk{font-size:23px;color:var(--c);font-weight:700}
[data-testid="stMain"] [data-testid="stHorizontalBlock"]{flex-wrap:wrap}
[data-testid="stMain"] [data-testid="stColumn"]{min-width:190px;flex:1 1 190px}
[data-testid="stMetricLabel"] p{white-space:normal!important;overflow:visible!important}
[data-testid="stMetricValue"]{font-size:26px}
[data-testid="stMetricValue"]>div{white-space:normal!important;text-overflow:clip!important;overflow-wrap:anywhere}
.flow{color:#8da2bc;font-size:13px;padding:8px 0}.stMetric{border:1px solid #243a53;border-radius:12px;padding:16px;background:#102035}
[data-testid="stMetricLabel"]{color:#9db2ca} [data-testid="stMetricValue"]{color:#e7eef7}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def resources():
    return load_config(), train_models()

config, models = resources()
if 'twin' not in st.session_state:
    st.session_state.twin = Twin(config, models)

with st.sidebar:
    st.markdown('### АЛЛЮР / TWIN')
    st.caption('Кейс 2 • Qostanai Industry Hackathon')
    scenario = st.selectbox('Сценарий демонстрации', list(SCENARIOS))
    live = st.toggle('Автообновление каждые 3 секунды', value=False)
    st.caption('Каждое обновление — новый снимок симуляции. Шаг не равен минуте реального производства.')
    left, right = st.columns(2)
    advance = left.button('Следующий шаг', width='stretch', disabled=st.session_state.twin.source != 'Симуляция')
    reset = right.button('Сброс', width='stretch')
    if reset:
        st.session_state.twin = Twin(config, models)
    if advance:
        st.session_state.twin.advance(SCENARIOS[scenario])
    st.divider()
    uploaded = st.file_uploader('Снимок завода CSV', type=['csv'])
    if st.button('Загрузить снимок', disabled=uploaded is None):
        try:
            st.session_state.twin.import_csv(uploaded.getvalue())
            st.success('Снимок загружен. Автосимуляция отключена.')
        except (ValueError, UnicodeError, pd.errors.ParserError) as error:
            st.error(str(error))
    st.download_button('Шаблон CSV', demo_observation(initial_line(config=config)).to_csv(index=False).encode('utf-8-sig'), 'allur_snapshot.csv', 'text/csv')
    st.caption('Данные демонстрационные. Схема условная, подключение к оборудованию АЛЛЮР не выполнено.')


def fmt_pct(value):
    return 'Нет выпуска' if value is None else f'{value * 100:.1f}%'


def advice(row):
    reasons = []
    if row.cycle_time > 58: reasons.append(f'Цикл {row.cycle_time:.1f} с превышает демобазу 48 с')
    if row.queue_length > 5: reasons.append(f'Очередь: {int(row.queue_length)} автомобилей')
    if row.downtime_min > 3: reasons.append(f'Простой: {row.downtime_min:.1f} мин за окно')
    if pd.notna(row.vibration) and row.vibration > 3: reasons.append(f'Вибрация: {row.vibration:.2f} мм/с')
    if row.confidence < .75: reasons.append('Неполная телеметрия: нужна ручная проверка')
    action = 'Проверить оборудование и перераспределить нагрузку' if row.bottleneck_risk >= row.defect_risk else 'Проверить параметры процесса и направить изделия на ОТК'
    return reasons or ['Существенных отклонений в доступных показателях нет'], action


@st.fragment(run_every=3 if live else None)
def dashboard():
    twin = st.session_state.twin
    if live and twin.source == 'Симуляция':
        twin.advance(SCENARIOS[scenario])
    data, kpi = twin.scored, twin.kpis
    st.markdown('<span class="tag">КЕЙС 02</span><span class="tag">ПРОТОТИП / ДЕМО</span>', unsafe_allow_html=True)
    st.title('Цифровой двойник завода')
    st.caption(f'{config["line"]["name"]} • 30 станций / 5 участков • Источник: {twin.source} • Снимок {twin.step} • Обновлено: {twin.updated_at} (UTC)')
    cols = st.columns(4)
    cols[0].metric('Выпуск ОТК, авто/ч', f'{kpi["throughput"]:.0f}')
    cols[1].metric('Средняя загрузка оборудования', fmt_pct(kpi['utilization']))
    cols[2].metric('Простой, маш.-мин', f'{kpi["equipment_downtime"]:.1f}')
    cols[3].metric('Качество ОТК · FPY', fmt_pct(kpi['quality']))
    st.caption('KPI за общее окно наблюдения 60 мин в демо. Машино-минуты — сумма по оборудованию, не простой всей линии. FPY — доля годных изделий по счётчикам ОТК.')
    tabs = st.tabs(['Карта завода', 'Инциденты', 'Панель руководителя', 'Прогноз и ИИ', 'Эффект и архитектура'])
    colors = {'Normal':'#37d9af', 'Warning':'#fbbf24', 'Critical':'#fb7185'}
    with tabs[0]:
        st.markdown('### Производственный поток')
        st.markdown('<div class="flow">Склад комплектующих → Сварка → Окраска → Салон → Шасси → Финальная сборка → Отгрузка</div>', unsafe_allow_html=True)
        st.caption('Склад и отгрузка показаны как границы потока; отдельная телеметрия этих участков в демо отсутствует. Цвет: зелёный — норма, жёлтый — внимание, красный — критично.')
        zones = st.columns(5)
        for col, (zone, name) in zip(zones, ZONES.items()):
            with col:
                st.markdown(f'**{name}**')
                for _, row in data[data.station_type.eq(zone)].iterrows():
                    process = html.escape(str(row.process_name))
                    sensor = {'rich':'Полные датчики','partial':'Часть датчиков','manual':'Ручной контроль'}[row.sensor_level]
                    st.markdown(f'<div class="node" style="--c:{colors[row.status]}"><b>{html.escape(row.station_id)} · {STATUS[row.status]}</b><br><small>{process}</small><div class="risk">{row.bottleneck_risk:.0%}</div><small>Риск узкого места · очередь {int(row.queue_length)}<br>{sensor}</small></div>', unsafe_allow_html=True)
        st.divider()
        selected = st.selectbox('Оборудование: подробности', data.station_id.tolist())
        row = data[data.station_id.eq(selected)].iloc[0]
        st.markdown(f'### {selected} · {row.process_name}')
        a,b,c,d = st.columns(4)
        a.metric('Цикл', f'{row.cycle_time:.1f} с')
        b.metric('Загрузка', fmt_pct(float(row.utilization)))
        c.metric('Риск брака · модель', f'{row.defect_risk:.0%}')
        d.metric('Полнота данных', f'{row.confidence:.0%}')
        reasons, action = advice(row)
        for reason in reasons: st.write('• ' + reason)
        st.info('Рекомендуемое действие: ' + action)
        st.caption('Полнота данных — эвристика наличия датчиков; это не вероятность правильного прогноза.')
    with tabs[1]:
        st.markdown('### Журнал инцидентов')
        active = [i for i in twin.incidents if i['state'] != 'Устранён']
        st.caption('Инцидент сохраняет ID между обновлениями. После нормализации закрывается автоматически; повторное событие получает новый ID. Журнал хранится в текущей сессии браузера.')
        a,b,c = st.columns(3)
        a.metric('Активные', len(active)); b.metric('Критичные', sum(i['severity']=='Критично' for i in active)); c.metric('Устранённые', len(twin.incidents)-len(active))
        if twin.incidents:
            display = pd.DataFrame(twin.incidents).rename(columns={'id':'ID','station_id':'Станция','severity':'Приоритет','title':'Описание','state':'Состояние','first_step':'Первый снимок','last_step':'Последний снимок','resolved_step':'Закрыт на снимке'})
            st.dataframe(display.drop(columns=['kind']), hide_index=True, width='stretch')
            st.download_button('Выгрузить журнал', display.to_csv(index=False).encode('utf-8-sig'), 'incidents.csv', 'text/csv')
        else: st.success('Активных инцидентов нет')
        opened = [i['id'] for i in active if i['state']=='Открыт']
        if opened:
            selected_id = st.selectbox('ID инцидента для принятия в работу', opened)
            if st.button('Принять в работу'):
                twin.acknowledge(selected_id)
                st.rerun()
    with tabs[2]:
        st.markdown('### Производственный план и эффективность')
        a,b,c,d = st.columns(4)
        a.metric('План смены, авто · демо', f'{kpi["plan"]}')
        b.metric('Прогноз выпуска, авто/смена', f'{kpi["shift_forecast"]:.0f}', f'{kpi["shift_forecast"]-kpi["plan"]:+.0f} к плану')
        c.metric('OEE на выходе линии', fmt_pct(kpi['oee']))
        d.metric('Узкое место, авто/ч', f'{kpi["bottleneck_capacity"]:.0f}', kpi['bottleneck_station'], delta_color='off')
        st.caption('Прогноз смены = текущий выпуск ОТК × 8 ч; это сценарий постоянного темпа. OEE = доступность × производительность × FPY выходной станции. Нулевой выпуск: качество и OEE неизвестны.')
        hist = pd.DataFrame([{k:v for k,v in h.items() if k!='risks'} for h in twin.history])
        st.plotly_chart(px.line(hist, x='step', y='throughput', markers=True, labels={'step':'Снимок','throughput':'Авто/ч'}, title='История выпуска ОТК', template='plotly_dark'), width='stretch')
        grouped = data.groupby('station_type', sort=False).agg(utilization=('utilization','mean'), downtime_min=('downtime_min','sum'), queue_length=('queue_length','sum'))
        grouped.index = grouped.index.map(ZONES)
        st.dataframe(grouped.rename(columns={'utilization':'Загрузка (0–1)','downtime_min':'Простой (маш.-мин)','queue_length':'Очередь (авто)'}), width='stretch')
        st.plotly_chart(px.bar(data, x='station_id', y='output_rate', color='station_type', labels={'station_id':'Станция','output_rate':'Авто/ч','station_type':'Участок'}, title='Пропускная способность оборудования', template='plotly_dark'), width='stretch')
        st.download_button('Выгрузить снимок и прогнозы', data.to_csv(index=False).encode('utf-8-sig'), 'allur_scored_snapshot.csv', 'text/csv')
    with tabs[3]:
        st.markdown('### Раннее предупреждение о простоях и узких местах')
        st.caption('Random Forest оценивает риск узкого места и брака, Isolation Forest — аномалии. Обучение и проверка выполнены на синтетических данных. Риск узкого места — сигнал для проверки возможного простоя, не вероятность отказа оборудования.')
        projected = []
        critical = config['thresholds']['critical']['bottleneck_risk']
        for _, row in data.iterrows():
            prediction = forecast(twin.history, row.station_id, critical=critical)
            projected.append({'Станция':row.station_id,'Риск сейчас':row.bottleneck_risk,'Риск через 5 снимков':prediction['projected'], 'До критического уровня, шагов':prediction['eta_steps'],'Риск брака':row.defect_risk,'Аномальность':row.anomaly_score,'Полнота данных':row.confidence})
        forecast_table = pd.DataFrame(projected).sort_values('Риск через 5 снимков', ascending=False).head(10)
        for column in ['Риск сейчас', 'Риск через 5 снимков', 'Риск брака', 'Аномальность', 'Полнота данных']:
            forecast_table[column] = forecast_table[column].map(lambda v: f'{v:.0%}' if pd.notna(v) else '—')
        st.dataframe(forecast_table, hide_index=True, width='stretch')
        st.caption('Прогноз тренда — линейная экстраполяция последних 8 снимков; минимум 3 для оценки наклона. ETA в шагах симуляции, не в минутах. Вероятности модели не калиброваны на данных завода.')
        st.markdown('#### Проверка моделей на отложенной синтетической выборке')
        st.dataframe(pd.DataFrame(get_model_metrics()).T, width='stretch')
    with tabs[4]:
        st.markdown('### Потенциальный эффект · расчёт предположений')
        st.caption('Введите базовые показатели пилота. Значения по умолчанию условные; результат не является подтверждённой экономией АЛЛЮР. Период расчёта — месяц.')
        a,b,c = st.columns(3)
        downtime = a.number_input('Простой за месяц, ч', min_value=0., value=40.)
        reduction = b.number_input('Сокращение простоя, %', min_value=0., max_value=100., value=15.)
        hour_cost = c.number_input('Стоимость часа простоя, ₸', min_value=0., value=200000.)
        defects = a.number_input('Изделия на доработку за месяц', min_value=0, value=100)
        defect_reduction = b.number_input('Сокращение доработки, %', min_value=0., max_value=100., value=10.)
        rework = c.number_input('Стоимость доработки изделия, ₸', min_value=0., value=30000.)
        effect = business_effect(downtime, reduction, hour_cost, defects, defect_reduction, rework)
        a.metric('Возвращённое время', f'{effect["hours"]:.1f} ч/мес')
        b.metric('Предотвращённая доработка', f'{effect["units"]:.1f} авто/мес')
        c.metric('Валовый эффект, ₸/мес', f'{effect["kzt"]:,.0f}')
        st.caption('Эффект = часы простоя × снижение × стоимость часа + доработка × снижение × стоимость изделия. Затраты внедрения не вычтены; для пилота исключить пересечение двух статей.')
        st.markdown('### Архитектура MVP')
        st.code('Симулятор / CSV → проверка снимка → ML и KPI → журнал инцидентов\n                                      ↓\n                           Streamlit: карта / аналитика / экспорт\n\nПилот: PLC / MES / ОТК → read-only адаптер → хранилище временных рядов\n                      → модель на истории завода → оператор → результат')
        st.markdown('### Демонстрация за 3 минуты')
        st.write('1. Карта завода: выбрать станцию и показать KPI.\n2. Сценарий «Узкое место S14»: 10 шагов, затем прогноз и журнал.\n3. Принять инцидент в работу; выбрать «Восстановление» и обновить.\n4. Показать сценарий брака S07, импорт CSV и расчёт эффекта.')
        st.caption('Для промышленного пилота: согласовать схему и определения KPI, подключить источники, хранить историю, проверить прогнозы на реальных отказах, добавить доступ по ролям. Прототип не управляет PLC.')

dashboard()
