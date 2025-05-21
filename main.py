import streamlit as st
import os
import pandas as pd
import numpy as np
import altair as alt
import fastf1 as ff1
from fastf1 import plotting
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(page_title="F1 Proyect")
# Código back end
# Cargar datos
@st.cache_data()
def get_weather_data():
    race_weather_2018 = pd.read_csv('/home/liliana/app/datasets/races_weather_2018.csv')
    race_weather_2019 = pd.read_csv('/home/liliana/app/datasets/races_weather_2019.csv')
    race_weather_2020 = pd.read_csv('/home/liliana/app/datasets/races_weather_2020.csv')
    race_weather_2021 = pd.read_csv('/home/liliana/app/datasets/races_weather_2021.csv')
    race_weather_2022 = pd.read_csv('/home/liliana/app/datasets/races_weather_2022.csv')
    race_weather_2023 = pd.read_csv('/home/liliana/app/datasets/races_weather_2023.csv')
    race_weather_2024 = pd.read_csv('/home/liliana/app/datasets/races_weather_2024.csv')

    weather = pd.concat([race_weather_2018, race_weather_2019,race_weather_2020,race_weather_2021,race_weather_2022,race_weather_2023,race_weather_2024], ignore_index=True)
    return weather
@st.cache_data()
def get_data():
    race_2018 = pd.read_csv('/home/liliana/app/datasets/races_2018.csv')
    race_2019 = pd.read_csv('/home/liliana/app/datasets/races_2019.csv')
    race_2020 = pd.read_csv('/home/liliana/app/datasets/races_2020.csv')
    race_2021 = pd.read_csv('/home/liliana/app/datasets/races_2021.csv')
    race_2022 = pd.read_csv('/home/liliana/app/datasets/races_2022.csv')
    race_2023 = pd.read_csv('/home/liliana/app/datasets/races_2023.csv')
    race_2024 = pd.read_csv('/home/liliana/app/datasets/races_2024.csv')

    races = pd.concat([race_2018, race_2019,race_2020,race_2021,race_2022,race_2023,race_2024], ignore_index=True)
    return races

@st.cache_data()
def get_session(year, country):
    session = ff1.get_session(year, country, 'R')
    session.load()
    return session


data = get_data()
weather_data = get_weather_data()

# Contenido de la página
st.title("Proyecto F1")
st.divider()
st.write("Visualiza estadísticas de los pilotos en las carreras de Formula 1 del periodo 2018-2024")
st.divider()

@st.fragment
def research():
    st.caption("_Los datos serán filtrados según su compatibilidad con el piloto 1 elegido_")

    # Inicializar variables de sesión si no existen
    for key in ['compare', 'driver1', 'driver2', 'year', 'race']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'compare' else False

    # Callbacks
    def update_race():
        st.session_state.race = st.session_state.race_select

    def update_pilot():
        st.session_state.driver1 = st.session_state.driver1_select

    # Checkbox de comparación
    st.session_state.compare = st.checkbox("Comparar dos pilotos", value=st.session_state.compare, key='compare_checkbox')

    # Obtener pilotos
    all_drivers = sorted(data['Driver'].unique())

    # Sección de pilotos
    col1, col2 = st.columns(2)

    with col1:
        driver1 = st.selectbox(
            "Piloto 1",
            all_drivers,
            index=all_drivers.index(st.session_state.driver1) if st.session_state.driver1 in all_drivers else 0,
            key='driver1_select',
            on_change=update_pilot
        )
        st.session_state.driver1 = driver1

    with col2:
        if st.session_state.compare:
            years_driver1 = set(data[data['Driver'] == st.session_state.driver1]['Year'].unique())
            drivers_for_pilot2 = sorted(list(set(data[data['Year'].isin(years_driver1)]['Driver'].unique()) - {st.session_state.driver1}))

            # Validar driver2
            if st.session_state.driver2 not in drivers_for_pilot2:
                st.session_state.driver2 = drivers_for_pilot2[0] if drivers_for_pilot2 else None

            driver2 = st.selectbox(
                "Piloto 2",
                drivers_for_pilot2,
                index=drivers_for_pilot2.index(st.session_state.driver2) if st.session_state.driver2 in drivers_for_pilot2 else 0,
                key='driver2_select'
            )
            st.session_state.driver2 = driver2
        else:
            st.session_state.driver2 = None  # Reset si ya no está comparando

    # Función para obtener años disponibles
    def get_available_years():
        if st.session_state.driver1 and (st.session_state.compare and st.session_state.driver2):
            years_driver1 = set(data[data['Driver'] == st.session_state.driver1]['Year'].unique())
            years_driver2 = set(data[data['Driver'] == st.session_state.driver2]['Year'].unique())
            return sorted(list(years_driver1 & years_driver2))
        elif st.session_state.driver1:
            return sorted(data[data['Driver'] == st.session_state.driver1]['Year'].unique())
        return sorted(data['Year'].unique())

    available_years = get_available_years()

    # Validar año seleccionado
    if st.session_state.year not in available_years:
        st.session_state.year = available_years[0] if available_years else None

    # Función para obtener carreras disponibles
    def get_available_races():
        if not st.session_state.year:
            return list(data['Race'].unique())

        year_data = data[data['Year'] == st.session_state.year]

        if st.session_state.driver1 and (st.session_state.compare and st.session_state.driver2):
            races_driver1 = set(year_data[year_data['Driver'] == st.session_state.driver1]['Race'].unique())
            races_driver2 = set(year_data[year_data['Driver'] == st.session_state.driver2]['Race'].unique())
            return list(races_driver1 & races_driver2)
        elif st.session_state.driver1:
            return list(year_data[year_data['Driver'] == st.session_state.driver1]['Race'].unique())
        return list(year_data['Race'].unique())

    available_races = get_available_races()

    # Validar carrera seleccionada
    if st.session_state.race not in available_races:
        st.session_state.race = available_races[0] if available_races else None

    # Sección de año y carrera
    col1, col2 = st.columns(2)

    with col1:
        year = st.selectbox(
            "Año",
            available_years,
            index=available_years.index(st.session_state.year) if st.session_state.year in available_years else 0,
            key='year_select'
        )
        st.session_state.year = year

    with col2:
        race = st.selectbox(
            "Circuito",
            available_races,
            index=available_races.index(st.session_state.race) if st.session_state.race in available_races else 0,
            key='race_select',
            on_change=update_race
        )

    # Filtro principal
    filter_condition = (data['Year'] == st.session_state.year) & \
                       (data['Race'] == st.session_state.race)

    if st.session_state.driver1:
        drivers_filter = [st.session_state.driver1]
        if st.session_state.compare and st.session_state.driver2:
            drivers_filter.append(st.session_state.driver2)
        filter_condition &= data['Driver'].isin(drivers_filter)

    filtered_data = data[filter_condition].copy()
    filtered_data['LapTime_seconds'] = pd.to_timedelta(filtered_data['LapTime']).dt.total_seconds()

    # Guardar vueltas válidas en sesión
    st.session_state.valid_laps = filtered_data[filtered_data['LapTime_seconds'].notna()].copy()

    st.divider()

def segundos_a_tiempo_formateado(segundos):
    minutos = int(segundos // 60)
    segundos_restantes = segundos % 60
    return f"{minutos}:{segundos_restantes:06.3f}"  
def times():
    filter_condition = (data['Year'] == st.session_state.year) & \
                       (data['Race'] == st.session_state.race)
    
    if st.session_state.driver1:
        driver_filter = [st.session_state.driver1]
        if st.session_state.compare and st.session_state.driver2:
            driver_filter.append(st.session_state.driver2)
        filter_condition &= (data['Driver'].isin(driver_filter))
    
    
    filtered_data = data[filter_condition]

    # Crear gráfico de tiempos por vuelta con Vega-Altair
    if not filtered_data.empty:
        st.subheader("Tiempos por Vuelta")
        
        # Convertir LapTime a segundos
        filtered_data['LapTime_seconds'] = pd.to_timedelta(filtered_data['LapTime']).dt.total_seconds()
        
        # Filtrar vueltas válidas
        st.session_state.valid_laps = filtered_data[filtered_data['LapTime_seconds'].notna()].copy()
        
        if not st.session_state.valid_laps.empty:
            # Calcular el tiempo promedio de todos los pilotos que terminaron esta carrera
            st.session_state.all_drivers_data = data[(data['Year'] == st.session_state.year) & 
                                  (data['Race'] == st.session_state.race) & 
                                  (data['LapTime'].notna())]
            st.session_state.all_drivers_data['LapTime_seconds'] = pd.to_timedelta(st.session_state.all_drivers_data['LapTime']).dt.total_seconds()
            avg_time_all_drivers = st.session_state.all_drivers_data['LapTime_seconds'].mean()
            
            # Crear gráfico base
            base = alt.Chart(st.session_state.valid_laps).encode(
                x=alt.X('LapNumber:Q', title='Número de Vuelta'),
                y=alt.Y('LapTime_seconds:Q', title='Tiempo (segundos)',
                       scale=alt.Scale(zero=False)),
                color='Driver:N',
                tooltip=['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife']
            )
            
            # Línea horizontal con el promedio general
            hline = alt.Chart(pd.DataFrame({'avg_time': [avg_time_all_drivers]})).mark_rule(
                color='gray',
                strokeDash=[5, 5]
            ).encode(
                y='avg_time:Q',
                size=alt.value(2)
            )
            
            # Texto para la línea de promedio
            hline_text = alt.Chart(pd.DataFrame({'x': [st.session_state.valid_laps['LapNumber'].max() * 0.9],
                                               'y': [avg_time_all_drivers],
                                               'text': [f'Promedio general: {avg_time_all_drivers:.2f}s']})
                                 ).mark_text(
                align='right',
                baseline='bottom',
                dx=-5,
                dy=-5,
                color='white'
            ).encode(
                x='x:Q',
                y='y:Q',
                text='text:N'
            )
            
            # Líneas de tiempo por vuelta
            lines = base.mark_line(point=False).encode(
                opacity=alt.value(0.7)
            )
            
            # Puntos para mejores vueltas personales
            best_laps = base.transform_filter(
                alt.datum.IsPersonalBest == True
            ).mark_point(
                size=100,
                filled=True
            ).encode(
                color=alt.Color('Driver:N', legend=None),
                shape=alt.value('diamond'),
                tooltip=['Driver', 'LapNumber', 'LapTime']
            )
            
            # Combinar gráficos
            chart = (hline + hline_text + lines + best_laps).properties(
                title=f'Tiempos por Vuelta - {st.session_state.race} {st.session_state.year}',
                width=800,
                height=400
                ).interactive()
            
            # Mostrar el gráfico
            st.altair_chart(chart, use_container_width=True)

            avg_time_formatted = segundos_a_tiempo_formateado(avg_time_all_drivers)
            # Mostrar el promedio general como métrica
            st.metric("Tiempo promedio de todos los pilotos", f"{avg_time_formatted}")
    
            # Mostrar estadísticas
            st.subheader("Estadísticas de Tiempos")

            # Función de conversión mejorada (MM:SS.sss)
            def format_laptime(seconds):
                if pd.isna(seconds):
                    return ""
                minutes = int(seconds // 60)
                seconds_remaining = seconds % 60
                return f"{minutes}:{seconds_remaining:06.3f}"

            # Crear tabla comparativa
            stats = st.session_state.valid_laps.groupby('Driver')['LapTime_seconds'].agg(['mean', 'min', 'max', 'std'])
            stats.columns = ['Promedio', 'Mejor', 'Peor', 'Desviación']

            # Aplicar formato a las columnas de tiempo
            time_columns = ['Promedio', 'Mejor', 'Peor']
            stats[time_columns] = stats[time_columns].applymap(format_laptime)

            # Calcular diferencia con promedio general (en segundos)
            stats['vs Promedio (s)'] = st.session_state.valid_laps.groupby('Driver')['LapTime_seconds'].mean() - avg_time_all_drivers
            stats['vs Promedio'] = stats['vs Promedio (s)'].apply(lambda x: f"{x:+.3f}s" if not pd.isna(x) else "")

            # Mostrar tabla formateada
            st.dataframe(
                stats[['Promedio', 'Mejor', 'Peor', 'vs Promedio', 'Desviación']].style.format({
                    'Desviación': '{:.3f}s',  # La desviación se mantiene en segundos
                    'vs Promedio (s)': '{:+.3f}'  # Columna oculta con el valor numérico
                })
            )

def tyres_individual():
     # Mostrar estrategia de neumáticos
            st.subheader("Estrategia de Neumáticos")

            year = st.session_state.all_drivers_data['Year'].unique()[0]
            country = st.session_state.all_drivers_data['Country'].unique()[0]

            # Procesar stint data
            stints = st.session_state.valid_laps.groupby(["Driver", "Stint", "Compound","Team"])['LapNumber'].agg(
                StintLength='count',
                StartLap='min',
                EndLap='max'
            ).reset_index()
            
            # Crear gráfico con Altair
            chart = alt.Chart(stints).mark_bar().encode(
                y=alt.Y('Driver:N', 
                        title='Piloto',
                        sort=alt.EncodingSortField(field='Position', order='ascending')),
                x=alt.X('StartLap:Q', title='Vuelta de inicio'),
                x2='EndLap:Q',
                color=alt.Color('Compound:N',
                            legend=alt.Legend(title="Compuesto"),
                            scale=alt.Scale(
                                domain=['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'],
                                range=['#FF3333', '#FFD700', '#FFFFFF', '#33CCFF', '#0066CC']
                            )),
                tooltip=['Driver', 'Team', 'Compound', 'StartLap', 'EndLap', 'StintLength']
            ).properties(
                title=f'Estrategias de neumáticos - {country} {year}',
                width=800,
                height=alt.Step(30)  # Altura de cada barra
            ).configure_axis(
                grid=False
            ).configure_view(
                strokeWidth=0  # Sin borde
            )
            
            # Mostrar gráfico
            st.altair_chart(chart, use_container_width=True)
            
@st.fragment
def positions():
    # Mostrar posiciones en carrera
    st.subheader("Cambio en posiciones")
    
    # Verificar si los datos existen en el session_state
    if 'all_drivers_data' not in st.session_state or st.session_state.all_drivers_data.empty:
        st.error("No se encontraron datos de vueltas válidas. Primero selecciona pilotos y carrera en la pestaña 'Estadísticas del piloto'")
        return
    
    # 1. Obtener la lista única de pilotos
    all_drivers = st.session_state.all_drivers_data['Driver'].unique().tolist()
    default_drivers = st.session_state.valid_laps['Driver'].unique().tolist() if 'valid_laps' in st.session_state else []

    # Checkbox para comparar todos los pilotos
    show_all = st.checkbox("Mostrar todos los pilotos", value=False, key='all_checkbox')
    
    if show_all:
        st.session_state.selected_drivers = st.multiselect(
            'Selecciona los pilotos a mostrar:',
            options=all_drivers,
            default=all_drivers
        )
    else:
        # 2. Widget multiselect para elegir pilotos
        st.session_state.selected_drivers = st.multiselect(
            'Selecciona los pilotos a mostrar:',
            options=all_drivers,
            default=default_drivers
        )
    
    if not st.session_state.selected_drivers:
        st.warning("Selecciona al menos un piloto")
        return
    
    # 3. Filtrar los datos según la selección
    filtered_data = st.session_state.all_drivers_data[
        st.session_state.all_drivers_data['Driver'].isin(st.session_state.selected_drivers)
    ]
    
    if filtered_data.empty:
        st.warning("No hay datos para los pilotos seleccionados")
        return
    
    # 4. Crear el gráfico de posiciones
    chart = alt.Chart(filtered_data).mark_line().encode(
        x=alt.X('LapNumber:Q', title='Número de Vuelta'),
        y=alt.Y('Position:Q', title='Posición', 
                scale=alt.Scale(domain=[1, 20], reverse=True)),  # Invertir escala para que 1 sea arriba
        color='Driver:N',
        tooltip=['Driver', 'LapNumber', 'Position', 'Compound']
    ).properties(
        title=f'Posiciones por Vuelta - {st.session_state.race} {st.session_state.year}',
        width=800,
        height=400
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True) 
    tyres_parrilla()          

def weather():
            weather_data = get_weather_data()
            filter_condition_weather = (weather_data['Year'] == st.session_state.year) & \
                       (weather_data['Race'] == st.session_state.race)
    
            filtered_weather_data = weather_data[filter_condition_weather]
            #st.dataframe(filtered_weather_data)
    
            st.subheader("Clima")
            #st.dataframe(filtered_weather_data)
            filtered_weather_data['Time'] = pd.to_timedelta(filtered_weather_data['Time']).dt.total_seconds()


            base_weather = alt.Chart(filtered_weather_data).encode(
                    x=alt.X('Time:Q', title='Tiempo'),
                    y=alt.Y('Humidity:Q', title='Humedad Relativa (%)',
                                scale=alt.Scale(domain=[0, 100], zero=False)) )
            
            lines_humidity= base_weather.mark_line(point=False).encode(
                    opacity=alt.value(0.7),
                    color=alt.value('green')
                )
            
            legend = alt.Chart(pd.DataFrame({
                'Variable': ['Humidity'],
                'Color': ['green']
            })).mark_point(size=50).encode(
                y=alt.Y('Variable:N', axis=alt.Axis(title=None, orient='right')),
                color=alt.Color('Color:N', scale=None)
            ).properties(title='Leyenda')
            
            chart_weather = (lines_humidity).properties(
                    title=f'Humedad relativa durante la carrera - {st.session_state.race} {st.session_state.year}',
                    width=580,
                    height=300
                ).interactive()
            
            
            st.altair_chart(alt.hconcat(chart_weather, legend), use_container_width=True)

            #Grafico temperaturas durante la carrera

            # Base del gráfico
            base = alt.Chart(filtered_weather_data).encode(
                x=alt.X('Time:Q', title='Tiempo'),
            )

            # Gráfico combinado con leyendas
            chart_combined = alt.layer(
                base.mark_line(color='blue', point=False).encode(
                    y=alt.Y('AirTemp:Q', title='Temperatura (°C)', scale=alt.Scale(zero=False)),
                    color=alt.value('blue'),  # Color fijo para AirTemp
                    opacity=alt.value(0.7)
                ).transform_filter(alt.datum.AirTemp).properties(name='AirTemp'),  # Filtro opcional para datos válidos

                base.mark_line(color='red', point=False).encode(
                    y=alt.Y('TrackTemp:Q'),
                    color=alt.value('red'),  # Color fijo para TrackTemp
                    opacity=alt.value(0.7)
                ).transform_filter(alt.datum.TrackTemp).properties(name='TrackTemp')
            ).properties(
                title=f'Temperaturas durante la carrera - {st.session_state.race} {st.session_state.year}',
                width=580,
                height=300
            ).interactive()

            # Leyenda manual (opcional, si Altair no la genera automáticamente)
            legend = alt.Chart(pd.DataFrame({
                'Variable': ['AirTemp', 'TrackTemp'],
                'Color': ['blue', 'red']
            })).mark_point(size=50).encode(
                y=alt.Y('Variable:N', axis=alt.Axis(title=None, orient='right')),
                color=alt.Color('Color:N', scale=None)
            ).properties(title='Leyenda')

            # Mostrar gráfico y leyenda (en fila o columna)
            st.altair_chart(alt.hconcat(chart_combined, legend), use_container_width=True)

def rotate(xy, *, angle):
    if None in xy or np.isnan(xy).any():  # Verifica nulos o NaN
        return np.array([np.nan, np.nan])  # Devuelve valores inválidos
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    return np.matmul(xy, rot_mat)

def track():
    st.subheader("Mapa del Circuito")
    
    if 'all_drivers_data' not in st.session_state or st.session_state.all_drivers_data.empty:
        st.error("Primero selecciona una carrera en la pestaña 'Estadísticas'")
        return
    
    try:
        year = int(st.session_state.all_drivers_data['Year'].unique()[0])
        location = st.session_state.all_drivers_data['Location'].unique()[0]


        with st.spinner('Cargando datos del circuito...'):
            session = get_session(year, location)
            session.load(telemetry=False, weather=False)  # Solo cargar datos básicos
            
            # Obtener vuelta más rápida y limpiar datos
            lap = session.laps.pick_fastest()
            st.write("Vuelta rápida")
            st.write(lap['LapTime'])

            pos = lap.get_pos_data()
            

            # Antes de rotar las coordenadas, filtra valores nulos
            pos = lap.get_pos_data()
            pos = pos.dropna(subset=['X', 'Y'])  # Elimina filas con X o Y nulos

            if pos.empty:
                st.warning("No hay datos de posición válidos para dibujar el circuito.")
                return
            
            circuit_info = session.get_circuit_info()
            track_angle = circuit_info.rotation / 180 * np.pi
            
            # Procesar coordenadas del circuito
            track_coords = pos[['X', 'Y']].astype(float).to_numpy()
            rotated_track = rotate(track_coords, angle=track_angle)
            track_df = pd.DataFrame(rotated_track, columns=['x', 'y']).dropna()
            
            # Procesar curvas
            corners = []
            offset = 500  # Distancia para etiquetas
            
            for _, corner in circuit_info.corners.iterrows():
                if pd.isna(corner['X']) or pd.isna(corner['Y']) or pd.isna(corner['Angle']):
                    continue 
                    
                # Rotar posición de la curva
                corner_x, corner_y = rotate([corner['X'], corner['Y']], angle=track_angle)
                
                # Calcular posición de la etiqueta
                angle_rad = corner['Angle'] / 180 * np.pi
                offset_x = offset * np.cos(angle_rad)
                offset_y = offset * np.sin(angle_rad)
                label_x, label_y = rotate(
                    [corner['X'] + offset_x, corner['Y'] + offset_y],
                    angle=track_angle
                )
                
                corners.append({
                    'number': corner['Number'],
                    'letter': corner['Letter'],
                    'x': corner_x,
                    'y': corner_y,
                    'label_x': label_x,
                    'label_y': label_y,
                    'label': f"{corner['Number']}{corner['Letter']}"
                })
            
            if not corners:
                st.warning("No se encontraron datos de curvas válidas")
                return
                
            corners_df = pd.DataFrame(corners)
            
            # Crear visualización
            base = alt.Chart(track_df).mark_line(
                color='blue',
                strokeWidth=10
            ).encode(
                x=alt.X('x:Q', axis=None),
                y=alt.Y('y:Q', axis=None, sort='descending'),
                order='index:O'
            )
            
            # Añadir elementos de las curvas
            corner_lines = alt.Chart(corners_df).mark_rule(
                color='white',
                strokeWidth=1
            ).encode(
                x='x:Q',
                y='y:Q',
                x2='label_x:Q',
                y2='label_y:Q'
            )
            
            corner_points = alt.Chart(corners_df).mark_point(
                size=100,
                fill='black',
                stroke='black',
                strokeWidth=1
            ).encode(
                x='label_x:Q',
                y='label_y:Q'
            )
            
            corner_labels = alt.Chart(corners_df).mark_text(
                align='center',
                baseline='middle',
                color='white',
                fontSize=16
            ).encode(
                x='label_x:Q',
                y='label_y:Q',
                text='label:N'
            )
            
            chart = (base + corner_lines + corner_points + corner_labels).properties(
                width=600,
                height=400,
                title=f"{session.event['Location']} - {year}"
            ).configure_view(
                strokeWidth=0
            )
            
            st.altair_chart(chart, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error al generar el mapa: {str(e)}")
        st.warning("Recomendación: Intenta con otra carrera o verifica los datos de entrada")

def tyres_parrilla():
    st.subheader("Estrategia de Neumáticos por Piloto")
        # Verificar si los datos existen y si hay pilotos seleccionados
    if ('all_drivers_data' not in st.session_state or st.session_state.all_drivers_data.empty or 
        'selected_drivers' not in st.session_state or not st.session_state.selected_drivers):
        st.error("Primero selecciona pilotos en la pestaña 'Posiciones'")
        return
    
    # Obtener datos básicos
    year = st.session_state.all_drivers_data['Year'].unique()[0]
    country = st.session_state.all_drivers_data['Country'].unique()[0]
    
    # Filtrar datos válidos solo para los pilotos seleccionados
    valid_laps = st.session_state.all_drivers_data[
        (st.session_state.all_drivers_data['Driver'].isin(st.session_state.selected_drivers)) & 
        (st.session_state.all_drivers_data['LapTime_seconds'].notna())
    ].copy()
    
    if valid_laps.empty:
        st.warning("No hay datos de vueltas válidas para los pilotos seleccionados.")
        return
    # Procesar stint data
    stints = valid_laps.groupby(["Driver", "Stint", "Compound","Team"])['LapNumber'].agg(
        StintLength='count',
        StartLap='min',
        EndLap='max'
    ).reset_index()
    
    # Crear gráfico con Altair
    chart = alt.Chart(stints).mark_bar().encode(
        y=alt.Y('Driver:N', 
                title='Piloto',
                sort=alt.EncodingSortField(field='Position', order='ascending')),
        x=alt.X('StartLap:Q', title='Vuelta de inicio'),
        x2='EndLap:Q',
        color=alt.Color('Compound:N',
                       legend=alt.Legend(title="Compuesto"),
                       scale=alt.Scale(
                           domain=['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'],
                           range=['#FF3333', '#FFD700', '#FFFFFF', '#33CCFF', '#0066CC']
                       )),
        tooltip=['Driver', 'Team', 'Compound', 'StartLap', 'EndLap', 'StintLength']
    ).properties(
        title=f'Estrategias de neumáticos - {country} {year}',
        width=800,
        height=alt.Step(30)  # Altura de cada barra
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0  # Sin borde
    )
    
    # Mostrar gráfico
    st.altair_chart(chart, use_container_width=True)


st.session_state.cargar_button = st.button('Cargar',use_container_width=True)

if not st.session_state.cargar_button:
    research()

#tab1, tab2, tab3 = st.tabs(["Estadísticas del piloto", "Carrera", "Clima"])


if st.session_state.cargar_button:
    # Primero ejecutar research() y times() para cargar los datos
    research()
    times()
    tyres_individual()
    positions()
    
    weather()
    with st.sidebar:
        st.header("Circuito")
        track()

