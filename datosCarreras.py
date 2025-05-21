import pandas as pd
import fastf1 as ff1

ff1.set_log_level('warning')

# Habilitamos el cache en nuestro equipo
path = '/home/liliana/app/cache'
ff1.Cache.enable_cache(path)

# Ignoramos los Warning
pd.options.mode.chained_assignment = None

# Cargar la sesion
def cargarSesion(año, circuito, tipo):
    try:
        sesion = ff1.get_session(año, circuito, tipo)
        sesion.load()  # Carga los datos
        laps = sesion.laps
        return laps, sesion
    except Exception as e:
        print(f"Error cargando sesión {año} {circuito} {tipo}: {str(e)}")
        return None  # Devuelve None si hay un error



# Lista para almacenar todas las sesiones
todas_las_sesiones = []

# Cargando las sesiones de quali para Silverstone de los últimos años
for year in range(2018, 2025):
    todas_las_sesiones = []
    for c in range(1,25):
        circuito = c
        tipo = 'R'  # Carrera (Race)

        try:
            laps, sesion = cargarSesion(year, circuito, tipo)
            if laps is not None:
                laps['Year'] = year  # Añadir columna con el año
                laps['Race'] = sesion.event['EventName']  # Añadir columna con el nombre del evento
                laps['Date'] = sesion.event['EventDate']
                laps['Location'] = sesion.event['Location']  
                laps['Country'] = sesion.event['Country']


                todas_las_sesiones.append(laps)
                print(f"Datos de {year} cargados correctamente")
            else:
                print(f"No se encontraron datos para {year}")
        except Exception as e:
            print(f"Error al cargar datos de {year}: {str(e)}")

    # Procesamiento y guardado de datos
    if not todas_las_sesiones:
        print(f"No se pudieron cargar datos para {year} año.")
    else:
        df_completo = pd.concat(todas_las_sesiones)
        df_completo.reset_index(drop=True, inplace=True)

        # Guardar en múltiples formatos
        df_completo.to_csv(f'/home/liliana/app/datasets/races_{year}.csv', index=False)

        print("\nResumen de datos:")
        print(f"- Año: {year}")
        print(f"- Total de registros: {len(df_completo)}")
        print(f"- Columnas disponibles: {list(df_completo.columns)}")
        print("\nArchivos generados:")
        print(f"- 'races_{year}.csv' ")
