import pandas as pd
import glob
import os

# ==============================================================================
# CONSOLIDACIÓN FINAL: junta dataset_grande/ y dataset_grande_retry/ en
# un único CSV volumétrico y un único CSV de grafos.
# Misma lógica que el script original, adaptada a las 2 carpetas y 4 categorías.
# ==============================================================================

# dataset_grande_retry primero: sus JSONs buenos tienen prioridad sobre
# los vacíos que quedaron en dataset_grande de la primera extracción fallida.
CARPETAS = ["dataset_grande_retry", "dataset_grande"]
CATEGORIAS = ["origin_hijacks", "path_hijacks", "route_leaks", "mass_outages"]


def procesar_categoria(carpeta_categoria, tipo_feature):
    datos = []
    for label_str in ['normal', 'ataque']:
        patron = os.path.join(carpeta_categoria, label_str, "*", "transform", tipo_feature, "*.json")
        archivos = glob.glob(patron)
        for f in archivos:
            try:
                partes_ruta = os.path.normpath(f).split(os.sep)
                evento = partes_ruta[3]
                df = pd.read_json(f)
                if df.shape[0] < df.shape[1]:
                    df = df.T
                df['Evento'] = evento
                df['Label'] = 0 if label_str == 'normal' else 1
                df['Colector'] = "rrc04"
                df['Categoria'] = os.path.basename(carpeta_categoria)
                datos.append(df)
            except Exception as e:
                print(f"  ⚠️ Error en {f}: {e}")
    return datos


def procesar_a_csv(tipo_feature, nombre_salida):
    print(f"\nBuscando {tipo_feature} en ambas carpetas...")
    datos_totales = []
    eventos_vistos = set()  # para evitar duplicados si un evento está en ambas carpetas

    for carpeta_raiz in CARPETAS:
        if not os.path.isdir(carpeta_raiz):
            print(f"  ⚠️ Carpeta no encontrada, se omite: {carpeta_raiz}")
            continue

        for categoria in CATEGORIAS:
            carpeta_categoria = os.path.join(carpeta_raiz, categoria)
            if not os.path.isdir(carpeta_categoria):
                continue

            datos_cat = procesar_categoria(carpeta_categoria, tipo_feature)

            # Filtrar duplicados: si el mismo evento ya vino de dataset_grande, no lo añadimos otra vez
            for df in datos_cat:
                clave = (df['Evento'].iloc[0], df['Label'].iloc[0], df['Categoria'].iloc[0])
                if clave not in eventos_vistos:
                    eventos_vistos.add(clave)
                    datos_totales.append(df)

        print(f"  {carpeta_raiz}: procesado")

    if datos_totales:
        df_final = pd.concat(datos_totales).fillna(0)
        df_final.index.name = 'Timestamp'
        df_final.reset_index(inplace=True)
        df_final.to_csv(nombre_salida, index=False)
        print(f"ÉXITO: {nombre_salida} — {len(df_final)} registros, {df_final['Evento'].nunique()} eventos únicos.")
    else:
        print(f"FALLO: No se encontraron datos para {tipo_feature}.")


procesar_a_csv("Features",      "volumen_rrc04_final.csv")
procesar_a_csv("GraphFeatures", "grafos_rrc04_final.csv")