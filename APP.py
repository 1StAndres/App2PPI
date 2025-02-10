import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from shapely.geometry import Point

# Inicializar el estado de la aplicación
if 'css_cargado' not in st.session_state:
    st.session_state.css_cargado = False

# Cargar datos
@st.cache_data
def calcular_distancias(df, top_n=10):
    """
    Calcula y muestra en Streamlit las distancias entre los compradores con mayores ingresos,
    considerando general, por género y por frecuencia.

    Parámetros:
    - df (DataFrame): Dataset con las columnas 'Latitud', 'Longitud', 'Ingreso_Anual_USD', 'Género' y 'Frecuencia_Compra'.
    - top_n (int): Número de compradores de mayores ingresos a considerar.

    No retorna valores, muestra directamente en la app de Streamlit.
    """

    # 📌 Verificar que el DataFrame contiene las columnas necesarias
    columnas_necesarias = {'Latitud', 'Longitud', 'Ingreso_Anual_USD', 'Género', 'Frecuencia_Compra'}
    columnas_faltantes = columnas_necesarias - set(df.columns)
    if columnas_faltantes:
        st.error(f"⚠️ Error: Faltan las siguientes columnas en el DataFrame: {columnas_faltantes}")
        return  # Detiene la ejecución para evitar errores

    # 📌 Seleccionar los Top-N compradores con mayores ingresos
    top_compradores = df.nlargest(top_n, 'Ingreso_Anual_USD')

    # 📌 Función para calcular matrices de distancia geodésicas
    def matriz_distancias(latitudes, longitudes):
        """Calcula la matriz de distancias geodésicas usando NumPy sin ciclos for."""
        coords = np.radians(np.column_stack((latitudes, longitudes)))
        lat1, lon1 = coords[:, None, 0], coords[:, None, 1]
        lat2, lon2 = coords[:, 0], coords[:, 1]

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371 * c  # Distancia en kilómetros

    # 📌 Matriz de distancias generales
    distancias_generales = matriz_distancias(top_compradores['Latitud'].values, top_compradores['Longitud'].values)

    # 📌 Crear variables categóricas como máscaras booleanas
    generos_dummies = pd.get_dummies(top_compradores['Género']).values
    frecuencia_dummies = pd.get_dummies(top_compradores['Frecuencia_Compra']).values

    # 📌 Aplicar máscaras booleanas en NumPy para obtener subconjuntos de coordenadas
    latitudes_genero = generos_dummies.T @ top_compradores['Latitud'].values
    longitudes_genero = generos_dummies.T @ top_compradores['Longitud'].values
    latitudes_frecuencia = frecuencia_dummies.T @ top_compradores['Latitud'].values
    longitudes_frecuencia = frecuencia_dummies.T @ top_compradores['Longitud'].values

    # 📌 Calcular matrices de distancia para cada subconjunto
    distancias_por_genero = matriz_distancias(latitudes_genero, longitudes_genero)
    distancias_por_frecuencia = matriz_distancias(latitudes_frecuencia, longitudes_frecuencia)

    # 📌 Mostrar resultados en Streamlit
    st.subheader("📊 Análisis de Distancias Discriminadas")
    
    with st.expander("📌 Distancias Generales"):
        st.write("Matriz de Distancias Generales (Top-N Compradores):")
        st.dataframe(pd.DataFrame(distancias_generales))

    with st.expander("📌 Distancias por Género"):
        st.write("Matriz de Distancias entre Compradores según Género:")
        st.dataframe(pd.DataFrame(distancias_por_genero))

    with st.expander("📌 Distancias por Frecuencia de Compra"):
        st.write("Matriz de Distancias entre Compradores según Frecuencia de Compra:")
        st.dataframe(pd.DataFrame(distancias_por_frecuencia))
    
    st.success("✅ Análisis completado y mostrado en la app.")

    
def mapa_personalizado(df, variables):
    """
    Genera un mapa filtrando los datos según las variables seleccionadas por el usuario.
    
    Parámetros:
    - df (DataFrame): Datos de clientes con 'Latitud' y 'Longitud'.
    - variables (dict): Variables seleccionadas con sus rangos {columna: (min, max)}.

    Muestra el mapa directamente en la app de Streamlit.
    """

    # 📌 Verificar si el DataFrame contiene las columnas necesarias
    columnas_necesarias = {"Latitud", "Longitud"}
    columnas_faltantes = columnas_necesarias - set(df.columns)
    
    if columnas_faltantes:
        st.error(f"⚠️ Error: Faltan las columnas necesarias en el DataFrame: {columnas_faltantes}")
        return

    # 📌 Aplicar filtros según las variables seleccionadas
    try:
        query_str = " & ".join([f"(@rango[0] <= `{col}` <= @rango[1])" for col in variables])
        df_filtrado = df.query(query_str, local_dict={'rango': variables})
    except Exception as e:
        st.error(f"⚠️ Error al aplicar filtros: {str(e)}")
        return

    # 📌 Validar si hay datos después del filtrado
    if df_filtrado.empty:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
        return

    # 📌 Crear un GeoDataFrame con los puntos filtrados
    gdf = gpd.GeoDataFrame(df_filtrado, 
                           geometry=gpd.points_from_xy(df_filtrado['Longitud'], df_filtrado['Latitud']), 
                           crs="EPSG:4326")

    # 📌 Cargar mapa base desde GeoPandas
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # 📌 Crear la figura y el eje para el mapa
    fig, ax = plt.subplots(figsize=(10, 6))

    # 📌 Dibujar el mapa base
    world.plot(ax=ax, color="lightgrey", edgecolor="black")

    # 📌 Graficar los puntos filtrados con un tamaño adecuado
    gdf.plot(ax=ax, markersize=30, color="blue", alpha=0.7)

    # 📌 Agregar etiquetas y formato
    plt.title("📍 Mapa Personalizado de Clientes", fontsize=14)
    plt.xlabel("Longitud", fontsize=12)
    plt.ylabel("Latitud", fontsize=12)

    # 📌 Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    # 📌 Mensaje de éxito
    st.success("✅ Mapa generado con éxito.")
    
def analizar_cluster_frecuencia(df, n_clusters=3):
    """
    Realiza análisis de clúster basado en la ubicación (Latitud, Longitud) y la Frecuencia de Compra.
    Muestra un mapa con la segmentación de clientes en diferentes clusters.

    Parámetros:
    - df: DataFrame con los datos de clientes.
    - n_clusters: Número de clústeres a formar (por defecto 3).
    """

    # Convertir la frecuencia de compra en valores numéricos
    frecuencia_map = {"Baja": 1, "Media": 2, "Alta": 3}
    df["Frecuencia_Numerica"] = df["Frecuencia_Compra"].map(frecuencia_map)

    # Selección de variables para clustering
    X = df[["Latitud", "Longitud", "Frecuencia_Numerica"]].dropna()

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[X.index, "Cluster"] = kmeans.fit_predict(X)

    # Crear un GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitud"], df["Latitud"]), crs="EPSG:4326")

    # Cargar mapa base
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip")

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dibujar mapa base
    world.plot(ax=ax, color="lightgrey", edgecolor="black")

    # Graficar clientes según su cluster
    gdf.plot(ax=ax, markersize=1, column="Cluster", cmap="viridis", legend=True, alpha=0.7)

    # Etiquetas y título
    plt.title(f"Clúster de Clientes por Ubicación y Frecuencia de Compra ({n_clusters} Clusters)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")

    # Mostrar gráfico en Streamlit
    st.pyplot(fig)
    
def cargar_datos():
    ruta = "https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/aplicaciones/analisis_clientes.csv"
    df = pd.read_csv(ruta)
    df[['Latitud', 'Longitud']] = df[['Latitud', 'Longitud']].apply(pd.to_numeric, errors='coerce')
    df = df.interpolate(method='linear', limit_direction='both')
    geometry = gpd.points_from_xy(df['Longitud'], df['Latitud'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return df, gdf

df, gdf = cargar_datos()

# Función para cargar CSS
@st.cache_data
def cargar_css(css_link=None, css_file=None):
    if css_link:
        st.markdown(f'<link rel="stylesheet" href="{css_link}">', unsafe_allow_html=True)
        st.session_state.css_cargado = True
    elif css_file:
        css = css_file.read().decode("utf-8")
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        st.session_state.css_cargado = True

# Interfaz para subir o ingresar CSS
st.sidebar.header("Personalización CSS")
css_link = st.sidebar.text_input("Ingresa URL del CSS:")
css_file = st.sidebar.file_uploader("O carga un archivo CSS", type=["css"])

if css_link or css_file:
    cargar_css(css_link, css_file)

# Título de la App
st.title("Análisis de Clientes con Streamlit")

# Mapa de calor de ingresos con mapa del mundo
def mapa_calor_ingresos(df):
    ruta_0 = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    df_mapa = gpd.read_file(ruta_0)

    fig, ax = plt.subplots(figsize=(12, 6))
    df_mapa.plot(ax=ax, color="lightgrey", edgecolor="black")
    
    sns.kdeplot(
        data=df, x="Longitud", y="Latitud", weights=df["Ingreso_Anual_USD"],
        cmap="inferno", fill=True, alpha=0.6, levels=50, ax=ax
    )
    plt.title("Mapa de Calor de Ingresos Anuales de Clientes")
    st.pyplot(fig)

# Mapa interactivo de ubicaciones de clientes
def mapa_ubicaciones_clientes(df):
    """
    Muestra la ubicación de los clientes en un mapa mundial sin usar bucles for.
    Utiliza GeoPandas y Matplotlib para optimizar el rendimiento.
    """

    # Convertir coordenadas a valores numéricos y eliminar valores nulos
    df[['Latitud', 'Longitud']] = df[['Latitud', 'Longitud']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['Latitud', 'Longitud'])

    # Crear un GeoDataFrame con las ubicaciones de los clientes
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitud'], df['Latitud']), crs="EPSG:4326")

    # Cargar mapa base desde Natural Earth (capa mundial de países)
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip")

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dibujar el mapa base
    world.plot(ax=ax, color="lightgrey", edgecolor="black")
    
    # Graficar los puntos de los clientes
    gdf.plot(ax=ax, markersize=1, color="red", alpha=0.7)

    # Etiquetas
    plt.title("Mapa de Ubicaciones de Clientes")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)
# Gráfico de barras por género y frecuencia de compra
def graficar_barras_genero_frecuencia(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="Género", palette="coolwarm", ax=axes[0])
    axes[0].set_title("Distribución por Género")
    sns.countplot(data=df, x="Frecuencia_Compra", palette="viridis", ax=axes[1])
    axes[1].set_title("Distribución por Frecuencia de Compra")
    st.pyplot(fig)

# Análisis de correlación
def analizar_correlacion(df):
    """
    Analiza la correlación entre Edad e Ingreso Anual:
    - Globalmente
    - Segmentado por Género
    - Segmentado por Frecuencia de Compra
    
    Muestra gráficos de dispersión con líneas de tendencia.
    """
    
    plt.figure(figsize=(15, 10))

    # Gráfico 1: Correlación Global
    plt.subplot(2, 2, 1)
    sns.regplot(x=df["Edad"], y=df["Ingreso_Anual_USD"], \
          scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
    plt.title("Correlación Global entre Edad e Ingreso Anual")

    # Gráfico 2: Correlación por Género
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x="Edad", y="Ingreso_Anual_USD", \
          hue="Género", alpha=0.6)
    plt.title("Correlación por Género")
    
    # Gráfico 3: Correlación por Frecuencia de Compra
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x="Edad", y="Ingreso_Anual_USD", \
          hue="Frecuencia_Compra", alpha=0.6)
    plt.title("Correlación por Frecuencia de Compra")

    # Gráfico 4: Correlación por Género con Línea de Tendencia
    plt.subplot(2, 2, 4)
    sns.lmplot(data=df, x="Edad", y="Ingreso_Anual_USD", hue="Género", aspect=1.5)
    plt.title("Regresión Lineal por Género")

    plt.tight_layout()
    st.pyplot(plt)

# Interfaz de selección en Streamlit
st.sidebar.header("Opciones de Visualización")
if st.session_state.css_cargado:
    opcion = st.sidebar.selectbox("Selecciona un análisis", ["Mapa de Calor", "Mapa de Ubicaciones", "Distribución de Clientes", "Clúster de Frecuencia", "Análisis de Correlación", "Mapas personalizados", "Distancias Discriminadas"])

    if opcion == "Mapa de Calor":
        mapa_calor_ingresos(df)
    elif opcion == "Mapa de Ubicaciones":
        mapa_ubicaciones_clientes(df)
    elif opcion == "Distribución de Clientes":
        graficar_barras_genero_frecuencia(df)
    elif opcion == "Clúster de Frecuencia":
        analizar_cluster_frecuencia(df)
    elif opcion == "Análisis de Correlación":
        analizar_correlacion(df)
    elif opcion == "Mapas personalizados":
        mapa_personalizado(df)
    elif opcion == "Distancias Discriminadas":
        calcular_distancias(df)
else:
    st.sidebar.warning("Por favor, carga un archivo CSS o ingresa un enlace antes de visualizar los análisis.")

if df is not None:
    st.sidebar.text("Datos cargados con éxito")
