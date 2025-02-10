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
def visualizar_mapa_clientes(df):
    """
    Superpone los clientes en el mapa del mundo usando matplotlib y geopandas.
    Genera tres mapas:
    - Ubicación global de los clientes.
    - Segmentación por Género.
    - Segmentación por Frecuencia de Compra.
    """

    # Asegurar que las coordenadas sean numéricas
    df[['Latitud', 'Longitud']] = df[['Latitud', 'Longitud']].apply(pd.to_numeric, errors='coerce')

    # Aplicar interpolación para corregir valores faltantes
    df_interpolado = df.interpolate(method='linear', limit_direction='both')

    # Crear geometría de puntos
    geometry = gpd.points_from_xy(df_interpolado['Longitud'], df_interpolado['Latitud'])

    # Crear GeoDataFrame con sistema de referencia espacial (EPSG:4326 - WGS 84)
    gdf = gpd.GeoDataFrame(df_interpolado, geometry=geometry, crs="EPSG:4326")

    # Cargar mapa base
    ruta_mapa = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    df_mapa = gpd.read_file(ruta_mapa)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Mapa 1: Clientes Globales
    df_mapa.plot(ax=axes[0], color="lightgrey", edgecolor="black")
    gdf.plot(ax=axes[0], markersize=20, color="red", alpha=0.6)
    axes[0].set_title("Ubicación Global de los Clientes")

    # Mapa 2: Clientes por Género
    df_mapa.plot(ax=axes[1], color="lightgrey", edgecolor="black")
    gdf.plot(ax=axes[1], markersize=20, column="Género", cmap="coolwarm", legend=True, alpha=0.6)
    axes[1].set_title("Clientes por Género")

    # Mapa 3: Clientes por Frecuencia de Compra
    df_mapa.plot(ax=axes[2], color="lightgrey", edgecolor="black")
    gdf.plot(ax=axes[2], markersize=20, column="Frecuencia_Compra", cmap="viridis", legend=True, alpha=0.6)
    axes[2].set_title("Clientes por Frecuencia de Compra")

    # Ajustar diseño
    plt.tight_layout()
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
    opcion = st.sidebar.selectbox("Selecciona un análisis", ["Mapa de Calor", "Mapa de Ubicaciones", "Distribución de Clientes", "Clúster de Frecuencia", "Análisis de Correlación"])

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
else:
    st.sidebar.warning("Por favor, carga un archivo CSS o ingresa un enlace antes de visualizar los análisis.")

if df is not None:
    st.sidebar.text("Datos cargados con éxito")
