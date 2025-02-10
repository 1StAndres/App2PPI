import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

# Mapa de calor de ingresos
def mapa_calor_ingresos(df):
    ruta_0 = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    df_mapa = gpd.read_file(ruta_0)

    # Crear la figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Dibujar el mapa base
    df_mapa.plot(ax=ax, color="lightgrey", edgecolor="black")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(
        data=df, x="Longitud", y="Latitud", weights=df["Ingreso_Anual_USD"],
        cmap="inferno", fill=True, alpha=0.6, levels=50
    )
    plt.title("Mapa de Calor de Ingresos Anuales")
    st.pyplot(fig)

# Gráfico de barras por género y frecuencia de compra
def graficar_barras_genero_frecuencia(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="Género", palette="coolwarm", ax=axes[0])
    axes[0].set_title("Distribución por Género")
    sns.countplot(data=df, x="Frecuencia_Compra", palette="viridis", ax=axes[1])
    axes[1].set_title("Distribución por Frecuencia de Compra")
    st.pyplot(fig)

# Análisis de clúster de clientes
def analizar_cluster_frecuencia(df, n_clusters=3):
    frecuencia_map = {"Baja": 1, "Media": 2, "Alta": 3}
    df["Frecuencia_Numerica"] = df["Frecuencia_Compra"].map(frecuencia_map)
    X = df[["Frecuencia_Numerica", "Edad", "Ingreso_Anual_USD"]].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[X.index, "Cluster"] = kmeans.fit_predict(X)
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Edad", y="Ingreso_Anual_USD", hue="Cluster", palette="viridis", alpha=0.7)
    plt.title("Clusters de Clientes")
    st.pyplot(fig)

# Interfaz de selección en Streamlit
st.sidebar.header("Opciones de Visualización")
if st.session_state.css_cargado:
    opcion = st.sidebar.selectbox("Selecciona un análisis", ["Mapa de Calor", "Distribución de Clientes", "Clúster de Frecuencia"])

    if opcion == "Mapa de Calor":
        mapa_calor_ingresos(df)
    elif opcion == "Distribución de Clientes":
        graficar_barras_genero_frecuencia(df)
    elif opcion == "Clúster de Frecuencia":
        analizar_cluster_frecuencia(df)
else:
    st.sidebar.warning("Por favor, carga un archivo CSS o ingresa un enlace antes de visualizar los análisis.")

if df is not None:
    st.sidebar.text("Datos cargados con éxito")

