import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Point

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

# Título de la App
st.title("Análisis de Clientes con Streamlit")

# Mapa de calor de ingresos
def mapa_calor_ingresos(df):
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
opcion = st.sidebar.selectbox("Selecciona un análisis", ["Mapa de Calor", "Distribución de Clientes", "Clúster de Frecuencia"])

if opcion == "Mapa de Calor":
    mapa_calor_ingresos(df)
elif opcion == "Distribución de Clientes":
    graficar_barras_genero_frecuencia(df)
elif opcion == "Clúster de Frecuencia":
    analizar_cluster_frecuencia(df)

st.sidebar.text("Datos cargados con éxito")
