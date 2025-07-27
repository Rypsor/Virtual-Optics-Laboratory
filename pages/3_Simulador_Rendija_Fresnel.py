# pages/3_Simulador_Rendija_Fresnel.py

"""
Simulador de Difracción por Rendija Única

Este script de Streamlit permite visualizar la difracción de la luz a través de
una sola rendija. El propósito principal es mostrar la evolución del patrón de
difracción desde el régimen de campo cercano (Fresnel) hasta el régimen de campo
lejano (Fraunhofer) a medida que cambia la distancia entre la rendija y la pantalla.

El simulador presenta varias visualizaciones interactivas:
1. Comparación directa entre los patrones de Fresnel y Fraunhofer para una distancia dada.
2. Un mapa de calor que muestra la evolución del patrón a lo largo de la distancia.
3. Un gráfico que muestra la convergencia (diferencia) entre ambos patrones.
4. Una representación visual del patrón de difracción en la pantalla.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

# --- Funciones de Cálculo ---

def calculate_fresnel(b_val, lambda_val, D_val, screen_width, resolution=1000):
    """
    Calcula el patrón de intensidad para la difracción de Fresnel y Fraunhofer.

    Esta es la función central de física que calcula la intensidad de la luz en la
    pantalla basándose en los parámetros de la rendija y la luz.

    Args:
        b_val (float): Ancho de la rendija 'b' en metros.
        lambda_val (float): Longitud de onda de la luz 'λ' en metros.
        D_val (float): Distancia de la rendija a la pantalla 'D' en metros.
        screen_width (float): Ancho de la pantalla de visualización en metros.
        resolution (int): Número de puntos a calcular en la pantalla.

    Returns:
        tuple: Una tupla conteniendo:
            - x_prime (np.array): Coordenadas de posición en la pantalla (m).
            - intensity_norm (np.array): Intensidad normalizada del patrón de Fresnel.
            - intensity_fraunhofer (np.array): Intensidad normalizada del patrón de Fraunhofer.
            - NF (float): El número de Fresnel calculado para esta configuración.
    """
    # Cálculo del Número de Fresnel (parámetro adimensional)
    NF = (b_val**2) / (lambda_val * D_val)

    # Coordenadas en la pantalla de visualización
    x_prime = np.linspace(-screen_width / 2, screen_width / 2, resolution)

    # Factor de escala para los argumentos de las integrales de Fresnel
    u_factor = np.sqrt(2 / (lambda_val * D_val))

    # Límites de integración para la integral de Fresnel
    u1 = u_factor * (x_prime - b_val / 2)
    u2 = u_factor * (x_prime + b_val / 2)

    # Cálculo de las integrales de Fresnel (S y C) usando la librería SciPy
    Su1, Cu1 = fresnel(u1)
    Su2, Cu2 = fresnel(u2)

    # Cálculo de la intensidad según la teoría de la difracción de Fresnel
    intensity = 0.5 * ((Cu2 - Cu1)**2 + (Su2 - Su1)**2)

    # Normalización de la intensidad para que el valor máximo sea 1.0
    if np.max(intensity) > 0:
        intensity_norm = intensity / np.max(intensity)
    else:
        intensity_norm = intensity

    # Cálculo del patrón de difracción de Fraunhofer (aproximación de campo lejano)
    # usando la función sinc
    sinc_arg = (np.pi * b_val * x_prime) / (lambda_val * D_val)
    intensity_fraunhofer = (np.sinc(sinc_arg / np.pi))**2

    return x_prime, intensity_norm, intensity_fraunhofer, NF

@st.cache_data
def precompute_distance_data(b_um, start_m, end_m, step_m, lambda_nm, screen_mm):
    """
    Pre-calcula los patrones de difracción para un rango de distancias 'D'.

    Optimiza la respuesta del slider principal, evitando recalcular en tiempo real.
    El decorador @st.cache_data almacena los resultados en caché.

    Args:
        b_um (float): Ancho de la rendija en micrómetros.
        start_m (float): Distancia inicial del rango en metros.
        end_m (float): Distancia final del rango en metros.
        step_m (float): Paso entre distancias en metros.
        lambda_nm (float): Longitud de onda en nanómetros.
        screen_mm (float): Ancho de la pantalla en milímetros.

    Returns:
        dict: Un diccionario donde las claves son las distancias 'D' y los valores
              son los resultados de calculate_fresnel para esa distancia.
    """
    results = {}
    # Conversión de unidades a SI (metros)
    lambda_val = lambda_nm * 1e-9
    screen_width = screen_mm * 1e-3
    b_val = b_um * 1e-6

    # Itera sobre el rango de distancias y calcula el patrón para cada una
    for D_val in np.arange(start_m, end_m + step_m, step_m):
        D_key = round(D_val, 4) # Clave redondeada para consistencia
        results[D_key] = calculate_fresnel(b_val, lambda_val, D_key, screen_width, resolution=1000)
    return results

@st.cache_data
def precompute_heatmap_data(b_um, lambda_nm, D_max_m, screen_mm):
    """
    Pre-calcula los datos para el mapa de calor de evolución (intensidad vs. distancia).

    Args:
        b_um (float): Ancho de la rendija en micrómetros.
        lambda_nm (float): Longitud de onda en nanómetros.
        D_max_m (float): Distancia máxima para el mapa de calor en metros.
        screen_mm (float): Ancho de la pantalla en milímetros.

    Returns:
        tuple: Una tupla conteniendo:
            - D_array (np.array): Eje de distancias para el mapa.
            - x_prime_array (np.array): Eje de posiciones para el mapa.
            - intensity_map (np.array): Matriz 2D de intensidades.
    """
    lambda_val = lambda_nm * 1e-9
    screen_width = screen_mm * 1e-3
    b_val = b_um * 1e-6
    # Resoluciones para los ejes del mapa de calor
    heatmap_x_res = 150
    heatmap_y_res = 400
    D_array = np.linspace(0.001, D_max_m, heatmap_x_res)
    x_prime_array = np.linspace(-screen_width / 2, screen_width / 2, heatmap_y_res)

    intensity_map = np.zeros((len(x_prime_array), len(D_array)))
    # Rellena la matriz de intensidad calculando el patrón para cada distancia
    for i, D_val in enumerate(D_array):
        _, intensity_1D, _, _ = calculate_fresnel(b_val, lambda_val, D_val, screen_width, resolution=heatmap_y_res)
        intensity_map[:, i] = intensity_1D
    return D_array, x_prime_array, intensity_map

@st.cache_data
def precompute_difference_data(b_um, start_m, end_m, lambda_nm, screen_mm):
    """
    Pre-calcula la diferencia entre los patrones de Fresnel y Fraunhofer.

    Esto se usa para el gráfico de convergencia, que muestra cómo el patrón
    de Fresnel se aproxima al de Fraunhofer a medida que aumenta la distancia.

    Args:
        b_um (float): Ancho de la rendija en micrómetros.
        start_m (float): Distancia inicial del rango en metros.
        end_m (float): Distancia final del rango en metros.
        lambda_nm (float): Longitud de onda en nanómetros.
        screen_mm (float): Ancho de la pantalla en milímetros.

    Returns:
        list: Una lista de tuplas, cada una con datos para una distancia específica.
    """
    diff_results = []
    # Selecciona 10 puntos de distancia para mostrar en el gráfico
    D_values = np.linspace(start_m, end_m, 10)
    for D_val in D_values:
        lambda_val = lambda_nm * 1e-9
        screen_width = screen_mm * 1e-3
        b_val = b_um * 1e-6
        x_prime, fres, frau, nf_val = calculate_fresnel(b_val, lambda_val, D_val, screen_width)
        difference = fres - frau # Calcula la diferencia punto a punto
        diff_results.append((D_val, x_prime, difference, nf_val))
    return diff_results

@st.cache_data
def precompute_slit_heatmap(b_um, lambda_nm, D_m, screen_mm):
    """
    Genera una imagen 2D que simula la apariencia visual del patrón de difracción.

    Toma el patrón de intensidad 1D y lo repite verticalmente para crear una imagen.

    Args:
        b_um (float): Ancho de la rendija en micrómetros.
        lambda_nm (float): Longitud de onda en nanómetros.
        D_m (float): Distancia seleccionada en metros.
        screen_mm (float): Ancho de la pantalla en milímetros.

    Returns:
        tuple: Una tupla conteniendo:
            - x_prime (np.array): Coordenadas de posición.
            - slit_map (np.array): Matriz 2D que representa el patrón visual.
    """
    lambda_val = lambda_nm * 1e-9
    screen_width = screen_mm * 1e-3
    b_val = b_um * 1e-6
    y_res = 512 # Resolución vertical de la imagen
    x_res = 1024 # Resolución horizontal de la imagen

    x_prime, intensity_1D, _, _ = calculate_fresnel(b_val, lambda_val, D_m, screen_width, resolution=x_res)
    # La función `tile` de NumPy repite el array 1D para formar una imagen 2D
    slit_map = np.tile(intensity_1D, (y_res, 1))
    return x_prime, slit_map


# --- Interfaz de Streamlit ---

# Configuración de la página (título en la pestaña del navegador, ícono, layout)
st.set_page_config(page_title="Evolución de la Difracción", page_icon="🗺️", layout="wide")
st.title("🗺️ Evolución de Fresnel a Fraunhofer")

# --- Inicialización del Estado de la Sesión ---
# st.session_state permite que los valores de los widgets persistan entre interacciones.
# Se inicializan los valores por defecto si no existen previamente.
if "lambda_nm" not in st.session_state:
    st.session_state.lambda_nm = 592.0
if "b_um" not in st.session_state:
    st.session_state.b_um = 800.0
if "screen_mm" not in st.session_state:
    st.session_state.screen_mm = 10.0
if "D_start_m" not in st.session_state:
    st.session_state.D_start_m = 0.001
if "D_end_m" not in st.session_state:
    st.session_state.D_end_m = 5.0
if "step_m" not in st.session_state:
    st.session_state.step_m = 0.01

# --- Controles en la Barra Lateral ---
st.sidebar.header("Parámetros Fijos")
# Sliders para los parámetros físicos principales.
# Streamlit actualiza st.session_state automáticamente gracias al argumento 'key'.
st.sidebar.slider("Longitud de Onda (λ, nm)", 380.0, 750.0, key="lambda_nm", step=0.1, format="%.1f")
st.sidebar.slider("Ancho de la Rendija 'b' (μm)", 100.0, 3000.0, key="b_um", step=1.0, format="%.1f")
st.sidebar.slider("Ancho de Visualización (mm)", 1.0, 20.0, key="screen_mm", step=0.1, format="%.1f")

st.sidebar.markdown("---")
st.sidebar.header("Rango del Slider de Distancia")
# Controles para definir el rango del slider de distancia principal
D_start_m = st.sidebar.number_input("Distancia INICIAL (m)", 0.001, 10.0, st.session_state.D_start_m, step=0.001, format="%.3f")
D_end_m = st.sidebar.number_input("Distancia FINAL (m)", 0.001, 10.0, st.session_state.D_end_m, step=0.1, format="%.2f")
step_m = st.sidebar.number_input("Paso del slider (m)", 0.001, 1.0, st.session_state.step_m, step=0.001, format="%.3f")

# Actualiza el estado de la sesión con los valores de rango definidos por el usuario
st.session_state.D_start_m = D_start_m
st.session_state.D_end_m = D_end_m
st.session_state.step_m = step_m

# --- Lógica Principal y Visualización ---

# Valida que el rango de distancia sea coherente
if st.session_state.D_start_m >= st.session_state.D_end_m:
    st.error("Error: La 'Distancia FINAL' debe ser mayor que la 'Distancia INICIAL'. Por favor, ajusta los valores en la barra lateral.")
else:
    # Muestra un spinner mientras se ejecutan los cálculos pesados
    with st.spinner('Calculando patrones de difracción...'):
        slider_data = precompute_distance_data(st.session_state.b_um, st.session_state.D_start_m, st.session_state.D_end_m, st.session_state.step_m, st.session_state.lambda_nm, st.session_state.screen_mm)
        D_map, x_map, intensity_map = precompute_heatmap_data(st.session_state.b_um, st.session_state.lambda_nm, st.session_state.D_end_m, st.session_state.screen_mm)
        difference_data = precompute_difference_data(st.session_state.b_um, st.session_state.D_start_m, st.session_state.D_end_m, st.session_state.lambda_nm, st.session_state.screen_mm)

    # Calcula la distancia a la cual el Número de Fresnel es 1 (frontera Fraunhofer)
    b_val_fixed = st.session_state.b_um * 1e-6
    lambda_val_fixed = st.session_state.lambda_nm * 1e-9
    D_at_NF1 = (b_val_fixed**2) / lambda_val_fixed
    st.sidebar.metric(label="Distancia para NF=1", value=f"{D_at_NF1:.3f} m")

    st.markdown("---")
    # Slider principal para que el usuario seleccione la distancia a visualizar
    D_m_selected = st.slider("Mueve para cambiar la Distancia 'D' (m)", st.session_state.D_start_m, st.session_state.D_end_m, st.session_state.D_start_m, st.session_state.step_m)

    # --- Gráfico 1: Comparativa y Mapa de Evolución ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.5]})

    # CORRECCIÓN: Busca la clave más cercana para evitar errores de punto flotante
    # Esto asegura que siempre se encuentre un valor válido en los datos precalculados.
    closest_D_key = min(slider_data.keys(), key=lambda k: abs(k - D_m_selected))
    x_prime, intensity_fresnel, intensity_fraunhofer, NF = slider_data[closest_D_key]

    # Gráfico de la izquierda: Comparación 1D de patrones
    ax1.plot(x_prime * 1000, intensity_fresnel, label='Fresnel', color='dodgerblue', lw=2)
    ax1.plot(x_prime * 1000, intensity_fraunhofer, 'k--', label='Fraunhofer', lw=2)
    ax1.set_title(f'Patrón a D = {closest_D_key:.3f} m  |  $N_F$ ≈ {NF:.2f}', fontsize=14)
    ax1.set_xlabel("Posición x' (mm)"); ax1.set_ylabel('Intensidad Normalizada'); ax1.grid(True, linestyle=':'); ax1.legend(); ax1.set_ylim(0, 1.4)

    # Gráfico de la derecha: Mapa de calor 2D de la evolución
    # Se usa una escala logarítmica (log1p) para mejorar el contraste visual
    log_intensity_map = np.log1p(intensity_map)
    im = ax2.imshow(log_intensity_map, aspect='auto', origin='lower', extent=[D_map.min(), D_map.max(), x_map.min() * 1000, x_map.max() * 1000], cmap='hot')
    fig1.colorbar(im, ax=ax2, label='Intensidad (escala log)', fraction=0.046, pad=0.04)
    ax2.set_title(f'Evolución para b = {st.session_state.b_um} μm', fontsize=14)
    ax2.set_xlabel("Distancia 'D' (m)"); ax2.set_ylabel("Posición x' (mm)")
    # Líneas verticales para indicar la vista actual y la frontera NF=1
    ax2.axvline(x=D_m_selected, color='cyan', linestyle='--', lw=2, label=f'Vista Actual ({D_m_selected:.2f}m)')
    if D_map.min() <= D_at_NF1 <= D_map.max(): ax2.axvline(x=D_at_NF1, color='magenta', linestyle=':', lw=3, label=f'Frontera NF=1 ({D_at_NF1:.2f}m)')
    ax2.legend(); fig1.tight_layout(pad=2.0); st.pyplot(fig1)

    # --- Gráfico 2: Convergencia a Fraunhofer ---
    st.markdown("---")
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap('plasma') # Mapa de colores para las diferentes distancias
    ax3.axhline(0, color='red', linestyle='--', lw=1.5, label='Diferencia Cero')
    # Itera sobre los datos de diferencia precalculados y los grafica
    for i, data in enumerate(difference_data):
        D_val, x_prime_diff, difference, nf_val = data
        color = cmap(i / (len(difference_data) -1))
        label = f'D={D_val:.2f}m (NF≈{nf_val:.1f})'
        ax3.plot(x_prime_diff * 1000, difference, color=color, label=label)
    ax3.set_title('Convergencia a Fraunhofer', fontsize=16); ax3.set_xlabel("Posición x' (mm)"); ax3.set_ylabel('Diferencia (Fresnel - Fraunhofer)'); ax3.grid(True, linestyle=':')
    ax3.legend(title='Distancia D y N. Fresnel', fontsize=9, ncol=2); ax3.set_xlim(-st.session_state.screen_mm/2, st.session_state.screen_mm/2)
    st.pyplot(fig2)

    # --- Gráfico 3: Visualización del Patrón ---
    st.markdown("---")
    st.subheader("Visualización del Patrón de la Rendija")
    with st.spinner('Generando visualización del patrón...'):
        # Usa la clave más cercana para consistencia con el primer gráfico
        x_slit, slit_map = precompute_slit_heatmap(st.session_state.b_um, st.session_state.lambda_nm, closest_D_key, st.session_state.screen_mm)
        fig3, ax4 = plt.subplots(figsize=(12, 4))
        ax4.imshow(slit_map, cmap='gray', aspect='auto', extent=[x_slit.min()*1000, x_slit.max()*1000, -1, 1])
        ax4.set_title(f"Apariencia Visual del Patrón a D = {closest_D_key:.3f} m")
        ax4.set_xlabel("Posición en la pantalla x' (mm)")
        ax4.get_yaxis().set_visible(False) # Oculta el eje Y que no aporta información
        st.pyplot(fig3)