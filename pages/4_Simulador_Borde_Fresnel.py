# pages/4_Simulador_Borde_Fresnel.py

"""
Simulador de Difracción de Fresnel por Bordes

Este script de Streamlit crea una herramienta interactiva para visualizar y
estudiar la difracción de Fresnel. Permite al usuario simular el patrón de
difracción generado por uno o dos bordes rectos.

Funcionalidades principales:
- Simulación para un único borde semi-infinito.
- Simulación para una rendija, activando un segundo borde.
- Controles interactivos (sliders) para ajustar la longitud de onda, la distancia
  a la pantalla, el ancho de visualización y la posición/separación de los bordes.
- Cálculo y visualización del Número de Fresnel ($N_F$) para el caso de la rendija.
- Generación de dos gráficos:
  1. Un perfil de intensidad 1D del patrón de difracción.
  2. Una representación visual 2D (mapa de calor) que simula la apariencia del
     patrón en una pantalla.

El código utiliza el estado de sesión de Streamlit (st.session_state) para
mantener la interactividad y el decorador @st.cache_data para optimizar el
rendimiento de los cálculos.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

# --- Funciones de Cálculo ---

@st.cache_data
def calculate_diffraction_pattern(lambda_val, D_val, screen_width, edge_1_pos, enable_edge_2=False, edge_2_pos=0.0, resolution=2000):
    """
    Calcula el patrón de difracción de Fresnel para uno o dos bordes.

    Esta función es el motor de cálculo principal. Puede operar en dos modos:
    1. Borde único (enable_edge_2=False): Calcula el patrón para una obstrucción
       semi-infinita.
    2. Doble borde / Rendija (enable_edge_2=True): Calcula el patrón para una
       apertura definida por dos bordes.

    Args:
        lambda_val (float): Longitud de onda de la luz en metros.
        D_val (float): Distancia del obstáculo a la pantalla en metros.
        screen_width (float): Ancho de la pantalla de visualización en metros.
        edge_1_pos (float): Posición del primer borde en metros.
        enable_edge_2 (bool): Flag para activar el modo de segundo borde (rendija).
        edge_2_pos (float): Posición del segundo borde en metros (si está activado).
        resolution (int): Número de puntos para calcular el patrón.

    Returns:
        tuple: Una tupla conteniendo:
            - x_prime (np.array): Coordenadas de posición en la pantalla (m).
            - intensity (np.array): Intensidad relativa del patrón de difracción.
    """
    # Genera el eje de coordenadas en la pantalla
    x_prime = np.linspace(-screen_width / 2, screen_width / 2, resolution)
    # Factor de escala común para los argumentos de las integrales de Fresnel
    u_factor = np.sqrt(2 / (lambda_val * D_val))

    if not enable_edge_2:
        # --- Cálculo para Borde Único ---
        # El patrón depende de la integral de Fresnel evaluada desde -inf hasta v
        v = (x_prime - edge_1_pos) * u_factor
        S_v, C_v = fresnel(v)
        intensity = 0.5 * ((C_v + 0.5)**2 + (S_v + 0.5)**2)
    else:
        # --- Cálculo para Doble Borde (Rendija) ---
        # El patrón depende de la diferencia de las integrales de Fresnel
        # evaluadas en las posiciones de cada borde.
        u1 = (x_prime - edge_1_pos) * u_factor
        u2 = (x_prime - edge_2_pos) * u_factor
        Su1, Cu1 = fresnel(u1)
        Su2, Cu2 = fresnel(u2)
        intensity = 0.5 * ((Cu2 - Cu1)**2 + (Su2 - Su1)**2)
        # Normaliza la intensidad para el caso de la rendija
        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)

    return x_prime, intensity

def generate_heatmap(x_coords, intensity_1D):
    """
    Crea un mapa de calor 2D a partir de un patrón de intensidad 1D.

    Simula la apariencia visual del patrón de difracción repitiendo la línea
    de intensidad verticalmente.

    Args:
        x_coords (np.array): Coordenadas del eje x (no se usa directamente pero es parte del estándar).
        intensity_1D (np.array): El perfil de intensidad 1D a visualizar.

    Returns:
        np.array: Una matriz 2D (imagen) que representa el patrón.
    """
    y_res = 512 # Resolución vertical de la imagen
    # np.tile repite el array 1D `y_res` veces para formar la imagen
    heatmap = np.tile(intensity_1D, (y_res, 1))
    return heatmap

# --- Interfaz de Streamlit ---
# Configuración inicial de la página
st.set_page_config(page_title="Difracción de Fresnel por Bordes", page_icon="🔪", layout="wide")
st.title("🔪 Difracción de Fresnel por Bordes")
st.markdown("Usa los controles para simular la difracción de uno o dos bordes (rendija).")

# --- Inicialización del Estado de la Sesión ---
# Define los valores por defecto para los widgets.
# st.session_state se usa para mantener los valores de los controles entre
# las interacciones del usuario. Este bloque se ejecuta solo una vez.
defaults = {
    "lambda_nm": 550.0, "D_m": 1.0, "screen_mm": 10.0,
    "edge_1_pos_mm": -1.0,
    "slit_separation_mm": 0.75
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Controles en la Barra Lateral ---
st.sidebar.header("Parámetros de la Simulación")

# Sliders para los parámetros físicos. El argumento 'key' los vincula
# directamente con st.session_state, manejando la actualización automáticamente.
st.sidebar.slider(
    "Longitud de Onda (λ, nm)",
    min_value=380.0, max_value=750.0,
    step=0.1, format="%.1f",
    key="lambda_nm"
)
st.sidebar.slider(
    "Distancia al Obstáculo (D, m)",
    min_value=1, max_value=10.0,
    step=0.01, format="%.2f",
    key="D_m"
)
st.sidebar.slider(
    "Ancho de Visualización (mm)",
    min_value=1.0, max_value=40.0,
    step=0.1, format="%.1f",
    key="screen_mm"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Configuración de Bordes")

# Checkbox para cambiar entre el modo de borde único y el de rendija
enable_edge_2 = st.sidebar.checkbox("Habilitar segundo borde (rendija)")

# Slider para la posición del primer borde. Se deshabilita si el modo rendija
# está activo, ya que en ese caso las posiciones se calculan automáticamente.
st.sidebar.slider(
    "Posición 1er Borde (mm)",
    min_value=-st.session_state.screen_mm/2, max_value=st.session_state.screen_mm/2,
    step=0.1, format="%.2f",
    key="edge_1_pos_mm",
    disabled=enable_edge_2
)

edge_2_pos_val = 0.0 # Valor por defecto para la posición del segundo borde

# Este bloque solo se ejecuta si el usuario activa el modo rendija
if enable_edge_2:
    st.sidebar.slider(
        "Separación entre bordes (mm)",
        min_value=0.01, max_value=2.0,
        step=0.01, format="%.2f",
        key="slit_separation_mm"
    )

    # Calcula la posición de los bordes para centrar la rendija en x=0
    edge_1_pos_val = - (st.session_state.slit_separation_mm / 2) * 1e-3
    edge_2_pos_val = (st.session_state.slit_separation_mm / 2) * 1e-3
    st.sidebar.info(f"Bordes en {edge_1_pos_val*1000:.3f} mm y {edge_2_pos_val*1000:.3f} mm")

    # Calcula y muestra el Número de Fresnel, relevante para la rendija
    slit_width_m = st.session_state.slit_separation_mm * 1e-3
    lambda_val = st.session_state.lambda_nm * 1e-9
    D_val = st.session_state.D_m
    fresnel_number = (slit_width_m**2) / (lambda_val * D_val)
    st.sidebar.metric(label="Número de Fresnel (NF)", value=f"{fresnel_number:.4f}")

# --- SIMULACIÓN Y VISUALIZACIÓN ---
st.markdown("---")
# Actualiza el título principal según el modo seleccionado
if enable_edge_2:
    st.subheader(f"Patrón de Rendija de {st.session_state.slit_separation_mm:.2f} mm")
else:
    st.subheader("Patrón de Borde Único")

# Convierte las unidades de los parámetros a SI (metros) para los cálculos
lambda_val = st.session_state.lambda_nm * 1e-9
D_val = st.session_state.D_m
screen_width_m = st.session_state.screen_mm * 1e-3

# Asegura que la posición del borde esté en metros para el modo de borde único
if not enable_edge_2:
    edge_1_pos_val = st.session_state.edge_1_pos_mm * 1e-3

# Llama a la función de cálculo principal
x_coords, intensity = calculate_diffraction_pattern(
    lambda_val, D_val, screen_width_m, edge_1_pos_val,
    enable_edge_2=enable_edge_2, edge_2_pos=edge_2_pos_val
)
# Genera la imagen 2D del patrón
heatmap = generate_heatmap(x_coords, intensity)

# --- Gráficas ---
# Crea una figura con dos subplots apilados verticalmente
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Subplot superior: Perfil de intensidad 1D
ax1.plot(x_coords * 1000, intensity, color='dodgerblue', lw=2)
# Dibuja líneas verticales para marcar la posición de los bordes
ax1.axvline(x=edge_1_pos_val * 1000, color='k', linestyle='--', lw=1.5, label=f'Borde 1 ({edge_1_pos_val*1000:.2f} mm)')

if enable_edge_2:
    ax1.axvline(x=edge_2_pos_val * 1000, color='k', linestyle=':', lw=1.5, label=f'Borde 2 ({edge_2_pos_val*1000:.2f} mm)')

    # Recalcula el NF aquí para asegurar que el título esté siempre actualizado
    slit_width_m_plot = st.session_state.slit_separation_mm * 1e-3
    lambda_val_plot = st.session_state.lambda_nm * 1e-9
    D_val_plot = st.session_state.D_m
    fresnel_number_plot = (slit_width_m_plot**2) / (lambda_val_plot * D_val_plot)

    # Título dinámico para el modo rendija (versión de texto plano para evitar errores)
    ax1.set_title(f"Patrón de Rendija | Número de Fresnel NF ≈ {fresnel_number_plot:.4f}", fontsize=14)

else:
    # Título para el modo de borde único
    ax1.set_title("Patrón de Borde Único", fontsize=14)

ax1.set_ylabel('Intensidad Relativa'); ax1.grid(True, linestyle=':'); ax1.legend()
ax1.set_ylim(bottom=0)
ax1.set_xlim(x_coords.min()*1000, x_coords.max()*1000)

# Subplot inferior: Visualización 2D del patrón
ax2.imshow(heatmap, cmap='gray', aspect='auto', extent=[x_coords.min()*1000, x_coords.max()*1000, -1, 1])
ax2.set_xlabel("Posición en la pantalla x' (mm)"); ax2.get_yaxis().set_visible(False)

# Ajusta el layout para que no haya solapamientos y muestra la figura en Streamlit
fig.tight_layout()
st.pyplot(fig)