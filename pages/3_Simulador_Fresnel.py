# pages/3_Simulador_Fresnel.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

# --- Función de cálculo individual (sin cambios) ---
def calculate_fresnel(b_val, lambda_val, D_val, screen_width):
    """Calcula un único patrón de difracción."""
    NF = (b_val**2) / (lambda_val * D_val)
    x_prime = np.linspace(-screen_width / 2, screen_width / 2, 2000)
    
    u_factor = np.sqrt(2 / (lambda_val * D_val))
    u1 = u_factor * (x_prime - b_val / 2)
    u2 = u_factor * (x_prime + b_val / 2)
    
    Su1, Cu1 = fresnel(u1)
    Su2, Cu2 = fresnel(u2)
    
    intensity = 0.5 * ((Cu2 - Cu1)**2 + (Su2 - Su1)**2)
    if np.max(intensity) > 0:
        intensity_norm = intensity / np.max(intensity)
    else:
        intensity_norm = intensity
        
    sinc_arg = (np.pi * b_val * x_prime) / (lambda_val * D_val)
    intensity_fraunhofer = (np.sinc(sinc_arg / np.pi))**2
    
    return x_prime, intensity_norm, intensity_fraunhofer, NF

# --- Función que pre-calcula todos los cuadros y los guarda en caché ---
@st.cache_data
def precompute_all_data(start_um, end_um, num_steps, lambda_nm, D_m, screen_mm):
    """
    Calcula los datos para cada paso del slider y los devuelve en un diccionario.
    Se re-ejecutará solo si uno de estos parámetros de la barra lateral cambia.
    """
    results = {}
    
    # Convertir unidades para los cálculos
    lambda_val = lambda_nm * 1e-9
    screen_width = screen_mm * 1e-3
    
    b_sequence = np.linspace(start_um * 1e-6, end_um * 1e-6, num_steps)
    
    for i, b_val in enumerate(b_sequence):
        results[i] = calculate_fresnel(b_val, lambda_val, D_m, screen_width)
        
    return results

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Difracción de Fresnel", page_icon="🌊")
st.title("🌊 Simulador de Fresnel Interactivo")
st.markdown("Configura todos los parámetros en la barra lateral. El slider principal te permitirá explorar el rango de anchos de rendija de forma instantánea.")

# --- Controles en la Barra Lateral ---
st.sidebar.header("Parámetros de la Simulación")

lambda_nm = st.sidebar.slider("Longitud de Onda (λ, nm)", 380, 750, 592, 10)
D_m = st.sidebar.slider("Distancia a Pantalla (D, m)", 0.1, 5.0, 1.0, 0.1)
screen_mm = st.sidebar.slider("Ancho de Visualización (mm)", 1, 20, 8, 1)

st.sidebar.markdown("---")
st.sidebar.header("Rango del Slider Principal")

b_start_um = st.sidebar.number_input("Ancho INICIAL de la rendija (μm)", 100, 5000, 2500, 50)
b_end_um = st.sidebar.number_input("Ancho FINAL de la rendija (μm)", 100, 5000, 400, 50)
num_steps = st.sidebar.slider("Número de Pasos", 10, 300, 100)

# --- Pre-cálculo con los parámetros de la barra lateral ---
with st.spinner('Calculando el set de patrones de difracción...'):
    all_data = precompute_all_data(b_start_um, b_end_um, num_steps, lambda_nm, D_m, screen_mm)

# --- Slider Principal ---
st.markdown("---")
st.subheader("Explora el Ancho de la Rendija")
step_index = st.slider(
    "Mueve para cambiar el ancho 'b' de la rendija",
    min_value=0, 
    max_value=num_steps - 1, 
    value=0
)

# --- Búsqueda y Visualización ---
x_prime, intensity_fresnel, intensity_fraunhofer, NF = all_data[step_index]

# Mostrar el ancho actual para el índice seleccionado
b_values_um = np.linspace(b_start_um, b_end_um, num_steps)
current_b_um = b_values_um[step_index]

# --- Dibujado del Gráfico ---
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(x_prime * 1000, intensity_fresnel, label=f'Fresnel', color='dodgerblue', lw=3)
ax.plot(x_prime * 1000, intensity_fraunhofer, 'k--', label='Fraunhofer (Teórico)', lw=2.5, alpha=0.8)

ax.set_title(f'Ancho de Rendija (b) = {current_b_um:.0f} μm  |  Número de Fresnel ($N_F$) ≈ {NF:.2f}', fontsize=16)
ax.set_xlabel("Posición en la pantalla x' (mm)", fontsize=14)
ax.set_ylabel('Intensidad Normalizada', fontsize=14)
ax.grid(True, linestyle=':')
ax.legend(fontsize=12)
ax.set_ylim(0, 1.2)
ax.set_xlim(x_prime.min() * 1000, x_prime.max() * 1000)

st.pyplot(fig)