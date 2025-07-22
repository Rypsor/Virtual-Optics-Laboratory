# pages/3_Simulador_Rendija_Fresnel.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

# --- La función de cálculo individual no cambia ---
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

# --- Función de pre-cálculo modificada ---
@st.cache_data
def precompute_all_data(start_um, end_um, step_um, lambda_nm, D_m, screen_mm):
    """
    Calcula los datos para cada paso del slider y los devuelve en un diccionario.
    Usa el valor de 'b' en micrómetros como clave del diccionario.
    """
    results = {}
    lambda_val = lambda_nm * 1e-9
    screen_width = screen_mm * 1e-3
    
    # Usamos np.arange para crear la secuencia con un paso fijo
    for b_um in np.arange(start_um, end_um + step_um, step_um):
        b_val = b_um * 1e-6
        results[b_um] = calculate_fresnel(b_val, lambda_val, D_m, screen_width)
        
    return results

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Difracción de Fresnel", page_icon="🌊")
st.title("🌊 Simulador de Fresnel Interactivo")
st.markdown("Configura los parámetros en la barra lateral. El slider principal te permite explorar directamente el ancho de la rendija.")

# --- Controles en la Barra Lateral ---
st.sidebar.header("Parámetros de la Simulación")

lambda_nm = st.sidebar.slider("Longitud de Onda (λ, nm)", 380, 750, 592, 10)
D_m = st.sidebar.slider("Distancia a Pantalla (D, m)", 0.1, 5.0, 1.0, 0.1)
screen_mm = st.sidebar.slider("Ancho de Visualización (mm)", 1, 20, 8, 1)

st.sidebar.markdown("---")
st.sidebar.header("Rango y Paso del Slider")

b_start_um = st.sidebar.number_input("Ancho INICIAL (μm)", 100, 5000, 2500, 50)
b_end_um = st.sidebar.number_input("Ancho FINAL (μm)", 100, 5000, 400, 50)
step_um = st.sidebar.number_input("Paso del slider (μm)", 10, 500, 50, 10)

# --- Pre-cálculo con los parámetros de la barra lateral ---
with st.spinner('Calculando el set de patrones de difracción...'):
    all_data = precompute_all_data(b_start_um, b_end_um, step_um, lambda_nm, D_m, screen_mm)

# --- Slider Principal Directo ---
st.markdown("---")
st.subheader("Explora el Ancho de la Rendija")
b_um_selected = st.slider(
    "Ancho de la rendija 'b' (μm)",
    min_value=b_start_um, 
    max_value=b_end_um, 
    value=b_start_um,
    step=step_um
)

# --- Búsqueda y Visualización ---
# La clave del diccionario ahora es el valor directo del slider
x_prime, intensity_fresnel, intensity_fraunhofer, NF = all_data[b_um_selected]

# --- Dibujado del Gráfico ---
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(x_prime * 1000, intensity_fresnel, label=f'Fresnel', color='dodgerblue', lw=3)
ax.plot(x_prime * 1000, intensity_fraunhofer, 'k--', label='Fraunhofer (Teórico)', lw=2.5, alpha=0.8)

ax.set_title(f'Ancho de Rendija (b) = {b_um_selected:.0f} μm  |  Número de Fresnel ($N_F$) ≈ {NF:.2f}', fontsize=16)
ax.set_xlabel("Posición en la pantalla x' (mm)", fontsize=14)
ax.set_ylabel('Intensidad Normalizada', fontsize=14)
ax.grid(True, linestyle=':')
ax.legend(fontsize=12)
ax.set_ylim(0, 1.2)
ax.set_xlim(x_prime.min() * 1000, x_prime.max() * 1000)

st.pyplot(fig)