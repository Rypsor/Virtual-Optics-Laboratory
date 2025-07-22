# pages/4_Simulador_Borde_Fresnel.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

# --- Función de cálculo con caché para el borde recto ---
@st.cache_data
def calculate_edge_diffraction(lambda_nm, D_m, screen_mm):
    """
    Calcula el patrón de difracción de Fresnel y su referente de Fraunhofer.
    """
    # Convertir unidades a metros
    lambda_val = lambda_nm * 1e-9
    screen_width = screen_mm * 1e-3
    x_prime = np.linspace(-screen_width / 2, screen_width / 2, 2000)

    # --- Cálculo de Fresnel ---
    v = x_prime * np.sqrt(2 / (lambda_val * D_m))
    S_v, C_v = fresnel(v)
    intensity_fresnel = 0.5 * ((C_v + 0.5)**2 + (S_v + 0.5)**2)

    # --- NUEVO: Cálculo de Fraunhofer (sombra perfecta / función escalón) ---
    intensity_fraunhofer = np.ones_like(x_prime)
    intensity_fraunhofer[x_prime < 0] = 0
    
    return x_prime, intensity_fresnel, intensity_fraunhofer

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Difracción por un Borde", page_icon="🔪")
st.title("🔪 Difracción de Fresnel por un Borde Recto")
st.markdown("Esta simulación compara el patrón de campo cercano (Fresnel) con la sombra perfecta del campo lejano (Fraunhofer).")

# --- Controles en la Barra Lateral ---
st.sidebar.header("Parámetros de la Simulación")
lambda_nm = st.sidebar.slider("Longitud de Onda (λ, nm)", 380, 750, 550, 10)
D_m = st.sidebar.slider("Distancia al Borde (D, m)", 0.1, 5.0, 1.0, 0.1)
screen_mm = st.sidebar.slider("Ancho de Visualización (mm)", 1, 20, 10, 1)

# --- Cálculo y Visualización ---
with st.spinner('Calculando patrón de difracción...'):
    x_prime, intensity_fresnel, intensity_fraunhofer = calculate_edge_diffraction(lambda_nm, D_m, screen_mm)

# --- Dibujado del Gráfico ---
fig, ax = plt.subplots(figsize=(12, 7))

# Graficar el resultado de Fresnel
ax.plot(x_prime * 1000, intensity_fresnel, color='crimson', lw=3, label='Fresnel (Campo Cercano)')

# --- NUEVO: Graficar el referente de Fraunhofer ---
ax.plot(x_prime * 1000, intensity_fraunhofer, 'k:', lw=3, label='Fraunhofer (Sombra Perfecta)')


# Añadir una línea para marcar la sombra geométrica
ax.axvline(x=0, color='gray', linestyle='--', lw=2, label='Posición del Borde (x=0)')

# Formato del gráfico
ax.set_title("Comparación de Difracción de Fresnel y Fraunhofer", fontsize=16)
ax.set_xlabel("Posición en la pantalla x' (mm)", fontsize=14)
ax.set_ylabel('Intensidad Relativa ($I/I_0$)', fontsize=14)
ax.grid(True, linestyle=':')
ax.legend()
ax.set_ylim(0, 1.5)
ax.set_xlim(x_prime.min() * 1000, x_prime.max() * 1000)

st.pyplot(fig)