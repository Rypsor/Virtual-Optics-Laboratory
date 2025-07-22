# pages/3_Simulador_Rendija_Fresnel.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

def calculate_fresnel(b_val, lambda_val, D_val, screen_width, resolution=1000):
    """Calcula un único patrón de difracción con una resolución específica."""
    NF = (b_val**2) / (lambda_val * D_val)
    x_prime = np.linspace(-screen_width / 2, screen_width / 2, resolution)
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

@st.cache_data
def precompute_distance_data(b_um, start_m, end_m, step_m, lambda_nm, screen_mm):
    """Calcula los datos para cada paso del slider y usa el valor de D como clave."""
    results = {}
    lambda_val = lambda_nm * 1e-9
    screen_width = screen_mm * 1e-3
    b_val = b_um * 1e-6
    for D_val in np.arange(start_m, end_m + step_m, step_m):
        D_key = round(D_val, 4)
        results[D_key] = calculate_fresnel(b_val, lambda_val, D_key, screen_width, resolution=1000)
    return results

@st.cache_data
def precompute_heatmap_data(b_um, lambda_nm, D_max_m, screen_mm):
    """Calcula los datos para el mapa de calor 2D."""
    lambda_val = lambda_nm * 1e-9
    screen_width = screen_mm * 1e-3
    b_val = b_um * 1e-6
    heatmap_x_res = 150 
    heatmap_y_res = 400 
    D_array = np.linspace(0.01, D_max_m, heatmap_x_res) 
    x_prime_array = np.linspace(-screen_width / 2, screen_width / 2, heatmap_y_res)
    intensity_map = np.zeros((len(x_prime_array), len(D_array)))
    for i, D_val in enumerate(D_array):
        _, intensity_1D, _, _ = calculate_fresnel(b_val, lambda_val, D_val, screen_width, resolution=heatmap_y_res)
        intensity_map[:, i] = intensity_1D
    return D_array, x_prime_array, intensity_map

@st.cache_data
def precompute_difference_data(b_um, start_m, end_m, lambda_nm, screen_mm):
    """Calcula 10 curvas de diferencia (Fresnel - Fraunhofer) y sus NF."""
    diff_results = []
    D_values = np.linspace(start_m, end_m, 10)
    for D_val in D_values:
        lambda_val = lambda_nm * 1e-9
        screen_width = screen_mm * 1e-3
        b_val = b_um * 1e-6
        x_prime, fres, frau, nf_val = calculate_fresnel(b_val, lambda_val, D_val, screen_width)
        difference = fres - frau
        diff_results.append((D_val, x_prime, difference, nf_val))
    return diff_results

st.set_page_config(page_title="Evolución de la Difracción", page_icon="🗺️", layout="wide")
st.title("🗺️ Evolución de Fresnel a Fraunhofer")

st.sidebar.header("Parámetros Fijos")
lambda_nm = st.sidebar.slider("Longitud de Onda (λ, nm)", 380, 750, 592, 10)
b_um = st.sidebar.slider("Ancho de la Rendija 'b' (μm)", 100, 3000, 800, 50)
screen_mm = st.sidebar.slider("Ancho de Visualización (mm)", 1, 20, 10, 1)
st.sidebar.markdown("---")
st.sidebar.header("Rango del Slider de Distancia")
D_start_m = st.sidebar.number_input("Distancia INICIAL (m)", 0.1, 10.0, 0.1, 0.1)
D_end_m = st.sidebar.number_input("Distancia FINAL (m)", 0.1, 10.0, 5.0, 0.1)
step_m = st.sidebar.number_input("Paso del slider (m)", 0.05, 1.0, 0.1, 0.05)

with st.spinner('Calculando todos los patrones de difracción...'):
    slider_data = precompute_distance_data(b_um, D_start_m, D_end_m, step_m, lambda_nm, screen_mm)
    D_map, x_map, intensity_map = precompute_heatmap_data(b_um, lambda_nm, D_end_m, screen_mm)
    difference_data = precompute_difference_data(b_um, D_start_m, D_end_m, lambda_nm, screen_mm)

b_val_fixed = b_um * 1e-6
lambda_val_fixed = lambda_nm * 1e-9
D_at_NF1 = (b_val_fixed**2) / lambda_val_fixed
st.sidebar.metric(label="Distancia para NF=1", value=f"{D_at_NF1:.2f} m")

st.markdown("---")
colT1, colT2 = st.columns(2)
with colT1:
    st.subheader("Gráfica Interactiva por Distancia")
with colT2:
    st.subheader("Mapa de Evolución del Patrón")
D_m_selected = st.slider("Mueve para cambiar la Distancia 'D' (m)", D_start_m, D_end_m, D_start_m, step_m)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.5]})
x_prime, intensity_fresnel, intensity_fraunhofer, NF = slider_data[round(D_m_selected, 4)]
ax1.plot(x_prime * 1000, intensity_fresnel, label='Fresnel', color='dodgerblue', lw=2)
ax1.plot(x_prime * 1000, intensity_fraunhofer, 'k--', label='Fraunhofer', lw=2)
ax1.set_title(f'Patrón a D = {D_m_selected:.2f} m  |  $N_F$ ≈ {NF:.2f}', fontsize=14)
ax1.set_xlabel("Posición x' (mm)", fontsize=12)
ax1.set_ylabel('Intensidad Normalizada', fontsize=12)
ax1.grid(True, linestyle=':')
ax1.legend()
ax1.set_ylim(0, 1.4)

log_intensity_map = np.log1p(intensity_map)
im = ax2.imshow(log_intensity_map, aspect='auto', origin='lower', extent=[D_map.min(), D_map.max(), x_map.min() * 1000, x_map.max() * 1000], cmap='hot')
fig1.colorbar(im, ax=ax2, label='Intensidad (escala log)', fraction=0.046, pad=0.04)
ax2.set_title(f'Transición para b = {b_um} μm', fontsize=14)
ax2.set_xlabel("Distancia a la Pantalla 'D' (m)", fontsize=12)
ax2.set_ylabel("Posición en la pantalla x' (mm)", fontsize=12)
ax2.axvline(x=D_m_selected, color='cyan', linestyle='--', lw=2, label=f'Vista Actual ({D_m_selected:.2f}m)')
if D_map.min() <= D_at_NF1 <= D_map.max():
    ax2.axvline(x=D_at_NF1, color='magenta', linestyle=':', lw=3, label=f'Frontera NF=1 ({D_at_NF1:.2f}m)')
ax2.legend()
fig1.tight_layout(pad=2.0)
st.pyplot(fig1)

st.markdown("---")
st.subheader("Diferencia entre Patrones de Fresnel y Fraunhofer")
fig2, ax3 = plt.subplots(figsize=(12, 6))
cmap = plt.get_cmap('plasma')
ax3.axhline(0, color='red', linestyle='--', lw=1.5, label='Diferencia Cero')

for i, data in enumerate(difference_data):
    D_val, x_prime_diff, difference, nf_val = data
    color = cmap(i / 9)
    label = f'D={D_val:.2f}m (NF≈{nf_val:.1f})'
    ax3.plot(x_prime_diff * 1000, difference, color=color, label=label)

ax3.set_title('Convergencia a Fraunhofer', fontsize=16)
ax3.set_xlabel("Posición en la pantalla x' (mm)", fontsize=14)
ax3.set_ylabel('Diferencia (Fresnel - Fraunhofer)', fontsize=14)
ax3.grid(True, linestyle=':')
# CAMBIO: Se reemplazó 'Nº' por 'N.' para evitar el error de parseo
ax3.legend(title='Distancia D y N. Fresnel', fontsize=9, ncol=2)
ax3.set_xlim(-screen_mm/2, screen_mm/2)
st.pyplot(fig2)