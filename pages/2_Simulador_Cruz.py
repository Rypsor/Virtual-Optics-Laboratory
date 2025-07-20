# pages/2_Simulador_Cruz.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Funciones de Cálculo y Dibujo (sin cambios) ---

def plot_aperture(ax, L1, L2, h1, h2, t, e):
    """
    Dibuja la forma de la abertura de la cruz en un eje de matplotlib.
    """
    max_width = L1 + L2 + t
    max_height = h1 + h2 + e
    plot_limit = max(max_width, max_height) * 0.7
    
    x_tilde = np.linspace(-plot_limit, plot_limit, 400)
    y_tilde = np.linspace(-plot_limit, plot_limit, 400)
    X_tilde, Y_tilde = np.meshgrid(x_tilde, y_tilde)
    
    aperture_mask = np.zeros_like(X_tilde)
    
    horiz_bar = (X_tilde >= -(L1 + t/2)) & (X_tilde <= (L2 + t/2)) & \
                (Y_tilde >= -e/2) & (Y_tilde <= e/2)
    
    vert_bar = (X_tilde >= -t/2) & (X_tilde <= t/2) & \
               (Y_tilde >= -(h2 + e/2)) & (Y_tilde <= (h1 + e/2))
               
    aperture_mask[horiz_bar | vert_bar] = 1
    
    ax.imshow(aperture_mask, cmap='gray',
              extent=[x*1e6 for x in [-plot_limit, plot_limit, -plot_limit, plot_limit]])
    ax.set_title('Forma de la Rendija', fontsize=15)
    ax.set_xlabel('Posición x (μm)', fontsize=12)
    ax.set_ylabel('Posición y (μm)', fontsize=12)

def calculate_intensity(kx, ky, L1, L2, h1, h2, t, e):
    """
    Calcula la intensidad del patrón de difracción de Fraunhofer para una cruz.
    """
    k_eps = 1e-12
    kx = kx + k_eps
    ky = ky + k_eps
    W = L1 + t + L2
    H = h1 + e + h2
    
    sinc_Ax = np.sinc(kx * W / (2 * np.pi))
    sinc_Ay = np.sinc(ky * e / (2 * np.pi))
    phase_A = np.exp(-1j * kx * (L2 - L1) / 2)
    A = W * e * sinc_Ax * sinc_Ay * phase_A

    sinc_Bx = np.sinc(kx * t / (2 * np.pi))
    sinc_By = np.sinc(ky * H / (2 * np.pi))
    phase_B = np.exp(-1j * ky * (h1 - h2) / 2)
    B = t * H * sinc_Bx * sinc_By * phase_B

    sinc_Cx = np.sinc(kx * t / (2 * np.pi))
    sinc_Cy = np.sinc(ky * e / (2 * np.pi))
    C = t * e * sinc_Cx * sinc_Cy
    
    E_total = A + B - C
    intensity = np.abs(E_total)**2
    return intensity

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Simulador de Cruz", page_icon="➕", layout="wide")
st.title("➕ Simulador de Difracción para Abertura en Cruz")
st.sidebar.header("Parámetros de la Simulación")

# --- Interfaz de Streamlit en la Barra Lateral ---
st.sidebar.subheader("Geometría de la Cruz (μm)")
L1_um = st.sidebar.slider("Brazo izquierdo (L1)", 0, 100, 50, 1)
L2_um = st.sidebar.slider("Brazo derecho (L2)", 0, 100, 50, 1)
h1_um = st.sidebar.slider("Brazo superior (h1)", 0, 100, 0, 1)
h2_um = st.sidebar.slider("Brazo inferior (h2)", 0, 100, 0, 1)
t_um = st.sidebar.slider("Grosor Vertical (t)", 1, 100, 50, 1)
e_um = st.sidebar.slider("Grosor Horizontal (e)", 1, 100, 15, 1)

st.sidebar.subheader("Luz y Pantalla")
lambda_nm = st.sidebar.slider("Longitud de Onda (λ, nm)", 380, 750, 532, 10)
D_cm = st.sidebar.slider("Distancia a Pantalla (D, cm)", 1.0, 100.0, 11.0, 0.5)

st.sidebar.subheader("Parámetros de Vista")
screen_size_cm = st.sidebar.slider("Ancho de Vista (cm)", 0.1, 5.0, 1.0, 0.1)
N_points = st.sidebar.select_slider("Resolución", options=[256, 512, 1024], value=512)
vmax_contrast = st.sidebar.slider("Ajuste de Contraste (brillo)", 0.1, 1.0, 0.5, 0.05)


# --- Conversión de unidades de la UI a metros (para los cálculos) ---
L1, L2, h1, h2, t, e = (val * 1e-6 for val in [L1_um, L2_um, h1_um, h2_um, t_um, e_um])
lambda_ = lambda_nm * 1e-9
D = D_cm * 1e-2
screen_size = screen_size_cm * 1e-2


# --- Verificación de Campo Lejano (mostrada en la app) ---
with st.expander("Verificar Condición de Campo Lejano", expanded=False):
    W_total = L1 + t + L2
    H_total = h1 + e + h2
    L_max = max(W_total, H_total)
    dist_minima = (L_max**2) / lambda_
    dist_recomendada = 10 * dist_minima

    st.write(f"**Dimensión máxima de la rendija (L_max):** `{L_max*1e6:.2f} μm`")
    st.write(f"**Tu distancia D actual:** `{D*100:.2f} cm`")
    st.write(f"**Distancia mínima recomendada (>10x):** `{dist_recomendada*100:.2f} cm`")
    
    if D >= dist_recomendada:
        st.success("✅ Condición de campo lejano recomendada CUMPLIDA.")
    elif D >= dist_minima:
        st.warning("⚠️ Condición de campo lejano CUMPLIDA, pero se recomienda una distancia mayor.")
    else:
        st.error("❌ ADVERTENCIA: NO se cumple la condición de campo lejano.")


# --- Cálculo y Visualización ---
st.markdown("---")

x_prime = np.linspace(-screen_size / 2, screen_size / 2, N_points)
y_prime = np.linspace(-screen_size / 2, screen_size / 2, N_points)
X_prime, Y_prime = np.meshgrid(x_prime, y_prime)

kx = (2 * np.pi / (lambda_ * D)) * X_prime
ky = (2 * np.pi / (lambda_ * D)) * Y_prime

intensity_pattern = calculate_intensity(kx, ky, L1, L2, h1, h2, t, e)

# --- Graficar en Streamlit ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


plot_aperture(ax1, L1, L2, h1, h2, t, e)

log_intensity = np.log1p(intensity_pattern)

# --- FIX: Añadir chequeo para evitar el error de valor ---
min_val = np.min(log_intensity)
max_val = np.max(log_intensity)

# Si todos los valores son iguales, no apliques el contraste para evitar el error.
if min_val >= max_val:
    vmax_val = None # Dejar que Matplotlib decida
else:
    # Si hay un rango de valores, aplica el contraste como antes.
    vmax_val = max_val * vmax_contrast

im = ax2.imshow(log_intensity, 
                cmap='Greens_r',
                vmax=vmax_val, # Usamos el valor calculado y seguro
                extent=[-screen_size/2*100, screen_size/2*100, 
                        -screen_size/2*100, screen_size/2*100])
ax2.set_facecolor('k')
ax2.set_facecolor('k')
ax2.set_title('Patrón de Difracción', fontsize=15)
ax2.set_xlabel("Posición en pantalla x' (cm)", fontsize=12)
ax2.set_ylabel("Posición en pantalla y' (cm)", fontsize=12)
st.pyplot(fig)