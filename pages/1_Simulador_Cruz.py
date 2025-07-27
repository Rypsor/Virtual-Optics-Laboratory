# pages/1_Simulador_Cruz.py

"""
Simulador de Difracción de Fraunhofer para Abertura en Cruz

Este script de Streamlit crea una aplicación web interactiva para simular y
visualizar el patrón de difracción de Fraunhofer generado por una abertura en
forma de cruz.

El usuario puede modificar interactivamente los siguientes parámetros:
- La geometría de la cruz (longitud y grosor de sus brazos).
- La longitud de onda de la luz incidente.
- La distancia entre la abertura y la pantalla de observación.
- Parámetros de visualización como el tamaño de la vista y la resolución.

El simulador realiza lo siguiente:
1.  Dibuja la forma de la abertura según los parámetros definidos.
2.  Calcula el patrón de difracción 2D utilizando la fórmula analítica
    derivada de la transformada de Fourier de la función de la abertura.
3.  Visualiza el patrón de difracción, aplicando un color que corresponde a la
    longitud de onda seleccionada y una escala logarítmica para mejorar el
    contraste.
4.  Incluye una sección educativa que verifica si se cumple la condición de
    campo lejano (Fraunhofer) para los parámetros seleccionados.
"""

# --- Importación de Librerías ---
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math

# --- Funciones de Utilidad y Visualización ---

def wavelength_to_rgb(wavelength_nm):
    """
    Convierte una longitud de onda de la luz (en nm) a un color RGB visible aproximado.

    Esta función implementa el algoritmo de Dan Bruton para mapear una longitud de
    onda del espectro visible a un color (R, G, B). Se usa para colorear el
    patrón de difracción.

    Args:
        wavelength_nm (float): La longitud de onda en nanómetros (típicamente 380-750).

    Returns:
        tuple: Una tupla (R, G, B) con valores entre 0 y 1.
    """
    gamma = 0.8
    wavelength_nm = np.clip(wavelength_nm, 380, 750) # Asegura que esté en el rango visible

    # Mapea la longitud de onda a componentes R, G, B
    if 380 <= wavelength_nm <= 439: R, G, B = -(wavelength_nm - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= wavelength_nm <= 489: R, G, B = 0.0, (wavelength_nm - 440) / (490 - 440), 1.0
    elif 490 <= wavelength_nm <= 509: R, G, B = 0.0, 1.0, -(wavelength_nm - 510) / (510 - 490)
    elif 510 <= wavelength_nm <= 579: R, G, B = (wavelength_nm - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= wavelength_nm <= 644: R, G, B = 1.0, -(wavelength_nm - 645) / (645 - 580), 0.0
    elif 645 <= wavelength_nm <= 750: R, G, B = 1.0, 0.0, 0.0
    else: R, G, B = 0.0, 0.0, 0.0

    # Ajusta la intensidad para los extremos del espectro
    factor = 0.0
    if 380 <= wavelength_nm <= 419: factor = 0.3 + 0.7 * (wavelength_nm - 380) / (420 - 380)
    elif 420 <= wavelength_nm <= 700: factor = 1.0
    elif 701 <= wavelength_nm <= 750: factor = 0.3 + 0.7 * (750 - wavelength_nm) / (750 - 700)

    # Aplica la corrección gamma y devuelve el color
    R = (R * factor) ** gamma; G = (G * factor) ** gamma; B = (B * factor) ** gamma
    return (R, G, B)

def plot_aperture(ax, L1, L2, h1, h2, t, e):
    """
    Dibuja la forma de la abertura de la cruz en un eje de Matplotlib.

    Args:
        ax (matplotlib.axes.Axes): El eje donde se dibujará la abertura.
        L1, L2, h1, h2, t, e (float): Parámetros geométricos de la cruz en metros.
    """
    # Define los límites del gráfico para que la cruz siempre esté bien enmarcada
    max_width = L1 + L2 + t; max_height = h1 + h2 + e
    plot_limit = max(max_width, max_height) * 0.9

    # Crea una malla de coordenadas para dibujar
    x_tilde = np.linspace(-plot_limit, plot_limit, 400)
    y_tilde = np.linspace(-plot_limit, plot_limit, 400)
    X_tilde, Y_tilde = np.meshgrid(x_tilde, y_tilde)

    # Crea una máscara booleana para definir la forma de la cruz
    aperture_mask = np.zeros_like(X_tilde)
    # Barra horizontal
    horiz_bar = (X_tilde >= -(L1 + t/2)) & (X_tilde <= (L2 + t/2)) & (Y_tilde >= -e/2) & (Y_tilde <= e/2)
    # Barra vertical
    vert_bar = (X_tilde >= -t/2) & (X_tilde <= t/2) & (Y_tilde >= -(h1 + e/2)) & (Y_tilde <= (h2 + e/2))

    # Combina las dos barras para formar la cruz
    aperture_mask[horiz_bar | vert_bar] = 1

    # Dibuja la máscara en el eje proporcionado
    ax.imshow(aperture_mask, cmap='gray', extent=[x*1e6 for x in [-plot_limit, plot_limit, -plot_limit, plot_limit]])
    ax.set_facecolor('black')
    ax.set_title('Forma de la Rendija', fontsize=15)
    ax.set_xlabel('Posición x (μm)', fontsize=12)
    ax.set_ylabel('Posición y (μm)', fontsize=12)

def sync_widget(master_key, widget_key):
    """
    Sincroniza un slider y un campo numérico en Streamlit.

    Asegura que si el usuario cambia el slider, el número se actualiza, y viceversa.

    Args:
        master_key (str): La clave en st.session_state que almacena el valor real.
        widget_key (str): La clave del widget que acaba de ser modificado.
    """
    st.session_state[master_key] = st.session_state[widget_key]

def calculate_intensity(kx, ky, L1, L2, h1, h2, t, e):
    """
    Calcula la intensidad del patrón de difracción de Fraunhofer.

    Esta función implementa la solución analítica de la difracción para una
    abertura en cruz, que es la transformada de Fourier al cuadrado de la
    función de la abertura.

    Args:
        kx, ky (np.array): Mallas de coordenadas en el espacio recíproco (frecuencia espacial).
        L1, L2, h1, h2, t, e (float): Parámetros geométricos de la cruz en metros.

    Returns:
        np.array: Una matriz 2D con la intensidad del patrón de difracción.
    """
    # Alias para la función sinc normalizada
    sinc = lambda x: np.sinc(x / np.pi)
    # Dimensiones totales de la cruz
    H = h1 + h2 + e
    W = L1 + L2 + t

    # La fórmula se descompone en varios términos para mayor claridad
    term1 = (t**2 * H**2 * sinc(kx * t / 2)**2 * sinc(ky * H / 2)**2)
    term2 = (e**2 * W**2 * sinc(kx * W / 2)**2 * sinc(ky * e / 2)**2)
    term3 = (t**2 * e**2 * sinc(kx * t / 2)**2 * sinc(ky * e / 2)**2)


    # términos cruzados
    cos_arg1 = ky * (h2 - h1) / 2 + kx * (L2 - L1) / 2
    term4 = (2 * t * e * H * W * sinc(kx * t / 2) * sinc(ky * H / 2) * sinc(kx * W / 2) * sinc(ky * e / 2) * np.cos(cos_arg1))

    cos_arg2 = ky * (h2 - h1) / 2
    term5 = (-2 * (t**2) * e * H * sinc(kx * t / 2)**2 * sinc(ky * H / 2) * sinc(ky * e / 2) * np.cos(cos_arg2))

    cos_arg3 = kx * (L2 - L1) / 2
    term6 = (-2 * t * e**2 * W * sinc(kx * t / 2) * sinc(kx * W / 2) * sinc(ky * e / 2)**2 * np.cos(cos_arg3))

    intensity_pattern = term1 + term2 + term3 + term4 + term5 + term6
    return intensity_pattern

# --- Configuración de la Página e Interfaz de Usuario con Streamlit ---
st.set_page_config(page_title="Simulador de Cruz", page_icon="➕", layout="wide")
st.title("➕ Simulador de Difracción para Abertura en Cruz")

# --- Controles de la Barra Lateral ---
st.sidebar.header("Parámetros de la Simulación")
# Inicializa el estado de la sesión con valores por defecto la primera vez que se corre
defaults = {"L1_um": 50.0, "L2_um": 50.0, "h1_um": 50.0, "h2_um": 50.0, "t_um": 15.0, "e_um": 15.0, "lambda_nm": 532.0, "D_cm": 100.0}
for key, value in defaults.items():
    if key not in st.session_state: st.session_state[key] = value

# Sección para los parámetros geométricos de la cruz
st.sidebar.subheader("Geometría de la Cruz (μm)")
# Cada parámetro tiene un slider y un campo numérico sincronizados
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Brazo izquierdo (L1)", 0.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.L1_um, key="L1_slider", on_change=sync_widget, args=("L1_um", "L1_slider")); col2.number_input("L1", 0.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.L1_um, key="L1_input", on_change=sync_widget, args=("L1_um", "L1_input"), label_visibility="collapsed")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Brazo derecho (L2)", 0.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.L2_um, key="L2_slider", on_change=sync_widget, args=("L2_um", "L2_slider")); col2.number_input("L2", 0.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.L2_um, key="L2_input", on_change=sync_widget, args=("L2_um", "L2_input"), label_visibility="collapsed")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Brazo superior (h1)", 0.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.h1_um, key="h1_slider", on_change=sync_widget, args=("h1_um", "h1_slider")); col2.number_input("h1", 0.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.h1_um, key="h1_input", on_change=sync_widget, args=("h1_um", "h1_input"), label_visibility="collapsed")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Brazo inferior (h2)", 0.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.h2_um, key="h2_slider", on_change=sync_widget, args=("h2_um", "h2_slider")); col2.number_input("h2", 0.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.h2_um, key="h2_input", on_change=sync_widget, args=("h2_um", "h2_input"), label_visibility="collapsed")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Grosor Vertical (t)", 1.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.t_um, key="t_slider", on_change=sync_widget, args=("t_um", "t_slider")); col2.number_input("t", 1.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.t_um, key="t_input", on_change=sync_widget, args=("t_um", "t_input"), label_visibility="collapsed")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Grosor Horizontal (e)", 1.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.e_um, key="e_slider", on_change=sync_widget, args=("e_um", "e_slider")); col2.number_input("e", 1.0, 1000.0, step=0.01, format="%.2f", value=st.session_state.e_um, key="e_input", on_change=sync_widget, args=("e_um", "e_input"), label_visibility="collapsed")

# Sección para los parámetros de la luz y la pantalla
st.sidebar.subheader("Luz y Pantalla")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Longitud de Onda (λ, nm)", 380.0, 750.0, step=0.1, format="%.1f", value=st.session_state.lambda_nm, key="lambda_slider", on_change=sync_widget, args=("lambda_nm", "lambda_slider")); col2.number_input("λ", 380.0, 750.0, step=0.1, format="%.1f", value=st.session_state.lambda_nm, key="lambda_input", on_change=sync_widget, args=("lambda_nm", "lambda_input"), label_visibility="collapsed")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Distancia a Pantalla (D, cm)", 1.0, 200.0, step=0.1, format="%.1f", value=st.session_state.D_cm, key="D_slider", on_change=sync_widget, args=("D_cm", "D_slider")); col2.number_input("D", 1.0, 200.0, step=0.1, format="%.1f", value=st.session_state.D_cm, key="D_input", on_change=sync_widget, args=("D_cm", "D_input"), label_visibility="collapsed")

# Sección para los parámetros de visualización
st.sidebar.subheader("Parámetros de Vista")
screen_size_cm = st.sidebar.slider("Ancho de Vista (cm)", 0.1, 5.0, 1.0, 0.1)
N_points = st.sidebar.select_slider("Resolución", options=[256, 512, 1024], value=512)
vmax_contrast = st.sidebar.slider("Ajuste de Contraste (brillo)", 0.01, 1.0, 0.5, 0.01)

# --- Conversión de Unidades ---
# Convierte los valores de los widgets (en μm, nm, cm) a unidades del Sistema
# Internacional (metros) para usarlos en los cálculos físicos.
params_um = ["L1_um", "L2_um", "h1_um", "h2_um", "t_um", "e_um"]
L1, L2, h1, h2, t, e = (st.session_state[k] * 1e-6 for k in params_um)
lambda_ = st.session_state.lambda_nm * 1e-9
D = st.session_state.D_cm * 1e-2
screen_size = screen_size_cm * 1e-2

# Expander para verificar si se cumple la condición de campo lejano
with st.expander("Verificar Condición de Campo Lejano", expanded=False):
    W_total = L1 + t + L2; H_total = h1 + e + h2
    L_max = max(W_total, H_total) # Dimensión más grande de la abertura
    if lambda_ > 0:
        dist_minima = (L_max**2) / lambda_
        dist_recomendada = 10 * dist_minima
    else:
        dist_minima = float('inf')
        dist_recomendada = float('inf')

    # Muestra la fórmula y los valores actuales usando formato LaTeX
    st.write("Expresión general:"); st.latex(r''' D \gg \frac{L_{max}^2}{\lambda} ''')
    st.write("Valores de la simulación:")
    # ... (código de formato LaTeX para mostrar los números de forma legible)
    D_cm_val = st.session_state.D_cm; L_max_cm = L_max * 100
    if L_max_cm > 0:
        exponent_L = math.floor(math.log10(abs(L_max_cm)))
        mantissa_L = L_max_cm / (10**exponent_L)
        l_max_latex = f"{mantissa_L:.2f} \\times 10^{{{exponent_L}}}"
    else: l_max_latex = "0.00"
    lambda_cm = st.session_state.lambda_nm * 1e-7
    exponent_lambda = math.floor(math.log10(abs(lambda_cm)))
    mantissa_lambda = lambda_cm / (10**exponent_lambda)
    lambda_latex = f"{mantissa_lambda:.2f} \\times 10^{{{exponent_lambda}}}"
    dist_minima_cm = dist_minima * 100
    latex_string = (f"{D_cm_val:.2f} \\, \\text{{cm}} \\gg " f"\\frac{{({l_max_latex} \\, \\text{{cm}})^2}}{{{lambda_latex} \\, \\text{{cm}}}}" f" \\rightarrow {D_cm_val:.2f} \\, \\text{{cm}} \\gg {dist_minima_cm:.2f} \\, \\text{{cm}}")
    st.latex(latex_string)

    # Informa al usuario si su configuración cumple con la condición
    st.write(f"**Tu distancia D actual:** `{D_cm_val:.2f} cm`")
    st.write(f"**Distancia mínima recomendada:** `>{dist_recomendada*100:.2f} cm`")

    if D >= dist_recomendada: st.success("✅ Condición de campo lejano recomendada CUMPLIDA.")
    elif D >= dist_minima: st.warning("⚠️ Condición de campo lejano CUMPLIDA, pero se recomienda una distancia mayor.")
    else: st.error("❌ ADVERTENCIA: NO se cumple la condición de campo lejano.")

st.markdown("---")

# --- Bucle Principal de Cálculo y Visualización ---

# Crea las mallas de coordenadas para la pantalla de visualización
x_prime = np.linspace(-screen_size / 2, screen_size / 2, N_points)
y_prime = np.linspace(-screen_size / 2, screen_size / 2, N_points)
X_prime, Y_prime = np.meshgrid(x_prime, y_prime)

# Convierte las coordenadas espaciales (x', y') a coordenadas en el espacio
# recíproco (kx, ky), que son las variables de la transformada de Fourier.
kx = (2 * np.pi / (lambda_ * D)) * X_prime
ky = (2 * np.pi / (lambda_ * D)) * Y_prime

# Llama a la función principal para calcular el patrón de intensidad
intensity_pattern = calculate_intensity(kx, ky, L1, L2, h1, h2, t, e)

# Crea la figura de Matplotlib con dos subplots uno al lado del otro
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Dibuja la forma de la abertura en el primer subplot
plot_aperture(ax1, L1, L2, h1, h2, t, e)

# --- Procesamiento y Visualización del Patrón de Difracción ---
# Se aplica una escala logarítmica para resaltar los detalles de baja intensidad
log_intensity = np.log1p(intensity_pattern)
min_val, max_val = np.min(log_intensity), np.max(log_intensity)

# Ajusta el contraste máximo basado en el slider del usuario
if min_val >= max_val: vmax_val = None
else: vmax_val = max_val * vmax_contrast

# Crea un mapa de colores personalizado que va de negro al color de la longitud de onda
visible_color = wavelength_to_rgb(st.session_state.lambda_nm)
colors = ['black', visible_color]
my_cmap = LinearSegmentedColormap.from_list('custom_wavelength', colors, N=256)

# Muestra el patrón de difracción en el segundo subplot
im = ax2.imshow(log_intensity, cmap=my_cmap, vmax=vmax_val, extent=[-screen_size/2*100, screen_size/2*100, -screen_size/2*100, screen_size/2*100])
ax2.set_facecolor('k')
ax2.set_title('Patrón de Difracción', fontsize=15)
ax2.set_xlabel("Posición en pantalla x' (cm)", fontsize=12)
ax2.set_ylabel("Posición en pantalla y' (cm)", fontsize=12)
fig.colorbar(im, ax=ax2, label='Intensidad Relativa (escala log)', fraction=0.046, pad=0.04)

# Muestra la figura completa en la aplicación Streamlit
st.pyplot(fig)