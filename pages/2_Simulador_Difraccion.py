# pages/2_Simulador_Diffraccion.py
# --- Importación de Librerías ---
import streamlit as st
import numpy as np
from PIL import Image
from scipy.ndimage import zoom, rotate
import io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Configuración de la Página ---
st.set_page_config(page_title="Simulador de Difracción", page_icon="🔬")

# --- Funciones de Utilidad y Visualización ---

def wavelength_to_rgb(wavelength_nm):
    """Convierte una longitud de onda de la luz (en nm) a un color RGB visible.

    Args:
        wavelength_nm (float): La longitud de onda en nanómetros.

    Returns:
        tuple: Un triplete (R, G, B) con valores entre 0 y 1.
    """
    gamma = 0.8; wavelength_nm = np.clip(wavelength_nm, 380, 750)
    # Algoritmo para aproximar el color RGB del espectro visible.
    if 380 <= wavelength_nm <= 439: R, G, B = -(wavelength_nm - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= wavelength_nm <= 489: R, G, B = 0.0, (wavelength_nm - 440) / (490 - 440), 1.0
    elif 490 <= wavelength_nm <= 509: R, G, B = 0.0, 1.0, -(wavelength_nm - 510) / (510 - 490)
    elif 510 <= wavelength_nm <= 579: R, G, B = (wavelength_nm - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= wavelength_nm <= 644: R, G, B = 1.0, -(wavelength_nm - 645) / (645 - 580), 0.0
    elif 645 <= wavelength_nm <= 750: R, G, B = 1.0, 0.0, 0.0
    else: R, G, B = 0.0, 0.0, 0.0
    # Ajusta la intensidad del color para que coincida con la sensibilidad del ojo.
    factor = 0.0
    if 380 <= wavelength_nm <= 419: factor = 0.3 + 0.7 * (wavelength_nm - 380) / (420 - 380)
    elif 420 <= wavelength_nm <= 700: factor = 1.0
    elif 701 <= wavelength_nm <= 750: factor = 0.3 + 0.7 * (750 - wavelength_nm) / (750 - 700)
    # Aplica la corrección gamma para el brillo y devuelve el color.
    R = (R * factor) ** gamma; G = (G * factor) ** gamma; B = (B * factor) ** gamma
    return (R, G, B)

# --- Funciones Principales de Simulación ---

@st.cache_data
def calculate_diffraction(image_bytes, scale_factor, rotation_angle, padding_factor, lambda_m, D_m, width_m, invert_mask=False):
    """
    Calcula el patrón de difracción de Fraunhofer para una abertura arbitraria usando FFT.

    Esta función modela el fenómeno físico donde el patrón de difracción en el campo lejano
    es la Transformada de Fourier de la función de la abertura.

    Args:
        image_bytes (bytes): La imagen de la abertura en formato de bytes.
        scale_factor (float): Factor para escalar la abertura.
        rotation_angle (float): Ángulo para rotar la abertura en grados.
        padding_factor (float): Factor para añadir relleno (padding) a la máscara.
        lambda_m (float): Longitud de onda de la luz en metros.
        D_m (float): Distancia de la abertura a la pantalla en metros.
        width_m (float): Ancho físico de referencia de la imagen original en metros.
        invert_mask (bool): Si es True, invierte la máscara (abertura negativa).

    Returns:
        tuple: Contiene el patrón de intensidad (array 2D), el ancho físico de la pantalla
               y el alto físico de la pantalla (ambos en metros).
    """
    # --- 1. Preparación de la Función de Abertura t(x, y) ---
    # Se convierte la imagen en una matriz numérica que representa la abertura.
    image = Image.open(io.BytesIO(image_bytes))
    # Se convierte a escala de grises y se binariza: 1.0 donde pasa la luz, 0.0 donde se bloquea.
    # Esta matriz es la representación digital de la "función de transmisión de la abertura".
    mask_original = (np.array(image.convert('L')) / 255.0 >= 0.5).astype(float)
    if invert_mask:
        mask_original = 1 - mask_original  # Abertura negativa o complementaria.

    # --- 2. Aplicación de Transformaciones Geométricas ---
    Ny_orig, Nx_orig = mask_original.shape
    mask_transformada = mask_original
    if scale_factor != 1.0:
        mask_transformada = zoom(mask_transformada, scale_factor, order=0) # order=0 para mantener la binarización.
    if rotation_angle != 0:
        mask_transformada = rotate(mask_transformada, rotation_angle, reshape=True, order=0)
        mask_transformada = (mask_transformada > 0.5).astype(float) # Asegurar que siga binarizada tras la rotación.

    # --- 3. Padding (Aumento de la resolución en el espacio de frecuencias) ---
    # Se coloca la máscara en un lienzo más grande. Físicamente, esto no cambia la abertura,
    # pero matemáticamente, el zero-padding en el dominio espacial equivale a una interpolación
    # (mayor muestreo) en el dominio de frecuencias. El resultado es un patrón de difracción
    # más suave y detallado .
    Ny, Nx = mask_transformada.shape
    padded_mask = mask_transformada
    if padding_factor > 1.0:
        Ny_padded, Nx_padded = int(Ny * padding_factor), int(Nx * padding_factor)
        padded_mask = np.zeros((Ny_padded, Nx_padded))
        start_y, start_x = (Ny_padded - Ny) // 2, (Nx_padded - Nx) // 2
        padded_mask[start_y:start_y + Ny, start_x:start_x + Nx] = mask_transformada

    # --- 4. Simulación de la Difracción mediante FFT ---
    # El campo eléctrico E(kx, ky) en el plano de la pantalla (campo lejano) es la
    # Transformada de Fourier 2D de la función de la abertura t(x, y).
    E_field = np.fft.fft2(padded_mask)

    # La FFT coloca la componente de frecuencia cero (el centro del patrón) en la esquina.
    # fftshift la mueve al centro de la matriz, coincidiendo con la observación física.
    E_field_shifted = np.fft.fftshift(E_field)

    # La intensidad observable I es proporcional al módulo al cuadrado del campo eléctrico: I ∝ |E|².
    intensity_pattern = np.abs(E_field_shifted)**2

    # --- 5. Escalado Físico del Resultado ---
    # Se relaciona el tamaño en píxeles del resultado de la FFT con dimensiones físicas (metros).
    # Primero, se calcula el tamaño de un píxel de la abertura original en metros.
    Ny_final, Nx_final = padded_mask.shape
    pixel_size_m = width_m / Nx_orig
    dx_final = pixel_size_m # Tamaño de muestreo en el plano de la abertura.

    # De la teoría de la Transformada de Fourier Discreta, el ancho total en el espacio de frecuencias
    # es inversamente proporcional al tamaño del píxel en el espacio real.
    # La relación entre la coordenada en la pantalla x' y la frecuencia espacial kx es: kx = (2π/λD)x'.
    # Combinando estas relaciones, se obtiene el ancho físico del patrón de difracción.
    screen_physical_width = (lambda_m * D_m) / dx_final
    screen_physical_height = screen_physical_width * (Ny_final / Nx_final)

    return intensity_pattern, screen_physical_width, screen_physical_height

def get_preview_mask(image_bytes, scale_factor, rotation_angle, width_m, invert_mask=False):
    """
    Prepara una vista previa de la máscara de la abertura con sus transformaciones.

    Esta es una función auxiliar para la UI. Realiza los mismos pasos de procesamiento
    de imagen que `calculate_diffraction` pero sin el costoso cálculo de FFT,
    permitiendo una previsualización en tiempo real.

    Args:
        (Argumentos similares a calculate_diffraction)

    Returns:
        tuple: Contiene la máscara para previsualización (array 2D), y sus
               dimensiones físicas calculadas en mm (ancho y alto).
    """
    image = Image.open(io.BytesIO(image_bytes)); mask_original = (np.array(image.convert('L')) / 255.0 >= 0.5).astype(float)
    if invert_mask: mask_original = 1 - mask_original
    Ny_orig, Nx_orig = mask_original.shape; mask_transformada = mask_original
    if scale_factor != 1.0: mask_transformada = zoom(mask_transformada, scale_factor, order=0)
    if rotation_angle != 0:
        mask_transformada = rotate(mask_transformada, rotation_angle, reshape=True, order=0)
        mask_transformada = (mask_transformada > 0.5).astype(float)
    
    # Calcula las dimensiones físicas de la máscara transformada para mostrarlas al usuario.
    Ny_trans, Nx_trans = mask_transformada.shape
    pixel_size_m = width_m / Nx_orig
    physical_width_mm = Nx_trans * pixel_size_m * 1e3
    physical_height_mm = Ny_trans * pixel_size_m * 1e3
    
    # Prepara un lienzo cuadrado para centrar la vista previa.
    canvas_size = 512; canvas = np.zeros((canvas_size, canvas_size)); Ny, Nx = mask_transformada.shape
    start_y, start_x = (canvas_size - Ny) // 2, (canvas_size - Nx) // 2
    sy, sx = max(0, -start_y), max(0, -start_x); ey, ex = min(Ny, canvas_size - start_y), min(Nx, canvas_size - start_x)
    cy, cx = max(0, start_y), max(0, start_x); cey, cex = min(canvas_size, cy + (ey - sy)), min(canvas_size, cx + (ex - sx))
    if (ey - sy) > 0 and (ex - sx) > 0:
        canvas[cy:cey, cx:cex] = mask_transformada[sy:ey, sx:ex]
    return canvas, physical_width_mm, physical_height_mm

# --- Interfaz de Usuario (Streamlit) ---
st.title("🔬 Simulador de Difracción por FFT")

# --- Barra Lateral (Sidebar) para los Controles ---
st.sidebar.header("Controles de la Simulación")

# Callback para limpiar resultados si se sube una nueva imagen.
def clear_results_on_upload():
    if 'calculation_results' in st.session_state: del st.session_state.calculation_results
uploaded_file = st.sidebar.file_uploader("Sube una imagen para la abertura", type=["jpg", "png", "bmp"], on_change=clear_results_on_upload)

# Callback para sincronizar sliders y campos numéricos.
def sync_widget(master_key, widget_key):
    st.session_state[master_key] = st.session_state[widget_key]

# Valores por defecto para los parámetros de la simulación.
defaults = {
    "lambda_nm": 532.0, "D_m": 1.0, "factor_escala": 1.0, "angulo_rotacion": 0.0, 
    "factor_padding": 4.0, "view_mm": 5.0
}
for key, value in defaults.items():
    if key not in st.session_state: st.session_state[key] = value

# Controles para la manipulación de la abertura.
st.sidebar.markdown("---")
st.sidebar.subheader("Transformaciones de la Abertura")
invertir_mask = st.sidebar.checkbox("Invertir Abertura (Negativo) 🌓")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Factor de Escala", 0.1, 5.0, step=0.0001, format="%.4f", value=st.session_state.factor_escala, key="escala_slider", on_change=sync_widget, args=("factor_escala", "escala_slider")); col2.number_input("Escala", 0.1, 5.0, step=0.0001, format="%.4f", value=st.session_state.factor_escala, key="escala_input", on_change=sync_widget, args=("factor_escala", "escala_input"), label_visibility="collapsed")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Ángulo de Rotación (°)", 0.0, 360.0, step=0.0001, format="%.4f", value=st.session_state.angulo_rotacion, key="angulo_slider", on_change=sync_widget, args=("angulo_rotacion", "angulo_slider")); col2.number_input("Ángulo", 0.0, 360.0, step=0.0001, format="%.4f", value=st.session_state.angulo_rotacion, key="angulo_input", on_change=sync_widget, args=("angulo_rotacion", "angulo_input"), label_visibility="collapsed")

# Controles para los parámetros físicos del experimento.
st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros Físicos")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Longitud de Onda (λ, nm)", 380.0, 750.0, step=0.0001, format="%.4f", value=st.session_state.lambda_nm, key="lambda_slider", on_change=sync_widget, args=("lambda_nm", "lambda_slider")); col2.number_input("λ", 380.0, 750.0, step=0.0001, format="%.4f", value=st.session_state.lambda_nm, key="lambda_input", on_change=sync_widget, args=("lambda_nm", "lambda_input"), label_visibility="collapsed")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Distancia a Pantalla (D, m)", 0.1, 10.0, step=0.0001, format="%.4f", value=st.session_state.D_m, key="D_slider", on_change=sync_widget, args=("D_m", "D_slider")); col2.number_input("D", 0.1, 10.0, step=0.0001, format="%.4f", value=st.session_state.D_m, key="D_input", on_change=sync_widget, args=("D_m", "D_input"), label_visibility="collapsed")

# Controles para la calidad y visualización de la simulación.
st.sidebar.markdown("---")
st.sidebar.subheader("Simulación y Vista")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Padding", 1.0, 16.0, step=0.0001, format="%.4f", value=st.session_state.factor_padding, key="padding_slider", on_change=sync_widget, args=("factor_padding", "padding_slider")); col2.number_input("Padding", 1.0, 16.0, step=0.0001, format="%.4f", value=st.session_state.factor_padding, key="padding_input", on_change=sync_widget, args=("factor_padding", "padding_input"), label_visibility="collapsed")
col1, col2 = st.sidebar.columns([3, 2]); col1.slider("Ancho de Visualización (mm)", 1.0, 10.0, step=0.1, format="%.4f", value=st.session_state.view_mm, key="view_slider", on_change=sync_widget, args=("view_mm", "view_slider")); col2.number_input("Vista", 1.0, 10.0, step=0.1, format="%.4f", value=st.session_state.view_mm, key="view_input", on_change=sync_widget, args=("view_mm", "view_input"), label_visibility="collapsed")

st.sidebar.markdown("---")
# Botón para iniciar el cálculo.
if st.sidebar.button("Calcular Difracción", type="primary", use_container_width=True):
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        D_m = st.session_state.D_m
        # Se establece un ancho físico de referencia para la imagen de entrada.
        # Esto es crucial para dar una escala real a la simulación.
        width_m = 5e-3  # 5 mm
        lambda_m = st.session_state.lambda_nm * 1e-9 # Convertir nm a m.
        with st.spinner('Calculando...'):
            # Llamada a la función principal de cálculo.
            st.session_state.calculation_results = calculate_diffraction(
                image_bytes, st.session_state.factor_escala, st.session_state.angulo_rotacion, 
                st.session_state.factor_padding, lambda_m, D_m, width_m, invert_mask=invertir_mask
            )
    else:
        st.sidebar.warning("Por favor, sube una imagen primero.")

# --- Disposición de los Resultados en la Página Principal ---
col1, col2 = st.columns(2)

# Columna izquierda: Vista previa de la abertura.
with col1:
    st.subheader("Vista Previa de la Abertura")
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        width_m = 5e-3 # Ancho de referencia.
        preview_mask, p_width_mm, p_height_mm = get_preview_mask(
            image_bytes, st.session_state.factor_escala, 
            st.session_state.angulo_rotacion, width_m, invert_mask=invertir_mask
        )
        st.image((preview_mask * 255).astype(np.uint8), use_container_width=True, caption="Ajuste en tiempo real.")
        
        st.markdown("###### Dimensiones Físicas de la Abertura")
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Ancho", f"{p_width_mm:.3f} mm")
        mcol2.metric("Alto", f"{p_height_mm:.3f} mm")
    else:
        st.info("Sube una imagen para ver la vista previa.")

# Columna derecha: Visualización del patrón de difracción calculado.
with col2:
    st.subheader("Patrón de Difracción")
    if 'calculation_results' in st.session_state:
        intensity_pattern, screen_w, screen_h = st.session_state.calculation_results
        view_m = st.session_state.view_mm * 1e-3 # Ancho de la ventana de visualización.
        
        # Preparación del gráfico con Matplotlib.
        fig, ax = plt.subplots()
        # Se usa una escala logarítmica (log1p = log(1+x)) para realzar los detalles tenues.
        log_intensity = np.log1p(intensity_pattern)
        
        # Creación de un mapa de color personalizado del negro al color de la luz.
        visible_color = wavelength_to_rgb(st.session_state.lambda_nm)
        my_cmap = LinearSegmentedColormap.from_list('custom_wavelength', ['black', visible_color], N=256)
        
        # Extensión física del gráfico en centímetros.
        plot_extent = [-screen_w / 2 * 100, screen_w / 2 * 100, -screen_h / 2 * 100, screen_h / 2 * 100]
        im = ax.imshow(log_intensity, cmap=my_cmap, extent=plot_extent)
        
        # Limita la vista al ancho seleccionado por el usuario.
        lim_cm = view_m / 2 * 100; view_h = view_m * (screen_h / screen_w); lim_y_cm = view_h / 2 * 100
        ax.set_xlim([-lim_cm, lim_cm]); ax.set_ylim([-lim_y_cm, lim_y_cm])
        
        ax.set_xlabel("Posición en pantalla x' (cm)"); ax.set_ylabel("Posición en pantalla y' (cm)")
        ax.set_facecolor('black'); fig.colorbar(im, ax=ax, label='Intensidad Relativa (escala log)', fraction=0.046, pad=0.04)
        fig.tight_layout()
        
        # Muestra el gráfico en la aplicación Streamlit.
        st.pyplot(fig)
        st.caption("Patrón de luz resultante en la pantalla distante.")
    else:
        st.info("Haz clic en 'Calcular Difracción' para ver el resultado.")
