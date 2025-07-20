# pages/1_Simulador_Difraccion.py
import streamlit as st
import numpy as np
from PIL import Image
from scipy.ndimage import zoom, rotate
import io
import matplotlib.pyplot as plt

# NOTA: Muevo la configuración de la página aquí para que cada app tenga su propio título y icono.
st.set_page_config(page_title="Simulador de Difracción", page_icon="🔬")

@st.cache_data
def calculate_diffraction(image_bytes, scale_factor, rotation_angle, padding_factor, lambda_m, D_m, width_m):
    """
    Realiza la cadena de cálculo: transformación, padding y FFT.
    """
    image = Image.open(io.BytesIO(image_bytes))
    mask_original = (np.array(image.convert('L')) / 255.0 >= 0.5).astype(float)

    # 1. Aplicar transformaciones
    mask_transformada = mask_original
    if scale_factor != 1.0:
        mask_transformada = zoom(mask_transformada, scale_factor, order=0)
    if rotation_angle != 0:
        mask_transformada = rotate(mask_transformada, rotation_angle, reshape=True, order=0)
        mask_transformada = (mask_transformada > 0.5).astype(float)

    # 2. Aplicar Padding
    Ny, Nx = mask_transformada.shape
    padded_mask = mask_transformada
    if padding_factor > 1.0:
        Ny_padded, Nx_padded = int(Ny * padding_factor), int(Nx * padding_factor)
        padded_mask = np.zeros((Ny_padded, Nx_padded))
        start_y, start_x = (Ny_padded - Ny) // 2, (Nx_padded - Nx) // 2
        padded_mask[start_y:start_y + Ny, start_x:start_x + Nx] = mask_transformada

    # 3. Calcular FFT
    E_field = np.fft.fft2(padded_mask)
    E_field_shifted = np.fft.fftshift(E_field)
    intensity_pattern = np.abs(E_field_shifted)**2

    # 4. Calcular dimensiones físicas
    Ny_final, Nx_final = padded_mask.shape
    aperture_physical_width_final = width_m * scale_factor
    dx = aperture_physical_width_final / Nx
    screen_physical_width = (lambda_m * D_m) / dx
    screen_physical_height = screen_physical_width * (Ny_final / Nx_final)
    
    return intensity_pattern, screen_physical_width, screen_physical_height

def get_preview_mask(image_bytes, scale_factor, rotation_angle):
    """
    Genera la vista previa en tiempo real de la abertura transformada.
    """
    image = Image.open(io.BytesIO(image_bytes))
    mask_original = (np.array(image.convert('L')) / 255.0 >= 0.5).astype(float)
    mask_transformada = mask_original
    if scale_factor != 1.0:
        mask_transformada = zoom(mask_transformada, scale_factor, order=0)
    if rotation_angle != 0:
        mask_transformada = rotate(mask_transformada, rotation_angle, reshape=True, order=0)
        mask_transformada = (mask_transformada > 0.5).astype(float)
    canvas_size = 512
    canvas = np.zeros((canvas_size, canvas_size))
    Ny, Nx = mask_transformada.shape
    start_y, start_x = (canvas_size - Ny) // 2, (canvas_size - Nx) // 2
    sy, sx = max(0, -start_y), max(0, -start_x)
    ey, ex = min(Ny, canvas_size - start_y), min(Nx, canvas_size - start_x)
    cy, cx = max(0, start_y), max(0, start_x)
    cey, cex = min(canvas_size, cy + (ey - sy)), min(canvas_size, cx + (ex - sx))
    if (ey - sy) > 0 and (ex - sx) > 0:
      canvas[cy:cey, cx:cex] = mask_transformada[sy:ey, sx:ex]
    return canvas

# --- El resto de tu código de UI permanece exactamente igual ---
st.title("🔬 Simulador Interactivo de Difracción de Fraunhofer")
st.sidebar.header("Controles de la Simulación")
def clear_results_on_upload():
    if 'calculation_results' in st.session_state:
        del st.session_state.calculation_results
uploaded_file = st.sidebar.file_uploader(
    "Sube una imagen para la abertura", type=["jpg", "png", "bmp"], on_change=clear_results_on_upload
)
st.sidebar.markdown("---")
st.sidebar.subheader("Transformaciones de la Abertura")
st.sidebar.info("Para mayor detalle en la difracción, aumenta el factor de escala.")
factor_escala = st.sidebar.slider("Factor de Escala (Resolución)", 0.2, 2.0, 1.0, 0.1)
angulo_rotacion = st.sidebar.slider("Ángulo de Rotación (°)", 0, 360, 0, 1)
st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros Físicos")
lambda_nm = st.sidebar.slider("Longitud de Onda (λ)", 300, 750, 532, 10)
st.sidebar.markdown("---")
st.sidebar.subheader("Simulación y Vista")
st.sidebar.info("Para suavizar el resultado, aumenta el factor de padding.")
factor_padding = st.sidebar.slider("Fineza (Factor de Padding)", 1.0, 8.0, 4.0, 0.5)
view_cm = st.sidebar.slider("Ancho de Visualización (Zoom)", 1.0, 50.0, 10.0, 0.5)
st.sidebar.markdown("---")
if st.sidebar.button("Calcular Difracción", type="primary", use_container_width=True):
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        D_m = 1
        width_m = 150e-6
        lambda_m = lambda_nm * 1e-9
        with st.spinner('Calculando...'):
            st.session_state.calculation_results = calculate_diffraction(
                image_bytes, factor_escala, angulo_rotacion, factor_padding, lambda_m, D_m, width_m
            )
    else:
        st.sidebar.warning("Por favor, sube una imagen primero.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Vista Previa de la Abertura")
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        preview_mask = get_preview_mask(image_bytes, factor_escala, angulo_rotacion)
        st.image((preview_mask * 255).astype(np.uint8), use_container_width=True, caption="Ajuste en tiempo real.")
    else:
        st.info("Sube una imagen para ver la vista previa.")
with col2:
    st.subheader("Patrón de Difracción")
    if 'calculation_results' in st.session_state:
        intensity_pattern, screen_w, screen_h = st.session_state.calculation_results
        view_m = view_cm * 1e-2
        fig, ax = plt.subplots()
        log_intensity = np.log1p(intensity_pattern)
        plot_extent = [-screen_w / 2 * 100, screen_w / 2 * 100, -screen_h / 2 * 100, screen_h / 2 * 100]
        ax.imshow(log_intensity, cmap='hot', extent=plot_extent)
        lim_cm = view_m / 2 * 100
        view_h = view_m * (screen_h / screen_w)
        lim_y_cm = view_h / 2 * 100
        ax.set_xlim([-lim_cm, lim_cm])
        ax.set_ylim([-lim_y_cm, lim_y_cm])
        ax.set_xlabel("Posición en pantalla x' (cm)")
        ax.set_ylabel("Posición en pantalla y' (cm)")
        ax.set_facecolor('black')
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Patrón de luz resultante en la pantalla distante.")
    else:
        st.info("Haz clic en 'Calcular Difracción' para ver el resultado.")