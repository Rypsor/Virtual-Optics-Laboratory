# app_principal.py
import streamlit as st

st.set_page_config(
    page_title="Mi Laboratorio Virtual",
    page_icon="🧪",
)

st.title("Mi Laboratorio Virtual 🧪")

st.sidebar.success("Selecciona una herramienta del laboratorio.")

st.markdown(
    """
    ### ¡Bienvenido a mi colección de herramientas científicas!
    
    Este es un proyecto que agrupa varias simulaciones y utilidades
    de física e ingeniería.
    
    **👈 Elige una de las aplicaciones en la barra lateral** para comenzar.
    """
)