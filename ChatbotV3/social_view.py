import streamlit as st
import googlemaps

# --- LÓGICA INTERNA (PRIVADA) ---
def _consultar_google_maps(ubicacion, categoria, api_key):
    """
    Función interna que conecta con la API.
    """
    try:
        gmaps = googlemaps.Client(key=api_key)
        query = f"{categoria} en {ubicacion}"
        resultados = gmaps.places(query=query)
        
        if resultados['status'] == 'OK':
            lugares = []
            for lugar in resultados['results'][:5]:
                nombre = lugar.get('name')
                rating = lugar.get('rating', 'N/A')
                direccion = lugar.get('formatted_address')
                lugares.append(f"📍 **{nombre}** (⭐ {rating})\n   🏠 {direccion}")
            
            return f"He encontrado estos sitios en Google Maps:\n\n" + "\n\n".join(lugares)
        else:
            return "Google no encontró resultados para esa búsqueda."
    except Exception as e:
        return f"Error de conexión: {str(e)}"

# --- FUNCIÓN PÚBLICA (LA QUE LLAMARÁS DESDE APP.PY) ---
def renderizar_sidebar():
    """
    Dibuja la barra lateral y devuelve el resultado de la búsqueda si hubo uno.
    Si no se buscó nada, devuelve None.
    """
    resultado_para_el_chat = None

    with st.sidebar:
        st.title("SocialView 🌍")
        st.caption("Módulo de Georreferenciación")
        st.divider()

        # Gestión de la API Key
        if "GOOGLE_MAPS_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_MAPS_KEY"]
            st.success("✅ API Conectada")
        else:
            api_key = st.text_input("🔑 API Key Google:", type="password")

        # Inputs
        ubicacion = st.text_input("📍 Ubicación:", placeholder="Ej: Madrid")
        categoria = st.selectbox("Buscar:", ["Parques", "Gimnasios", "Restaurantes", "Hospitales"])

        # Botón
        if st.button("Buscar"):
            if api_key and ubicacion:
                with st.spinner("Buscando..."):
                    resultado = _consultar_google_maps(ubicacion, categoria, api_key)
                    
                    # Mostramos un adelanto en la barra
                    st.info("¡Datos enviados al chat!")
                    
                    # Devolvemos el texto para que app.py lo use
                    resultado_para_el_chat = resultado
            else:
                st.warning("Falta ubicación o API Key")

    return resultado_para_el_chat
