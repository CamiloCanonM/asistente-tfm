import streamlit as st
import pandas as pd
import googlemaps
from streamlit_js_eval import get_geolocation
from langchain_core.messages import AIMessage

# ==========================================
# 1. CONFIGURACIÓN Y CONEXIÓN
# ==========================================
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

# ==========================================
# 2. MOTOR DE BÚSQUEDA (LÓGICA)
# ==========================================
def buscar_lugares_google(ciudad, palabra_clave, lat_gps=None, lon_gps=None):
    gmaps = obtener_cliente_google()
    
    # Validación de seguridad
    if not gmaps: 
        return None, "Error: No se detectó la API Key de Google."

    try:
        # --- A. DEFINIR EL CENTRO DEL MAPA ---
        if lat_gps is not None and lon_gps is not None:
            # Opción 1: Usamos el GPS del usuario
            lat_centro, lon_centro = lat_gps, lon_gps
            origen_texto = "tu ubicación actual"
        else:
            # Opción 2: Usamos la ciudad escrita
            geocode_result = gmaps.geocode(ciudad)
            if not geocode_result: 
                return None, f"No encontré la ciudad: {ciudad}"
            
            loc = geocode_result[0]['geometry']['location']
            lat_centro, lon_centro = loc['lat'], loc['lng']
            origen_texto = ciudad

        # --- B. CONSULTAR API DE LUGARES (PLACES) ---
        places_result = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000, # Radio de 2km
            open_now=False
        )

        lugares = places_result.get('results', [])
        if not lugares: 
            return None, f"No encontré {palabra_clave} cerca de {origen_texto}."

        # --- C. CONSTRUIR LA TABLA DE DATOS ---
        data_completa = []

        # 1. Agregamos al USUARIO (Punto AZUL) 🔵
        if lat_gps is not None and lon_gps is not None:
            data_completa.append({
                'latitude': float(lat_gps),
                'longitude': float(lon_gps),
                'name': "🔵 TU UBICACIÓN",
                'address': "Estás aquí",
                'rating': "YO",
                'place_id': "USER_LOC", # ID especial interno
                'color': '#0000FF',     # Azul Puro
                'size': 40              # Tamaño mediano
            })

        # 2. Agregamos los LUGARES (Puntos ROJOS) 🔴
        for l in lugares:
            lat = l['geometry']['location']['lat']
            lng = l['geometry']['location']['lng']
            name = l.get('name', 'Sin nombre')
            addr = l.get('vicinity', 'Dirección desconocida')
            rating = l.get('rating', 'N/A')
            pid = l.get('place_id')

            data_completa.append({
                'latitude': float(lat),
                'longitude': float(lng),
                'name': name,
                'address': addr,
                'rating': rating,
                'place_id': pid,
                'color': '#FF0000', # Rojo Puro
                'size': 20          # Tamaño normal
            })
            
        df_resultado = pd.DataFrame(data_completa)
        return df_resultado, f"✅ Encontré {len(lugares)} {palabra_clave} cerca de {origen_texto}."

    except Exception as e:
        return None, f"Error técnico: {str(e)}"

# ==========================================
# 3. INTERFAZ GRÁFICA (RENDERIZADO)
# ==========================================
def renderizar_sidebar():
    st.subheader("📍 Mapa Interactivo")
    
    # --- A. GESTIÓN DE MEMORIA (STATE) ---
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None
    if "lugar_seleccionado_id" not in st.session_state:
        st.session_state.lugar_seleccionado_id = None

    # --- B. CONTROLES DE USUARIO ---
    tipo = st.selectbox("¿Qué buscas?", ["Farmacias", "Hospitales", "Parques", "Gimnasios", "Restaurantes"])
    
    usar_gps = st.checkbox("📍 Usar mi GPS")
    
    lat_usuario, lon_usuario = None, None
    ciudad = ""

    if usar_gps:
        loc = get_geolocation() # Pide permiso al navegador
        if loc:
            lat_usuario = loc['coords']['latitude']
            lon_usuario = loc['coords']['longitude']
            st.caption(f"📡 GPS OK")
        else:
            st.warning("⚠️ Esperando señal GPS...")
    else:
        ciudad = st.text_input("Ciudad:", "Bogota")

    # --- C. BOTÓN PRINCIPAL DE BÚSQUEDA ---
    if st.button("🔍 BUSCAR AHORA", use_container_width=True):
        if usar_gps and lat_usuario is None:
            st.error("El GPS aún no ha cargado. Espera un segundo.")
        else:
            with st.spinner("Conectando satélite..."):
                # Llamamos a la lógica
                df, msg = buscar_lugares_google(ciudad, tipo, lat_gps=lat_usuario, lon_gps=lon_usuario)
                
                if df is not None:
                    st.session_state.mapa_data = df
                    st.session_state.lugar_seleccionado_id = None # Limpiamos selecciones previas
                    # Guardamos mensaje en el chat
                    st.session_state.chat_history.append(AIMessage(content=f"[SISTEMA MAPAS]: {msg}"))
                    st.rerun() # Recargamos para mostrar el mapa fresco
                else:
                    st.error(msg)

    # --- D. MOSTRAR RESULTADOS (SI EXISTEN) ---
    if st.session_state.mapa_data is not None:
        
        st.divider()
        
        # 1. PREPARACIÓN VISUAL DEL MAPA
        # Hacemos una copia para pintar de verde sin dañar los datos originales
        df_display = st.session_state.mapa_data.copy()
        selected_id = st.session_state.lugar_seleccionado_id
        
        # LÓGICA DE RESALTADO (HIGHLIGHT) 🟢
        if selected_id:
            # Buscamos la fila con ese ID y le cambiamos el color y tamaño
            mask = df_display['place_id'] == selected_id
            df_display.loc[mask, 'color'] = '#00FF00' # Verde Neón
            df_display.loc[mask, 'size'] = 500        # GIGANTE
            
            st.info("🎯 Lugar seleccionado resaltado en VERDE.")
            
            # Botón para quitar el resaltado
            if st.button("🔙 Ver mapa normal", use_container_width=True):
                st.session_state.lugar_seleccionado_id = None
                st.rerun()

        # 2. PINTAR EL MAPA
        # Usamos las columnas 'color' y 'size' que preparamos
        st.map(df_display, color='color', size='size', use_container_width=True)

        # 3. LISTA INTERACTIVA (TARJETAS)
        st.caption("Resultados detallados (Haz clic para expandir):")
        
        # Filtramos para que TU UBICACIÓN no salga en la lista de texto
        df_lista = st.session_state.mapa_data[st.session_state.mapa_data['place_id'] != "USER_LOC"].reset_index(drop=True)

        for i, row in df_lista.iterrows():
            # Título de la tarjeta
            with st.expander(f"🏥 {row['name']} ({row['rating']}⭐)"):
                st.markdown(f"**Dirección:** {row['address']}")
                
                # --- BOTÓN 1: SEÑALAR EN MAPA ---
                # Al hacer clic, guardamos el ID en memoria y recargamos
                if st.button("🎯 SEÑALAR EN EL MAPA", key=f"btn_focus_{i}", use_container_width=True):
                    st.session_state.lugar_seleccionado_id = row['place_id']
                    st.rerun()

                # --- BOTÓN 2: NAVEGAR CON GOOGLE ---
                # Enlace universal para abrir app de mapas
                link_google = f"https://www.google.com/maps/search/?api=1&query={row['name'].replace(' ', '+')}&query_place_id={row['place_id']}"
                st.link_button("🚗 IR CON GOOGLE MAPS", link_google, use_container_width=True)

        # Botón final de limpieza
        st.divider()
        if st.button("🗑️ Limpiar Todo"):
            st.session_state.mapa_data = None
            st.session_state.lugar_seleccionado_id = None
            st.rerun()
