import streamlit as st
import pandas as pd
import googlemaps
import math
from streamlit_js_eval import get_geolocation
from langchain_core.messages import AIMessage

# ==========================================
# 1. LOGICA Y CALCULOS
# ==========================================
def calcular_distancia(lat1, lon1, lat2, lon2):
    R = 6371  
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)

def buscar_lugares_google(ciudad, palabra_clave, lat_gps=None, lon_gps=None):
    try:
        gmaps = googlemaps.Client(key=st.secrets.get("GOOGLE_API_KEY"))
    except: return None, "Error API Key"

    try:
        if lat_gps and lon_gps:
            lat_ref, lon_ref = lat_gps, lon_gps
        else:
            geo = gmaps.geocode(ciudad)
            if not geo: return None, "Ciudad no encontrada"
            loc = geo[0]['geometry']['location']
            lat_ref, lon_ref = loc['lat'], loc['lng']

        res = gmaps.places_nearby(location=(lat_ref, lon_ref), keyword=palabra_clave, radius=2000)
        lugares = res.get('results', [])
        
        if not lugares: return None, "No hay resultados."

        data = []
        if lat_gps and lon_gps:
            data.append({'latitude': lat_ref, 'longitude': lon_ref, 'name': "🔵 TU UBICACIÓN", 'place_id': "USER_LOC", 'color': '#0000FF', 'size': 40, 'distancia': 0.0, 'address': 'Aquí', 'rating': 'YO'})

        for l in lugares:
            lat, lng = l['geometry']['location']['lat'], l['geometry']['location']['lng']
            km = calcular_distancia(lat_ref, lon_ref, lat, lng)
            data.append({
                'latitude': lat, 'longitude': lng, 
                'name': l.get('name'), 
                'address': l.get('vicinity'), 
                'rating': l.get('rating', 'N/A'), 
                'place_id': l.get('place_id'), 
                'color': '#FF0000', 'size': 20, 'distancia': km
            })
            
        return pd.DataFrame(data).sort_values(by='distancia'), f"Encontrados {len(lugares)}"

    except Exception as e: return None, str(e)

# ==========================================
# 2. SOLO CONTROLES (Para la Barra Lateral)
# ==========================================
def renderizar_sidebar():
    st.subheader("🌎 Modulo de Georreferenciación")
    if "mapa_data" not in st.session_state: st.session_state.mapa_data = None
    if "lugar_seleccionado_id" not in st.session_state: st.session_state.lugar_seleccionado_id = None

   tipo = st.text_input("🔍 ¿Qué buscas?", placeholder="Ej: farmacias...")
    usar_gps = st.checkbox("📍 Usar GPS")
    
    lat, lon = None, None
    if usar_gps:
        loc = get_geolocation()
        if loc: lat, lon = loc['coords']['latitude'], loc['coords']['longitude']
    
    ciudad = st.text_input("Ciudad:", "Bogota") if not usar_gps else ""

    # ESTE BOTON SOLO GUARDA LOS DATOS, NO PINTA EL MAPA AQUI
    if st.button("🔍 BUSCAR ", use_container_width=True):
        if usar_gps and not lat:
            st.error("Esperando GPS...")
        else:
            with st.spinner("Buscando..."):
                df, msg = buscar_lugares_google(ciudad, tipo, lat, lon)
                if df is not None:
                    st.session_state.mapa_data = df
                    st.session_state.lugar_seleccionado_id = None
                    # Agregamos mensaje al chat para confirmar
                    st.session_state.chat_history.append(AIMessage(content=f"✅ He cargado el mapa de **{tipo}** en el panel central."))
                    st.rerun()
                else: st.error(msg)

# ==========================================
# 3. SOLO VISUALIZACION (Para el Centro)
# ==========================================
def mostrar_mapa_central():
    # Solo mostramos si hay datos cargados
    if st.session_state.get("mapa_data") is not None:
        
        st.divider()
        st.subheader("🗺️ Resultados de la Búsqueda")
        
        df_show = st.session_state.mapa_data.copy()
        sid = st.session_state.lugar_seleccionado_id
        
        # Logica de resaltado
        if sid:
            df_show.loc[df_show['place_id'] == sid, ['color', 'size']] = ['#00FF00', 100] # Verde Gigante

        # 1. EL MAPA (GRANDE EN EL CENTRO)
        st.map(df_show, color='color', size='size', use_container_width=True)
        
        # Boton reset foco
        if sid:
            if st.button("🔙 Quitar Zoom", key="btn_reset_center"):
                st.session_state.lugar_seleccionado_id = None
                st.rerun()

        # 2. LA LISTA DEBAJO DEL MAPA
        for i, r in df_show[df_show['place_id'] != "USER_LOC"].iterrows():
            with st.expander(f"{'🥇' if i==0 else '📍'} {r['name']} ({r['distancia']} km)"):
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.write(f"🏠 {r['address']}")
                    st.write(f"⭐ {r['rating']}")
                with c2:
                    if st.button("🎯 Ver en Mapa", key=f"btn_c_{i}", use_container_width=True):
                        st.session_state.lugar_seleccionado_id = r['place_id']
                        st.rerun()
                    
                    link = f"https://www.google.com/maps/search/?api=1&query={r['name'].replace(' ', '+')}&query_place_id={r['place_id']}"
                    st.link_button("🚗 Ir con GPS", link, use_container_width=True)

        if st.button("🗑️ Cerrar Mapa", key="close_map_center"):
            st.session_state.mapa_data = None
            st.rerun()
