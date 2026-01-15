import streamlit as st
import pandas as pd
import googlemaps
import math
from streamlit_js_eval import get_geolocation
from langchain_core.messages import AIMessage

# ==========================================
# 1. LOGICA MATEMÁTICA Y GOOGLE
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
        # Intenta obtener la API Key de secrets o input temporal
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None, "⚠️ Falta GOOGLE_API_KEY en .streamlit/secrets.toml"
        
        gmaps = googlemaps.Client(key=api_key)
    except: return None, "Error iniciando cliente Google Maps."

    try:
        if lat_gps and lon_gps:
            lat_ref, lon_ref = lat_gps, lon_gps
        else:
            geo = gmaps.geocode(ciudad)
            if not geo: return None, f"No encontré la ciudad: {ciudad}"
            loc = geo[0]['geometry']['location']
            lat_ref, lon_ref = loc['lat'], loc['lng']

        # Busqueda
        if not palabra_clave: palabra_clave = "restaurante"
        
        # Radio de 2km (2000 metros)
        res = gmaps.places_nearby(location=(lat_ref, lon_ref), keyword=palabra_clave, radius=2000)
        lugares = res.get('results', [])
        
        if not lugares: return None, "No hay resultados cercanos."

        data = []
        # Añadir usuario si usa GPS para que salga en el mapa
        if lat_gps and lon_gps:
            data.append({
                'latitude': lat_ref, 'longitude': lon_ref, 
                'name': "🔵 TU UBICACIÓN", 'place_id': "USER_LOC", 
                'color': '#0000FF', 'size': 400, 'distancia': 0.0, 
                'address': 'Ubicación actual', 'rating': 'YO'
            })

        for l in lugares:
            lat = l['geometry']['location']['lat']
            lng = l['geometry']['location']['lng']
            km = calcular_distancia(lat_ref, lon_ref, lat, lng)
            data.append({
                'latitude': lat, 'longitude': lng, 
                'name': l.get('name'), 
                'address': l.get('vicinity'), 
                'rating': l.get('rating', 'N/A'), 
                'place_id': l.get('place_id'), 
                'color': '#FF0000', 'size': 200, 'distancia': km
            })
            
        return pd.DataFrame(data).sort_values(by='distancia'), f"Encontrados {len(lugares)}"

    except Exception as e: return None, str(e)

# ==========================================
# 2. RENDERIZADO BARRA LATERAL (CONTROLES)
# ==========================================
def renderizar_sidebar():
    # Inicialización de estado
    if "mapa_data" not in st.session_state: st.session_state.mapa_data = None
    if "lugar_seleccionado_id" not in st.session_state: st.session_state.lugar_seleccionado_id = None

    # Inputs
    tipo = st.text_input("🔍 ¿Qué buscas?", placeholder="Ej: Farmacia, Parque, Gym...")
    if not tipo: tipo = "Farmacia" # Valor por defecto seguro

    usar_gps = st.checkbox("📍 Usar mi GPS")
    
    lat, lon = None, None
    if usar_gps:
        try:
            loc = get_geolocation() # Requiere streamlit_js_eval
            if loc: lat, lon = loc['coords']['latitude'], loc['coords']['longitude']
        except:
            st.warning("Habilita el GPS en tu navegador.")
    
    ciudad = st.text_input("Ciudad:", "Madrid") if not usar_gps else ""

    if st.button("🔍 BUSCAR LUGARES"):
        if usar_gps and not lat:
            st.error("⏳ Esperando señal GPS (permite la ubicación)...")
        else:
            with st.spinner(f"Buscando {tipo}..."):
                df, msg = buscar_lugares_google(ciudad, tipo, lat, lon)
                
                if df is not None:
                    # 1. Guardar datos del mapa
                    st.session_state.mapa_data = df
                    st.session_state.lugar_seleccionado_id = None
                    
                    # -------------------------------------------------------
                    # 🔥 EL PUENTE CON LA IA (AQUÍ ESTÁ LA MAGIA)
                    # -------------------------------------------------------
                    texto_para_ia = f"Resultados de búsqueda para '{tipo}':\n"
                    count = 0
                    for index, row in df.iterrows():
                        if row['place_id'] != "USER_LOC" and count < 5: # Pasamos los 5 más cercanos
                            texto_para_ia += f"- {row['name']} (a {row['distancia']} km) - Calif: {row['rating']}\n"
                            count += 1
                    
                    # Guardamos en la memoria global para que app.py lo lea
                    st.session_state.geo_contexto = texto_para_ia
                    st.toast("✅ IA Actualizada con nuevos lugares")
                    # -------------------------------------------------------

                    st.rerun()
                else: 
                    st.error(msg)


# ==========================================
# 3. RENDERIZADO CENTRAL (MAPA + TARJETAS)
# ==========================================
def mostrar_mapa_central():
    if st.session_state.get("mapa_data") is not None:
        
        st.divider()
        st.subheader("🗺️ Mapa Interactivo")
        
        df_show = st.session_state.mapa_data.copy()
        sid = st.session_state.lugar_seleccionado_id
        
        # Resaltar selección (Ponemos el punto verde y grande)
        if sid:
            df_show.loc[df_show['place_id'] == sid, ['color', 'size']] = ['#00FF00', 800]

        # Mapa nativo de Streamlit (Compatible con V3)
        st.map(df_show, color='color', size='size', use_container_width=True) 
        
        if sid and st.button("🔙 Ver todos", key="btn_reset"):
            st.session_state.lugar_seleccionado_id = None
            st.rerun()

        st.markdown("### 📍 Listado de Lugares")

        # Filtramos para no mostrar la ubicación del usuario en las tarjetas
        lugares = df_show[df_show['place_id'] != "USER_LOC"]
        
        # Grid de tarjetas
        cols = st.columns(3)

        for i, (index, r) in enumerate(lugares.iterrows()):
            col_actual = cols[i % 3] 
            
            with col_actual:
                # HTML Card Estilo V3 (Fondo Blanco, Letras Negras)
                card_html = f"""
                <div style="
                    background-color: #ffffff; 
                    border: 1px solid #e0e0e0; 
                    border-radius: 12px; 
                    padding: 15px; 
                    margin-bottom: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
                    height: 200px; 
                    display: flex; 
                    flex-direction: column; 
                    justify-content: space-between;">
                    
                    <div>
                        <h4 style="margin:0 0 5px 0; color:#000000; font-size:16px; font-weight:bold;">{r['name']}</h4>
                        <p style="margin:0; color:#555555; font-size:12px; line-height:1.4;">{r['address']}</p>
                    </div>
                    
                    <div style="margin-top:10px; border-top:1px solid #f0f0f0; padding-top:8px; display:flex; justify-content:space-between; align-items:center;">
                        <span style="background:#E3F2FD; color:#1565C0; padding:4px 8px; border-radius:8px; font-size:12px; font-weight:bold;">
                            📍 {r['distancia']} km
                        </span>
                        <span style="color:#F5A623; font-weight:bold; font-size:12px;">
                            ⭐ {r['rating']}
                        </span>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Botones de Acción
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🎯 Foco", key=f"focus_{r['place_id']}", use_container_width=True):
                        st.session_state.lugar_seleccionado_id = r['place_id']
                        st.rerun()
                with c2:
                    # Link oficial de Google Maps
                    link = f"https://www.google.com/maps/search/?api=1&query={r['name'].replace(' ', '+')}&query_place_id={r['place_id']}"
                    st.link_button("🗺️ Ir", link, use_container_width=True)

        st.write("")
        if st.button("🗑️ Cerrar Mapa", key="close_map_main"):
            st.session_state.mapa_data = None
            st.rerun()
