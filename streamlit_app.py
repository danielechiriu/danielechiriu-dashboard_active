import streamlit as st
import pandas as pd
import plotly.express as px
from math import radians, sin, cos, sqrt, atan2
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime, timedelta
import math
from openai import OpenAI

st.set_page_config(page_title="Dashboard Activelabel", layout="wide")
st.markdown("""
    <style>
        /* Sfondo principale e contenitori */
        body, .stApp {
            background-color: white !important;
            color: black !important;
        }

        /* Testo e componenti principali */
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: white !important;
        }

        [data-testid="stSidebar"], section[data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
        }

        /* Tabelle e widget */
        div[data-testid="stDataFrame"] {
            background-color: white !important;
        }
        div[data-testid="stDataFrame"] * {
            color: black !important;
        }

        /* Metriche e card */
        div[data-testid="stMetric"] {
            background-color: #f8f9fa !important;
            color: #2E4053 !important;
        }

        /* Rimuove eventuale overlay scuro del tema */
        [class*="st-emotion-cache"] {
            background-color: white !important;
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=250)
with col2:
    st.markdown(
        """
        <div style='display: flex; align-items: center; height: 100%; padding-top: 45px;'>
            <h1 style='font-size: 60px; margin: 0;'>Dashboard Activelabel</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
<style>
div[data-testid="stDataFrame"] table {
    border-radius: 10px;
    border: 1px solid #ddd;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
thead tr th {
    background-color: #f4f6f8 !important;
    color: #2E4053 !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# Link al file Drive (pubblico, leggibile)
file_id = "1Zp_2qP8Td1TVMdbY9mlzOHYxIlICmNZL"
url = f"https://drive.google.com/uc?id={file_id}&export=download"

geolocator = Nominatim(user_agent="dashboard_active_label")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def get_province(lat, lon):
    try:
        location = reverse((lat, lon), language="it")
        if location and "address" in location.raw:
            address = location.raw["address"]
            # 🔍 Proviamo diversi campi in ordine di importanza
            return (
                address.get("state_district") or
                address.get("county") or
                address.get("state") or
                address.get("region") or
                address.get("city") or
                address.get("town") or
                address.get("village") or
                address.get("municipality") or
                address.get("country")
            )
    except Exception:
        return None
    return None

try:
    # Caricamento del file
    df = pd.read_csv(url, sep=" ", header=None)
    df.columns = ["Operator", "Device", "Reading date", "Reading hour", "Read value", "Effective T", "QR", "Desired T",
                  "Item ID", "Writing date", "Writing hour", "LAT", "LON"]

    @st.cache_data(show_spinner="Determinazione province in corso...")
    def get_province_cached(lat, lon):
        try:
            location = reverse((lat, lon), language="it")
            if location and "address" in location.raw:
                address = location.raw["address"]
                # 🔍 Proviamo diversi campi in ordine di importanza
                return (
                        address.get("state_district") or
                        address.get("county") or
                        address.get("state") or
                        address.get("region") or
                        address.get("city") or
                        address.get("town") or
                        address.get("village") or
                        address.get("municipality") or
                        address.get("country")
                )
        except Exception:
            return None
        return None

    # Applichiamo la funzione solo se LAT e LON sono validi
    df["Province"] = df.apply(
        lambda r: get_province_cached(r["LAT"], r["LON"]) if pd.notna(r["LAT"]) and pd.notna(r["LON"]) else None,
        axis=1
    )

    mappa_prodotti = {
        "B": {"nome": "Banana", "scadenza_giorni": 7},
        "M": {"nome": "Mela", "scadenza_giorni": 10},
        "A": {"nome": "Arancia", "scadenza_giorni": 14},
        "L": {"nome": "Latte", "scadenza_giorni": 5},
        "Y": {"nome": "Yogurt", "scadenza_giorni": 12},
        "P": {"nome": "Parmigiano", "scadenza_giorni": 24},
    }
    scadenze_per_prodotto = {v["nome"]: v["scadenza_giorni"] for k, v in mappa_prodotti.items()}

    def assegna_prodotto_e_scadenza(df):
        df = df.copy()
        df["Reading date"] = pd.to_datetime(df["Reading date"], dayfirst=True, errors="coerce")

        prodotti, scadenze_iniziali, scadenze_residue = [], [], []
        prima_lettura = {}

        for _, row in df.iterrows():
            qr_val = str(row.get("QR", "")).strip()
            qr_val = qr_val.replace(" ", "").upper()

            if not qr_val or qr_val == "NAN":
                prodotti.append(None)
                scadenze_iniziali.append(None)
                scadenze_residue.append(None)
                continue

            iniziale = qr_val[0]
            prodotto_info = mappa_prodotti.get(iniziale, {"nome": "Sconosciuto", "scadenza_giorni": 7})
            nome_prodotto = prodotto_info["nome"]
            giorni_scadenza = prodotto_info["scadenza_giorni"]

            data_lettura = row["Reading date"]
            if pd.isna(data_lettura):
                prodotti.append(nome_prodotto)
                scadenze_iniziali.append(None)
                scadenze_residue.append(None)
                continue

            if qr_val not in prima_lettura:
                prima_lettura[qr_val] = data_lettura
                data_scadenza = data_lettura + timedelta(days=giorni_scadenza)
                scadenza_residua = giorni_scadenza
            else:
                giorni_passati = (data_lettura - prima_lettura[qr_val]).days
                scadenza_residua = giorni_scadenza - giorni_passati
                data_scadenza = prima_lettura[qr_val] + timedelta(days=giorni_scadenza)

            prodotti.append(nome_prodotto)
            scadenze_iniziali.append(data_scadenza.date())
            scadenze_residue.append(scadenza_residua)

        df["Product"] = prodotti
        df["Expiry date (initial)"] = scadenze_iniziali
        df["Reading date"] = df["Reading date"].dt.strftime("%d/%m/%Y")
        df["Expiry date (initial)"] = pd.to_datetime(df["Expiry date (initial)"], dayfirst=True, errors="coerce").dt.strftime("%d/%m/%Y")
        df["Days left"] = scadenze_residue
        return df

    df = assegna_prodotto_e_scadenza(df)
    with st.spinner("⏳ Calcolo colonne aggiuntive in corso..."):
        df = assegna_prodotto_e_scadenza(df)
        df["Province"] = df.apply(
            lambda r: get_province_cached(r["LAT"], r["LON"]) if pd.notna(r["LAT"]) and pd.notna(r["LON"]) else None,
            axis=1
        )


    def _expiry_factor_from_days(g, initial_shelf_life):
        if pd.isna(g):
            return 0.0
        if initial_shelf_life == 0:
            return 0.0 if g < 0 else 1.0
        g = float(g)
        if g >= 0:
            return 0.6 + 0.4 * (g / initial_shelf_life)
        elif g >= -7:
            return 0.3 + (0.6 - 0.3) * ((g + 7.0) / 7.0)
        elif g >= -14:
            return 0.25 + (0.3 - 0.25) * ((g + 14.0) / 7.0)

        return max(0.0, 0.25 * math.exp((g + 14.0) / 14.0))


    def calcola_indice_freschezza(row):
        temp_diff = abs(row["Effective T"] - row["Desired T"])
        temp_factor = max(0.0, 1.0 - (temp_diff / 10.0))
        giorni_residui = row.get("Days left", 0)
        nome_prodotto = row.get("Product", "Sconosciuto")
        initial_shelf_life = scadenze_per_prodotto.get(nome_prodotto, 7)
        expiry_factor = _expiry_factor_from_days(giorni_residui, initial_shelf_life)
        combined = (temp_factor ** 0.6) * (expiry_factor ** 1.0)
        return round(100.0 * combined, 1)


    def calcola_waste_cost(df):
        prezzi_medi = {
            "Banana": 0.3,
            "Mela": 0.5,
            "Arancia": 0.6,
            "Latte": 1.2,
            "Yogurt": 1.0,
            "Sconosciuto": 0.5
        }

        prodotti_sprecati = df[df["indice_freschezza"] < 50]["Product"]
        costo_totale = sum(prezzi_medi.get(p, 0.5) for p in prodotti_sprecati)
        return round(costo_totale, 2)

    # Calcolo dell'indice di freschezza se non esiste ancora
    if "indice_freschezza" not in df.columns:
        df["indice_freschezza"] = df.apply(calcola_indice_freschezza, axis=1)

    tipo = "Car"
    fattori = {"Car": 0.12, "Truck": 0.6, "Refrigerated Truck": 0.9}
    fattore_emissione = fattori[tipo]

    # --- Calcolo metriche generali su tutti i dati ---
    total_shipments = len(df)
    compliant = len(df[df["indice_freschezza"] >= 50])
    incidenti = len(df[df["indice_freschezza"] < 50])
    waste_cost = calcola_waste_cost(df)  # esempio: 10€ per prodotto "non conforme"

    perc_compliant = (compliant / total_shipments * 100) if total_shipments > 0 else 0
    perc_incident = (incidenti / total_shipments * 100) if total_shipments > 0 else 0

    st.markdown("""
        <style>
            .snapshot-title {
                font-family: 'Segoe UI', sans-serif;
                font-size: 42px;
                font-weight: 700;
                text-align: center;
                color: #2E4053;
                letter-spacing: 1px;
                margin-top: 10px;
                margin-bottom: 30px;
            }
            div[data-testid="stMetric"] {
                background-color: #f8f9fa;
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                text-align: center;
                transition: all 0.2s ease-in-out;
            }
            div[data-testid="stMetric"]:hover {
                transform: translateY(-4px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            }
            @media (max-width: 900px) {
                div[data-testid="column"] {
                    flex: 1 1 50% !important;
                    min-width: 240px !important;
                }
            }
            @media (max-width: 600px) {
                div[data-testid="column"] {
                    flex: 1 1 100% !important;
                }
            }
        </style>
        <div class="snapshot-title">🚦 Executive Snapshot</div>
    """, unsafe_allow_html=True)

    # --- Layout colonne metriche ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ % Compliant Shipments", f"{perc_compliant:.1f}%")
    with col2:
        st.metric("⚠️ % Shipments with Incidents", f"{perc_incident:.1f}%")
    with col3:
        st.metric("📦 Total Shipments", total_shipments)
    with col4:
        st.metric("💸 Total Waste Cost (€)", f"{waste_cost:,.2f}")

    st.markdown("---")

    operatori_disponibili = sorted(df["Operator"].dropna().unique().tolist())

    operatori_selezionati = st.multiselect(
        "Filtra per operatore",
        operatori_disponibili,
        help="Seleziona uno o più operatori per filtrare la tabella principale"
    )
    if operatori_selezionati:
        filtered = df[df["Operator"].isin(operatori_selezionati)]
    else:
        filtered = df

    colonne_da_mostrare = [
        "Operator", "Reading date", "Read value", "Effective T", "QR",
        "Desired T", "Product", "indice_freschezza"
    ]

    if 'selected_qr' not in st.session_state:
        st.session_state.selected_qr = "Tutti"

    if 'df_selection' not in st.session_state:
        st.session_state.df_selection = []

    st.write("Tabella delle ultime scansioni. Clicca su una riga per vedere lo storico e la mappa di quel QR.")

    filtered_last_scan = filtered.dropna(subset=["Reading date"]) \
        .sort_values("Reading date") \
        .groupby("QR") \
        .tail(1)

    df_sorted = filtered_last_scan.copy()
    if st.session_state.selected_qr != "Tutti":
        df_sorted['_sort_key'] = (df_sorted['QR'] == st.session_state.selected_qr)
        df_sorted = df_sorted.sort_values(by='_sort_key', ascending=False).drop(columns=['_sort_key'])

    df_to_display = df_sorted[colonne_da_mostrare]

    event = st.dataframe(
        df_to_display,  # Mostra il DF ordinato e filtrato per colonne
        on_select="rerun",
        selection_mode="single-row"
    )

    new_selection = event.selection.rows
    old_selection = st.session_state.df_selection

    if new_selection:
        selected_row_index = new_selection[0]
        qr_from_click = df_sorted.iloc[selected_row_index]["QR"]
        st.session_state.selected_qr = qr_from_click

    elif not new_selection and old_selection:
        st.session_state.selected_qr = "Tutti"
    st.session_state.df_selection = new_selection

    st.subheader("📜 Scan history and details")
    unique_qr = filtered["QR"].dropna().unique().tolist()

    options = ["Tutti"] + unique_qr
    if st.session_state.selected_qr not in options:
        st.session_state.selected_qr = "Tutti"

    selected_qr = st.selectbox(
        "Seleziona QR per visualizzare solo le sue scansioni (o clicca una riga nella tabella)",
        options,
        key="selected_qr"
    )

    if selected_qr != "Tutti":
        map_data = filtered[filtered["QR"] == selected_qr]

        # --- 📜 Storico delle scansioni ---
        storico = map_data.sort_values("Reading date", ascending=True)

        st.dataframe(
            storico,
            use_container_width=True,
            hide_index=True
        )
        map_zoom = 6
    else:
        map_data = filtered
        map_zoom = 4

    # --- Mappa con Plotly ---
    if not map_data.empty and "LAT" in map_data.columns and "LON" in map_data.columns:
        map_data["indice_freschezza"] = map_data.apply(calcola_indice_freschezza, axis=1)

        fig_map = px.scatter_mapbox(
            map_data,
            lat="LAT",
            lon="LON",
            color="indice_freschezza",
            hover_data=[
                "QR",
                "Item ID",
                "Product",
                "Desired T",
                "Effective T",
                "Days left",
                "indice_freschezza",
                "Operator",
                "Province",
                "Reading date",
                "Reading hour"
            ],
            color_continuous_scale="RdYlGn",
            range_color=(0, 100),
            mapbox_style="open-street-map",
            zoom=map_zoom,
            height=550
        )

        fig_map.update_traces(marker=dict(size=15, opacity=0.85))
        fig_map.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="#f9f9f9"
        )

        config = {
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d"]
        }

        st.subheader("🗺️ Geolocation of QR Scans")
        st.plotly_chart(fig_map, use_container_width=True, config=config)

except Exception as e:
    st.error(f"Errore nel caricamento del file: {e}")

# --- Riepilogo freschezza per QR (solo temperature) ---
st.markdown("---")
st.subheader("❄️ QR Freshness Summary")

# Convertiamo le date in datetime per ordinamento
df["Reading date"] = pd.to_datetime(df["Reading date"], dayfirst=True, errors="coerce")

first_scans = df.loc[df.groupby("QR")["Reading date"].idxmin()][["QR", "Reading date"]]
first_scans = first_scans.rename(columns={"Reading date": "First Scan Date"})

# Prendiamo l'ultima scansione per ogni QR
last_scans = (
    df.sort_values("Reading date")
      .groupby("QR")
      .tail(1)
      .copy()
)

last_scans = pd.merge(last_scans, first_scans, on="QR", how="left")
# Calcolo dell'indice di freschezza
last_scans["indice_freschezza"] = last_scans.apply(calcola_indice_freschezza, axis=1)

# Aggiungiamo una valutazione qualitativa
def freshness_status(value):
    if value >= 80:
        return "🟢 Excellent"
    elif value >= 50:
        return "🟡 Moderate"
    else:
        return "🔴 Poor"

last_scans["Status"] = last_scans["indice_freschezza"].apply(freshness_status)
last_scans["First Scan Date"] = pd.to_datetime(last_scans["First Scan Date"]).dt.strftime('%d/%m/%Y')
last_scans["Expiry date (initial)"] = pd.to_datetime(last_scans["Expiry date (initial)"]).dt.strftime('%d/%m/%Y')
# 🔎 Se è stato selezionato un QR, filtriamo la tabella, altrimenti mostriamo tutti
if selected_qr != "Tutti":
    summary_data = last_scans[last_scans["QR"] == selected_qr]
    st.caption(f"Showing latest freshness data for QR **{selected_qr}**")
else:
    summary_data = last_scans
    st.caption("Showing latest freshness data for all QR codes")

# Mostriamo la tabella riassuntiva
cols = ["QR", "First Scan Date", "Expiry date (initial)", "Days left", "Desired T", "Effective T", "indice_freschezza", "Status"]
summary_data_display = summary_data[cols].rename(columns={
    "First Scan Date": "First Scan Date",
    "Expiry date (initial)": "Expiry Date",
    "Days left": "Days Left"
})

st.dataframe(
    summary_data[cols].sort_values("indice_freschezza"),
    use_container_width=True,
    hide_index=True
)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Raggio terrestre in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# --- Calcolo distanza per ogni veicolo e ripartizione CO₂ tra i QR ---
distances = []
for operator, group_op in df.groupby("Operator"):
    group_op = group_op.sort_values("Reading date")
    total_dist = 0.0
    coords = list(zip(group_op["LAT"], group_op["LON"]))

    # Calcolo distanza totale percorsa dall'operatore
    for i in range(1, len(coords)):
        if all(not pd.isna(x) for x in (*coords[i-1], *coords[i])):
            total_dist += haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])

    # --- Emissioni totali dell'operatore ---
    qrs = group_op["QR"].dropna().unique()
    distances.append({
        "Operator": operator,
        "Distanza_km": total_dist,
        "QR_list": qrs
    })

df_distance = pd.DataFrame(distances)

# --- Selettore tipo di trasporto ---
st.markdown("---")
st.subheader("🌱 CO₂ Emissions by QR (based on travelled distance)")

col1, col2, col3 = st.columns([1.5, 1.5, 1])
# 1️⃣ Filtra per operatore
operatori = sorted(df_distance["Operator"].dropna().unique().tolist())
selected_operator = col1.multiselect("👷 Operatore", operatori, help="Seleziona uno o più operatori")
# 2️⃣ Determina i QR disponibili in base all'operatore selezionato
if selected_operator:
    qrs_filtrabili = sorted([
        qr
        for op in selected_operator
        for qr_list in df_distance.loc[df_distance["Operator"] == op, "QR_list"]
        for qr in qr_list
    ])
else:
    # Nessun operatore selezionato → mostra tutti i QR
    qrs_filtrabili = sorted(df["QR"].dropna().unique().tolist())

selected_qr_filter = col2.multiselect("🔢 QR Code", qrs_filtrabili, help="Filtra per codice QR")

# 3️⃣ Filtro per intervallo di distanza
min_dist, max_dist = float(df_distance["Distanza_km"].min()), float(df_distance["Distanza_km"].max())

if min_dist < max_dist:
    selected_distance = col3.slider("🚚 Distanza (km)", min_dist, max_dist, (min_dist, max_dist))
else:
    col3.info("Filtro distanza non disponibile (dati insufficienti).")
    selected_distance = (min_dist, max_dist)

tipo = st.selectbox("Transport type", ["Car", "Truck", "Refrigerated Truck"])
fattori = {"Car": 0.12, "Truck": 0.6, "Refrigerated Truck": 0.9}
fattore_emissione = fattori[tipo]

# --- Calcolo emissioni CO₂ e ripartizione tra QR ---
expanded_rows = []
for _, row in df_distance.iterrows():
    emissioni_totali = row["Distanza_km"] * fattore_emissione
    qrs = row["QR_list"]
    if len(qrs) > 0:
        emissioni_per_qr = emissioni_totali / len(qrs)
        for qr in qrs:
            expanded_rows.append({
                "Operator": row["Operator"],
                "QR": qr,
                "Distanza_km": row["Distanza_km"],
                "Emissioni_CO2_kg": emissioni_per_qr
            })

df_emissioni = pd.DataFrame(expanded_rows)

df_emissioni_filtered = df_emissioni.copy()
if selected_operator:
    df_emissioni_filtered = df_emissioni_filtered[df_emissioni_filtered["Operator"].isin(selected_operator)]
if selected_qr_filter:
    df_emissioni_filtered = df_emissioni_filtered[df_emissioni_filtered["QR"].isin(selected_qr_filter)]
df_emissioni_filtered = df_emissioni_filtered[
    df_emissioni_filtered["Distanza_km"].between(*selected_distance)
]

# --- Visualizzazione tabella ---
st.dataframe(
    df_emissioni_filtered.sort_values("Emissioni_CO2_kg", ascending=False),
    use_container_width=True,
    hide_index=True
)

# --- Metrica totale ---
totale_co2 = df_emissioni_filtered["Emissioni_CO2_kg"].sum()
st.metric("Total estimated CO₂", f"{totale_co2:.2f} kg")

# --- Sezione finale: AI Analyst ---
ai_container = st.container()

with ai_container:
    st.markdown("---")
    st.subheader("🤖 AI Analyst")

    st.markdown(
        "Clicca **Analizza Dashboard** per generare automaticamente un report "
        "basato sui dati elaborati (freschezza, emissioni, spedizioni, operatori)."
    )

    api_key = st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        st.warning("⚠️ Nessuna API Key configurata. Inseriscila in Streamlit Secrets o direttamente nel codice.")
    else:
        client = OpenAI(api_key=api_key)

        if "report_generato" not in st.session_state:
            st.session_state.report_generato = False

        genera_report = st.button("📊 Report Dashboard")

        # Mostra spinner e genera solo al click
        if genera_report:
            with st.spinner("Analisi in corso..."):
                sintesi = {
                    "Totale spedizioni": len(df),
                    "% compliant": round(perc_compliant, 2),
                    "% incidenti": round(perc_incident, 2),
                    "Costo sprechi (€)": waste_cost,
                    "CO2 totale stimata (kg)": round(totale_co2, 2),
                    "Media indice freschezza": round(df["indice_freschezza"].mean(), 2),
                    "Operatori unici": df["Operator"].nunique(),
                    "Prodotti analizzati": df["Product"].nunique(),
                }

                prompt = f"""Sei un data analyst. Analizza i dati seguenti e genera un report completo, con osservazioni e raccomandazioni. Dati sintetici: {sintesi}
                            Rispondi in modo chiaro e strutturato, suddividendo in:
                            - Panoramica generale
                            - Punti critici o anomalie
                            - Raccomandazioni operative"""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Sei un analista esperto di logistica e qualità del trasporto alimentare."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                )

                st.session_state.report_ai = response.choices[0].message.content
                st.session_state.report_generato = True

        # Mostra report se già generato
        if st.session_state.report_generato:
            st.success("✅ Analisi completata")
            st.markdown("### 📈 Report AI")
            st.write(st.session_state.report_ai)

            st.markdown("### 💬 Chiedi altro sui dati")
            user_question = st.text_input("Scrivi una domanda (es. 'Quale prodotto ha la freschezza media più bassa?')")

            if user_question:
                with st.spinner("Elaborazione risposta..."):
                    context = df.describe(include="all").to_string()
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system",
                             "content": "Sei un assistente analitico che risponde su dati di logistica e qualità alimentare."},
                            {"role": "user", "content": f"Domanda: {user_question}\n\nContesto dati:\n{context}"}
                        ],
                        temperature=0.4,
                    )
                    st.markdown("### 🔍 Risposta AI")
                    st.write(response.choices[0].message.content)