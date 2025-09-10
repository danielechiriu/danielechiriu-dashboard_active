import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Activelabel", layout="wide")
st.title("📊 Dashboard Activelabel")

# Link al file Drive (pubblico, leggibile)
file_id = "1Zp_2qP8Td1TVMdbY9mlzOHYxIlICmNZL"
url = f"https://drive.google.com/uc?id={file_id}&export=download"

try:
    # Caricamento del file
    df = pd.read_csv(url, sep=" ", header=None)
    df.columns = ["Operator", "Device", "Reading date", "Reading hour", "Read value", "Effective T", "QR", "Desired T",
                  "Item ID", "Writing date", "Writing hour", "LAT", "LON"]
    
    # Filtro per operatore
    operatore = st.text_input("Filtra per operatore (es: Ricci, Chiriu, Pinna)")
    if operatore:
        df = df[df["Operator"].str.lower() == operatore.lower()]

    st.dataframe(df)

    # Mostra mappa se ci sono coordinate valide
    if not df.empty and "LAT" in df.columns and "LON" in df.columns:
        with st.expander("📍 Mappa"):
            st.map(df[["LAT", "LON"]])

except Exception as e:
    st.error(f"Errore nel caricamento del file: {e}")


