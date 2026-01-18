import streamlit as st
import pandas as pd
import json
import time
import os
import plotly.express as px

# --- הגדרות ---
LOG_FILE = "events_log.json"
st.set_page_config(
    page_title="Drone Command Center",
    page_icon="🦅",
    layout="wide"
)

def load_data():
    if not os.path.exists(LOG_FILE):
        return []
    
    data = []
    # פתיחת הקובץ עם מנגנון להתמודדות עם שגיאות קריאה
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue 
                
                event = json.loads(line)
                
                # מוודאים שקיבלנו מילון תקין
                if not isinstance(event, dict):
                    continue
                
                flat_event = {
                    "Time": event.get("timestamp", "N/A"),
                    "Drone ID": event.get("drone_id", "Unknown"),
                    "Type": event.get("event_type", "Unknown"),
                    "Threats": ", ".join([d.get("label", "Unknown") for d in event.get("detections", [])]),
                    "Confidence": ", ".join([f"{d.get('confidence', 0):.0%}" for d in event.get("detections", [])]),
                    "lat": event.get("location", {}).get("lat", 0),
                    "lon": event.get("location", {}).get("lon", 0)
                }
                data.append(flat_event)
            except Exception:
                # אם שורה אחת פגומה, מדלגים עליה ולא קורסים
                continue
    return data

# --- כותרת הדשבורד ---
st.title("🦅 Sentinel Drone - Command Center")
st.markdown("Real-time surveillance and threat detection dashboard")

# --- מנגנון רענון אוטומטי ---
if st.button('🔄 Refresh Data'):
    st.rerun()

# --- טעינת הנתונים ---
raw_data = load_data()

if raw_data:
    df = pd.DataFrame(raw_data)
    
    # המרת עמודת הזמן לפורמט תקין
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values(by='Time', ascending=False) # הכי חדש למעלה

    # --- מדדים עיקריים (KPIs) ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_events = len(df)
    last_event_time = df.iloc[0]['Time'].strftime("%H:%M:%S")
    unique_threats = len(df[df['Type'] == 'alert'])
    
    col1.metric("Total Events", total_events)
    col2.metric("Last Alert", last_event_time)
    col3.metric("Active Threats", unique_threats, delta_color="inverse")
    col4.metric("Drone Status", "ONLINE", delta_color="normal")

    # --- מפה וטבלה ---
    col_map, col_list = st.columns([1, 2])

    with col_map:
        st.subheader("📍 Threat Map")
        # Streamlit מחפש אוטומטית עמודות lat/lon
        st.map(df[['lat', 'lon']])

    with col_list:
        st.subheader("📋 Event Log")
        # עיצוב הטבלה: צביעת שורות של התראות
        st.dataframe(
            df[['Time', 'Drone ID', 'Threats', 'Confidence']],
            use_container_width=True,
            hide_index=True
        )

    # --- גרף התפלגות איומים ---
    st.subheader("📊 Threat Analysis")
    # ספירה פשוטה של סוגי האיומים
    all_threats = []
    for threats in df['Threats']:
        all_threats.extend(threats.split(", "))
    
    threat_counts = pd.Series(all_threats).value_counts()
    st.bar_chart(threat_counts)

else:
    st.warning("Waiting for data... Ensure the drone client is running.")
    st.info(f"Looking for log file at: {os.path.abspath(LOG_FILE)}")

# --- ריענון אוטומטי כל 2 שניות ---
time.sleep(2)
st.rerun()