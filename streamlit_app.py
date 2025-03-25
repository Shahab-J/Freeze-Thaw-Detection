import ee
import geemap
import streamlit as st
from datetime import date
import numpy as np
from PIL import Image
from google.oauth2 import service_account

# Step 1: Access the Service Account JSON from Streamlit secrets
try:
    service_account_json = st.secrets["GEE_SERVICE_ACCOUNT_JSON"]
    credentials = service_account.Credentials.from_service_account_info(
        service_account_json, 
        scopes=["https://www.googleapis.com/auth/earthengine.readonly"]
    )
    ee.Initialize(credentials)
    st.write("✅ Earth Engine initialized successfully!")
except Exception as e:
    st.write(f"❌ Error during authentication: {e}")

# Step 2: Map & User Inputs
start_date = st.date_input("Start Date", date(2023, 10, 1), min_value=date(2015, 1, 1), max_value=date(2025, 12, 31))
end_date = st.date_input("End Date", date(2024, 6, 30), min_value=date(2015, 1, 1), max_value=date(2025, 12, 31))
resolution = st.selectbox("Resolution (m)", [10, 30, 100], index=1)

# Initialize the map using geemap
Map = geemap.Map()

# Add basemap and set the region of interest
Map.add_basemap('SATELLITE')
Map.centerObject(ee.Geometry.Point([-72.75, 46.29]), 12)

# Add drawing controls to allow the user to draw an ROI
Map.add_draw_control()

# Display the map using Streamlit's HTML component
map_html = Map.to_html()  # Save HTML content of the map
st.write("🔹 Please **draw** your ROI on the map and click **Submit**.")
st.components.v1.html(map_html, height=500)  # Use st.components.v1.html to display the map

# Step 3: Check if ROI is drawn
roi = Map.user_roi
if roi:
    st.write("✅ ROI is drawn.")
else:
    st.write("❌ No ROI drawn yet.")
