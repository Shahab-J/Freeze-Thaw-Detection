import ee
import geemap
import streamlit as st
from datetime import date
import numpy as np
from PIL import Image
from google.oauth2 import service_account

# Step 1: Access the Service Account JSON from Streamlit secrets
try:
    # Load the service account JSON from Streamlit secrets
    service_account_json = st.secrets["GEE_SERVICE_ACCOUNT_JSON"]
    
    # Create credentials from the secrets (no file path used here)
    credentials = service_account.Credentials.from_service_account_info(
        service_account_json, 
        scopes=["https://www.googleapis.com/auth/earthengine.readonly"]
    )
    
    # Initialize Earth Engine
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
st.write("🔹 Please **draw** your ROI on the map and click **Submit**.")
st.components.v1.html(Map.to_html(), height=500)

# Step 3: Process Sentinel-1 Data
def process_sentinel1(start_date, end_date, roi):
    """Process Sentinel-1 data."""
    if roi is None:
        st.write("❌ No ROI selected. Please draw an ROI before processing.")
        return None

    selected_resolution = resolution  # User-selected resolution

    # Process Sentinel-1 data (this should be implemented using your Sentinel-1 processing code)
    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    )

    if collection.size().getInfo() == 0:
        st.write("❌ No Sentinel-1 images found for the selected date range and ROI.")
        return None

    st.write(f"🔍 Found {collection.size().getInfo()} Sentinel-1 images in ROI.")
    return collection

# Get the drawn ROI
roi = Map.user_roi

# Run the processing if ROI is selected
if roi is not None:
    processed_images = process_sentinel1(str(start_date), str(end_date), roi)

    # Continue with your code to display results
    # For example, display processed images
    if processed_images:
        # Show the processed images (implement your method for this)
        pass
else:
    st.write("❌ Please draw an ROI to proceed.")
