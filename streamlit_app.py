import ee
import geemap
import streamlit as st
from datetime import date
import numpy as np
from PIL import Image
from google.oauth2 import service_account

import subprocess
import sys

# Check if 'earthengine-api' and 'Pillow' are installed, otherwise install them
def install_missing_libraries():
    missing_libs = []
    try:
        import earthengine_api
    except ImportError:
        missing_libs.append("earthengine-api")
    
    try:
        import PIL
    except ImportError:
        missing_libs.append("Pillow")
    
    if missing_libs:
        for lib in missing_libs:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

install_missing_libraries()

# Continue with the rest of the code
import ee
import geemap
import streamlit as st
from datetime import date
import urllib.request
import numpy as np
from PIL import Image

# Earth Engine Authentication
try:
    service_account_json = st.secrets["GEE_SERVICE_ACCOUNT_JSON"]
    credentials = ee.ServiceAccountCredentials(service_account_json["client_email"], service_account_json["private_key"])
    ee.Initialize(credentials)
    st.write("✅ Earth Engine initialized successfully!")
except Exception as e:
    st.write(f"❌ Error during authentication: {e}")

# Your rest of the code...


import subprocess

# Check if earthengine-api and Pillow are installed
subprocess.run([sys.executable, "-m", "pip", "list"])


# Step 1: Access the Service Account JSON from Streamlit secrets
try:
    service_account_json = st.secrets["GEE_SERVICE_ACCOUNT_JSON"]
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

# Initialize map using geemap
Map = geemap.Map()

# Add basemap and set the region of interest
Map.add_basemap('SATELLITE')
Map.centerObject(ee.Geometry.Point([-72.75, 46.29]), 12)

# Optional: Add other map controls or layers here
Map.add_draw_control()

# Display map
st.components.v1.html(Map.to_html(), height=500)

# Step 3: Process Sentinel-1 Data
def process_sentinel1(start_date, end_date, roi):
    if roi is None:
        st.write("❌ No ROI selected. Please draw an ROI before processing.")
        return None

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

# Step 4: Show Classified Images
def show_classified_images(classified_images):
    image_list = classified_images.toList(classified_images.size())
    for i in range(classified_images.size().getInfo()):
        img = ee.Image(image_list.get(i))
        url = img.select('FT_State').getThumbURL({"min": 0, "max": 1, "dimensions": 512, "palette": ["blue", "red"]})
        image_array = np.array(PIL.Image.open(urllib.request.urlopen(url)))
        st.image(image_array, caption=f"Classified Image {i+1}", use_column_width=True)

# Step 5: Statistics
def summarize_statistics(classified_collection, user_roi):
    summary = []
    for i in range(classified_collection.size().getInfo()):
        img = ee.Image(classified_collection.toList(classified_collection.size()).get(i))
        stats = img.select("FT_State").reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=user_roi,
            scale=resolution,
            maxPixels=1e13
        ).getInfo()

        hist = stats.get("FT_State", {})
        thawed_count = int(hist.get("0", 0))
        frozen_count = int(hist.get("1", 0))
        total_count = thawed_count + frozen_count
        thawed_percent = (thawed_count / total_count * 100) if total_count > 0 else 0
        frozen_percent = (frozen_count / total_count * 100) if total_count > 0 else 0

        summary.append(f"Image {i+1}: Frozen={frozen_count} ({frozen_percent:.1f}%) | Thawed={thawed_count} ({thawed_percent:.1f}%)")

    st.write("\n".join(summary))

# Final check if processed images are available
processed_images = process_sentinel1(str(start_date), str(end_date), Map.user_roi)

if processed_images:
    show_classified_images(processed_images)
    summarize_statistics(processed_images, Map.user_roi)
