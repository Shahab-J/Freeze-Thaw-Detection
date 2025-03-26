
import ee
import geemap
import streamlit as st
import subprocess
import sys
from datetime import date
import numpy as np
from PIL import Image
import urllib.request
from google.oauth2 import service_account


import subprocess
import sys

def install_packages():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "earthengine-api==1.5.6", "Pillow==9.0.0"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")

# Check if libraries are installed, if not, install them
install_packages()






# Step 1: Check if required libraries are installed
required_libraries = [
    'geemap', 
    'earthengine-api', 
    'rasterio', 
    'streamlit', 
    'numpy', 
    'Pillow', 
    'matplotlib', 
    'folium', 
    'setuptools'
]

missing_libraries = []

for lib in required_libraries:
    try:
        __import__(lib)
        st.write(f"✅ {lib} is installed.")
    except ImportError:
        missing_libraries.append(lib)
        st.write(f"❌ {lib} is not installed.")

if missing_libraries:
    st.write("Please install the missing libraries using the following command:")
    st.code(f"pip install {' '.join(missing_libraries)}")

# Step 2: Access the Service Account JSON from Streamlit secrets
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

# Step 3: Map & User Inputs
start_date = st.date_input("Start Date", date(2023, 10, 1), min_value=date(2015, 1, 1), max_value=date(2025, 12, 31))
end_date = st.date_input("End Date", date(2024, 6, 30), min_value=date(2015, 1, 1), max_value=date(2025, 12, 31))
resolution = st.selectbox("Resolution (m)", [10, 30, 100], index=1)

# Initialize the map using geemap
Map = geemap.Map()

# Add basemap and set the region of interest
Map.add_basemap('SATELLITE')
Map.centerObject(ee.Geometry.Point([-72.75, 46.29]), 12)

# Optional: Add other map controls or layers here
Map.add_draw_control()

# Display the map using Streamlit's HTML component
st.components.v1.html(Map.to_html(), height=500)

# Step 4: Process Sentinel-1 Data
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
    # Process the images as per your logic (e.g., apply Refined Lee filtering, etc.)
    return collection

processed_images = process_sentinel1(str(start_date), str(end_date), Map.user_roi)

# Step 5: Show Classified Images
def show_classified_images(classified_images):
    """Display classified images in Streamlit."""
    image_list = classified_images.toList(classified_images.size())
    for i in range(classified_images.size().getInfo()):
        img = ee.Image(image_list.get(i))
        url = img.select('FT_State').getThumbURL({"min": 0, "max": 1, "dimensions": 512, "palette": ["blue", "red"]})
        image_array = np.array(PIL.Image.open(urllib.request.urlopen(url)))
        st.image(image_array, caption=f"Classified Image {i+1}", use_column_width=True)

# Call the function to display images
if processed_images:
    show_classified_images(processed_images)

# Step 6: Show Statistics
def summarize_statistics(classified_collection, user_roi):
    """Summary stats for freeze-thaw classification."""
    summary = []
    # Loop through images and extract statistics for frozen/thawed areas
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

# Call the function to show stats
if processed_images:
    summarize_statistics(processed_images, Map.user_roi)
