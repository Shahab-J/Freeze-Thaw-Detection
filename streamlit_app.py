import ee
import geemap
import streamlit as st
from datetime import date
import numpy as np
from PIL import Image
from google.oauth2 import service_account
import subprocess
import sys
import urllib.request


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





import streamlit as st
import geemap
import ee
import folium
import numpy as np
import rasterio
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from geemap import foliumap
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# Initialize Earth Engine
ee.Initialize()

# Define Streamlit app title and description
st.title("Freeze-Thaw Cycle Detection Tool")
st.write("""
    This tool allows you to visualize and analyze the freeze-thaw (FT) cycles in agricultural regions of Canada using Sentinel-1 SAR data.
    You can select regions of interest (ROI), specify date ranges, and clip the results to agricultural land for freeze-thaw classification and prediction.
""")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# ROI selection using geemap
roi = st.sidebar.selectbox("Select Region of Interest (ROI)", ['Yamaska', 'Quebec', 'Other'])
if roi == 'Yamaska':
    roi_coords = [-72.75, 46.29]  # Example coordinates for Yamaska
elif roi == 'Quebec':
    roi_coords = [-71.2082, 46.8139]  # Example coordinates for Quebec City
else:
    roi_coords = st.sidebar.text_input("Enter coordinates (lat, lon)", "46.29,-72.75")

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime(2017, 10, 1))
end_date = st.sidebar.date_input("End Date", datetime(2023, 6, 30))

# Land cover clipping option
clip_to_cropland = st.sidebar.checkbox("Clip to Agricultural Land (Cropland only)", value=True)

# Define resolution selection
resolution = st.sidebar.selectbox("Select resolution", [10, 30, 100])

# Create a map with geemap
Map = geemap.Map(center=roi_coords, zoom=8)

# Add base map
Map.add_basemap("SATELLITE")

# Draw ROI on the map
roi_geometry = Map.draw_roi()
st.write(f"ROI Geometry: {roi_geometry}")

# Function to get Sentinel-1 data and process it
def get_sentinel_data(start_date, end_date, roi_geometry, resolution):
    # Define Sentinel-1 imagery collection
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(roi_geometry) \
                .filterDate(ee.Date(start_date), ee.Date(end_date)) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    # Apply geospatial resolution filter
    collection_res = collection.select('VV', 'VH').reproject(crs='EPSG:4326', scale=resolution)

    return collection_res

# Function to process FT data
def process_freeze_thaw(collection):
    # Exponential Freeze-Thaw Algorithm (EFTA)
    def efta_algorithm(image):
        # Implement your EFTA function here
        return image

    # Apply EFTA algorithm to collection
    ft_collection = collection.map(efta_algorithm)
    return ft_collection

# Fetch and process data
ft_collection = get_sentinel_data(start_date, end_date, roi_geometry, resolution)
processed_ft = process_freeze_thaw(ft_collection)

# Show processed FT results
st.write("Freeze-Thaw Detection Results:")

# Show the result on the map
Map.add_ee_layer(processed_ft, {}, "Freeze-Thaw")

# Display the map
Map.to_streamlit()

# Option to export the results
export_option = st.sidebar.checkbox("Export Results")
if export_option:
    export_format = st.sidebar.selectbox("Select Export Format", ['GeoTIFF', 'JPEG'])
    st.write(f"Exporting to {export_format}... (This is a mock-up, implement actual export logic)")
    # Implement export functionality here

# Show a sample image of the freeze-thaw results (this can be replaced with actual data)
sample_image = Image.open('sample_ft_result.jpg')
st.image(sample_image, caption='Sample Freeze-Thaw Result')

# Footer or Credits
st.write("### Credits")
st.write("This tool was developed by [Your Name], using Sentinel-1 SAR data and machine learning models. Special thanks to my supervisor and co-supervisor.")

