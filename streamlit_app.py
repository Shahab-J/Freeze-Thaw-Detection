import streamlit as st
import geemap
import ee
import numpy as np
import pandas as pd
from geemap import geemap

# Authenticate and Initialize Earth Engine API
ee.Authenticate()
ee.Initialize()

# Title
st.title('Freeze-Thaw Cycle Detection Tool')

# Sidebar options
st.sidebar.header('Select Parameters')

# User inputs for region of interest (ROI)
roi = st.sidebar.text_area("Enter Coordinates (Latitude, Longitude):", "46.29, -72.75")
coords = tuple(map(float, roi.split(',')))

# Display map
map = geemap.Map(center=coords, zoom=10)
map.add_basemap('SATELLITE')

# Add marker for the selected coordinates
map.add_marker(coords)

# Display map
st.write(map)

# Date Range Picker
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2020-12-31'))

# Landsat or Sentinel Data Selection
sensor = st.sidebar.selectbox('Select Satellite Data', ['Sentinel-1', 'Landsat'])

# Function to retrieve Sentinel-1 data for Freeze-Thaw Detection
def get_freeze_thaw_data(sensor, start_date, end_date, coords):
    if sensor == 'Sentinel-1':
        # Load Sentinel-1 data
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(ee.Geometry.Point(coords)) \
            .filterDate(start_date, end_date)
        return collection
    else:
        # Example for Landsat data
        collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA') \
            .filterBounds(ee.Geometry.Point(coords)) \
            .filterDate(start_date, end_date)
        return collection

# Displaying the freeze-thaw data on map
data = get_freeze_thaw_data(sensor, start_date, end_date, coords)

# Show the image collection
st.write('Selected Data:')
st.write(data)

# Option to Download the Result
download_option = st.sidebar.checkbox("Download Results")

if download_option:
    # Process and download the data (example of exporting processed data)
    export_data = data.mean()  # Example processing
    path = '/mnt/data/ft_data.tif'
    geemap.download_ee_image_to_local(export_data, path)
    st.write(f"Download your result [here](sandbox:/mnt/data/ft_data.tif)")

# Add additional features as required (e.g., EFTA calculation, model prediction, etc.)
