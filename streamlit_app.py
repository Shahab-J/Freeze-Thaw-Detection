import streamlit as st
import ee
import geemap.foliumap as geemap
from streamlit_folium import st_folium

# Initialize Earth Engine (ensure authentication is handled before this)
ee.Initialize()

st.title("Earth Engine ROI Selector")

# Create a Folium-based geemap Map and add layers
Map = geemap.Map(center=[45.0, -73.0], zoom=6)  # center on Montreal, for example
Map.add_basemap("SATELLITE")
Map.add_draw_control()  # enable drawing tools (polygon/rectangle)

st.markdown("**Draw a Region of Interest on the map:**")
# Render the map and get the output data (bidirectional)
map_data = st_folium(Map, width=700, height=500)  # uses streamlit-folium under the hood

# Check if a shape was drawn
if map_data and map_data.get("last_active_drawing"):
    # Extract the geometry of the last drawn feature (GeoJSON format)
    drawn_geojson = map_data["last_active_drawing"]["geometry"]
    st.write("**ROI GeoJSON:**", drawn_geojson)
    # Convert GeoJSON to an Earth Engine Geometry
    roi_ee = geemap.geojson_to_ee(drawn_geojson)
    # (Now use roi_ee in Earth Engine operations, e.g., filtering image collections)
    # Example: Fetch a Sentinel-1 image within the ROI
    image = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi_ee).first()
    st.write("Fetched an example image from Sentinel-1 within the ROI.")
