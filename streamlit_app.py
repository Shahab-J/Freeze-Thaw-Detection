import streamlit as st
import geemap
import ee
import json
import os

# Initialize Earth Engine
def initialize_gee():
    # Fetch the GEE_SERVICE_ACCOUNT_JSON secret from Streamlit secrets
    service_account_json = st.secrets["GEE_SERVICE_ACCOUNT_JSON"]

    # Save the secret as a temporary file
    with open("service_account.json", "w") as json_file:
        json.dump(service_account_json, json_file)

    # Authenticate using the service account
    ee.Authenticate(service_account='service_account.json', authorization_code=None)
    ee.Initialize()

# Call the GEE initialization function
initialize_gee()

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

# Button to submit the selection and process data
if st.button('Submit and Process'):
    # Call your code here to process the Freeze-Thaw mapping, using the selected parameters
    st.write(f"Processing data for the following parameters:\nROI: {roi}\nDate Range: {start_date} to {end_date}\nResolution: {resolution} meters")

    # Add your code to integrate EFTA and model prediction here
    # For example, call the model or function that processes the Sentinel-1 data and uses the EFTA algorithm.
    
    # Placeholder for Freeze-Thaw map
    st.write("Generating Freeze-Thaw map... (This is a placeholder until model integration)")
    
    # Example of showing the map
    sample_image = Image.open('sample_ft_result.jpg')  # Replace with actual FT result
    st.image(sample_image, caption='Sample Freeze-Thaw Result')
    
    # Show a processed FT result map
    Map.add_ee_layer(roi_geometry, {}, "Freeze-Thaw")  # Add your processed data layer here
    Map.to_streamlit()

    # Option to export the results
    export_option = st.sidebar.checkbox("Export Results")
    if export_option:
        export_format = st.sidebar.selectbox("Select Export Format", ['GeoTIFF', 'JPEG'])
        st.write(f"Exporting to {export_format}... (This is a mock-up, implement actual export logic)")
        # Implement actual export functionality based on your model's output

# Footer or Credits
st.write("### Credits")
st.write("This tool was developed by [Your Name], using Sentinel-1 SAR data and machine learning models. Special thanks to my supervisor and co-supervisor.")
