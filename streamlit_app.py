# ✅ Step 0: Must be FIRST

import streamlit as st
st.set_page_config(layout="wide")

import ee
import io
import sys
import math
import json
import time
import geemap
import folium
import subprocess
import numpy as np
import urllib.request
from PIL import Image
from datetime import date
import ipywidgets as widgets
from folium.plugins import Draw
import matplotlib.pyplot as plt
import geemap.foliumap as geemap
from google.auth import credentials
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from streamlit_folium import st_folium
from google.oauth2 import service_account
from streamlit_folium import folium_static
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut



# ========== ✅ ❄️ Background with snow ===================
def inject_clean_background(image_):
    st.markdown(f"""
        <style>
        /* Top white strip */
        .top-white {{
            position: fixed;
            top: 0;
            left: 0;
            height: 0.5cm;
            width: 100%;
            background: white;
            z-index: 9998;
        }}

        /* Background image */
        [data-testid="stAppViewContainer"] {{
            background-image: ("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center top;
        }}
        </style>

        <div class="top-white"></div>
    """, unsafe_allow_html=True)

inject_clean_background("https://raw.githubusercontent.com/Shahab-J/Freeze-Thaw-Detection/main/assets/20201215_155514.jpg")





# ✅ Step 1: Setup
# ========== ✅ Title and Setup ===================
st.title("🧊 Soil Freeze–Thaw Mapping Tool")


# ========== ✅ Authenticate Earth Engine ========== 
try:
    service_account = st.secrets["GEE_SERVICE_ACCOUNT"]
    private_key = st.secrets["GEE_PRIVATE_KEY"]
    
    # Creating the credentials for Earth Engine
    credentials = ee.ServiceAccountCredentials(
        service_account,
        key_data=json.dumps({
            "type": "service_account",
            "client_email": service_account,
            "private_key": private_key,
            "token_uri": "https://oauth2.googleapis.com/token"
        })
    )
    
    # Initialize Earth Engine
    ee.Initialize(credentials)
#   st.success("✅ Earth Engine initialized successfully.")  # Success message
    
except Exception as e:
    st.error(f"❌ EE Auth failed: {e}")
    st.stop()  # Stop execution if authentication fails



# ========== ✅ Set up map with default satellite view ==========
st.subheader("Draw your ROI below")
st.markdown(
    "<p style='font-size: 12px;'> (choose 'Satellite' or 'OpenStreetMap' for map view using the Layer Switcher in the top right of the map)</p>", 
    unsafe_allow_html=True
)

# ========== ✅ Sidebar UI ==========
with st.sidebar.expander("📘 How to Use the Tool", expanded=False):
    st.markdown("""
        <div style="font-size:12pt; font-weight:bold; margin-top: 10px;">Step 1: Search for a Location (Optional)</div>
        <div style="font-size:11pt;">
        Use the <i>search box</i> to type any city or landmark (e.g., <b>Montreal</b>).<br>
        The map will zoom to that location to help you easily draw your ROI.
        </div>

        <div style="font-size:12pt; font-weight:bold; margin-top: 10px;">Step 2: Draw Your ROI</div>
        <div style="font-size:11pt;">
        Use the tools on the <b>left side of the map</b> — rectangle or polygon is recommended.<br>
        You can draw your Region of Interest anywhere in Canada.<br>
        The map supports zooming, dragging, and switching basemaps.
        </div>

        <div style="font-size:12pt; font-weight:bold; margin-top: 10px;">Step 3: Set Parameters</div>
        <div style="font-size:11pt;">
        Choose a <b>start and end date</b> between <b>October 1 and June 30</b>.<br>
        We recommend selecting a full freeze–thaw season (e.g., Oct 1 to June 30 of the next year).<br>
        Even if your range is shorter (e.g., Oct to Nov), the app internally adjusts the date range but shows results only for your selection.<br><br>
        Select a <b>spatial resolution</b>: 10 m (high detail), 30 m (default), or 100 m (faster).<br>
        Optionally, use the checkbox to <b>Clip ROI to cropland, grasslands, barren lands</b> based on the 2020 NALCMS dataset.
        </div>

        <div style="font-size:12pt; font-weight:bold; margin-top: 10px;">Step 4: Click “Submit ROI & Start Processing”</div>
        <div style="font-size:11pt;">
        This triggers the full workflow: ROI validation, date adjustments, Sentinel-1 loading, metric computation, and freeze–thaw classification.<br><br>
        <b>Important!</b> After clicking the button, a message appears:<br>
        <i>“Please wait. Do not zoom or tap on the map… Scroll down to view results.”</i>
        </div>

        <div style="font-size:12pt; font-weight:bold; margin-top: 10px;">Step 5: View the Results</div>
        <div style="font-size:11pt;">
        A dropdown will appear labeled “View All Freeze–Thaw Results”.<br>
        Scroll down to view the images in chronological order.<br>
        Each classified panel shows thawed areas in <b>red</b> and frozen in <b>blue</b>, along with pixel counts and percentages.<br><br>
        <b>Download Option:</b> You can download high-resolution TIFFs for analysis or archiving.
        </div>
    """, unsafe_allow_html=True)



                                                                      
st.sidebar.title("Set Parameters")
def_start = date(2023, 10, 1)
def_end = date(2024, 6, 30)

# Initialize the resolution key if not already in session state
if 'resolution' not in st.session_state:
    st.session_state.resolution = 30  # Default resolution is 30 meters

# Sidebar inputs for start date, end date, resolution, and clipping option
start_date = st.sidebar.date_input("Start Date", value=def_start)
end_date = st.sidebar.date_input("End Date", value=def_end)

# Ensure the resolution is selected correctly from session state
resolution = st.sidebar.selectbox(
    "Resolution (meters)", 
    [10, 30, 100], 
    index=[10, 30, 100].index(st.session_state.resolution)  # Set the correct index based on session state
)

# Update session state with the selected resolution
st.session_state.resolution = resolution

clip_to_agri = st.sidebar.checkbox("🌾 Clip ROI to cropland, grasslands, and barren lands", value=True)
submit = st.sidebar.button("🚀 Submit ROI & Start Processing")

# ========== ✅ Session State Initialization ==========
if 'start_date' not in st.session_state:
    st.session_state.start_date = start_date
if 'end_date' not in st.session_state:
    st.session_state.end_date = end_date

# ========== ✅ Geocoding Search Bar ==========
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
from geopy.geocoders import Nominatim
import time

# Initialize the geolocator object
geolocator = Nominatim(user_agent="streamlit_app")

def search_location_with_retry(place, retries=3, delay=10):
    """Search for a location with retry logic in case of failure."""
    attempt = 0
    while attempt < retries:
        try:
            location = geolocator.geocode(place, timeout=60)  # Increased timeout
            if location:
                return [location.latitude, location.longitude]
            else:
                st.warning("Location not found. Please check the place name and try again.")
                return None
        except (GeocoderUnavailable, GeocoderTimedOut) as e:
            st.warning(f"Geocoding service is unavailable. Retrying... ({attempt + 1}/{retries})")
            attempt += 1
            time.sleep(delay)  # Wait for some time before retrying
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    st.error("Failed to geocode after multiple attempts. Please try searching for the place again later.")
    return None



def add_search_bar(map_object):
    # Search function to get coordinates from the place name using Nominatim
    def search_location(place):
        location = search_location_with_retry(place)
        if location:
            map_object.location = location
            map_object.zoom_start = 12  # Zoom level

            # Update user_roi based on the geocoded location
            user_roi = ee.Geometry.Point(location)  # Re-define the ROI based on new map location
            st.session_state.user_roi = user_roi  # Update session state with new ROI

            # Add a marker on the map for the location
            folium.Marker(location, popup=place).add_to(map_object)
        else:
            st.warning("Please try searching for the place again later.")

    place = st.text_input("Enter place (city, landmark, etc.):")
    if place:
        search_location(place)


# ========== ✅ Map Setup ==========
# Create the map centered at a location
m = folium.Map(location=[46.29, -72.75], zoom_start=12, control_scale=True)

# Add Satellite basemap (default)
satellite_tile = folium.TileLayer(
    tiles="Esri.WorldImagery", attr="Esri", name="Satellite", overlay=False, control=True
).add_to(m)

# Add Layer control to switch between Satellite and OpenStreetMap
folium.LayerControl(position="topright").add_to(m)

# Add drawing control to the map
draw = Draw(export=False)
draw.add_to(m)

# Add the search bar (the user input field for place search)
add_search_bar(m)

# ========== ✅ Render the map once with the updated location ==========
output = st_folium(m, width=1300, height=450)  # Display map with updated location



# ✅ Step 2: Sentinel-1 Processing for Streamlit
def process_sentinel1(start_date, end_date, roi, resolution):
    """Loads and processes Sentinel-1 data for the selected ROI and time range."""
    if roi is None:
        st.error("❌ No ROI selected. Please draw an ROI before processing.")
        return None

    # Convert ROI to Earth Engine Geometry
    roi = ee.Geometry(roi)

    # Filter Sentinel-1 Collection
    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    )

    # Use client-side safe method to count image availability
    image_count = collection.size()
    image_count_val = image_count.getInfo()

    if image_count_val == 0:
        st.error("❌ No Sentinel-1 images found in the selected date range and ROI.")
        return None

    # ✅ Refined Lee Filter Function
    def RefinedLee(img):
        img_natural = ee.Image(10.0).pow(img.select('VH').divide(10.0))
        kernel = ee.Kernel.square(7)
        mean = img_natural.reduceNeighborhood(ee.Reducer.mean(), kernel)
        variance = img_natural.reduceNeighborhood(ee.Reducer.variance(), kernel)
        sample_stats = variance.divide(mean.multiply(mean))
        b = sample_stats.divide(sample_stats.add(1.0))
        refined = mean.add(b.multiply(img_natural.subtract(mean))).float()
        refined_db = refined.log10().multiply(10.0).rename('refined_lee')
        return img.addBands(refined_db)

    # ✅ Terrain Normalization Function
    def normalizeS1(image):
        srtm = ee.Image('USGS/SRTMGL1_003')
        cos2thetaref = ee.Number(40).divide(180).multiply(math.pi).cos().pow(2)
        cos2theta = image.select('angle').divide(180).multiply(math.pi).cos().pow(2)
        vh_corrected = image.select('refined_lee').multiply(cos2thetaref).divide(cos2theta).rename('VH_corrected')
        return image.addBands(vh_corrected)

    # ✅ Apply filtering and correction
    processed_collection = (
        collection
        .map(RefinedLee)
        .map(normalizeS1)
        .map(lambda img: img.reproject(crs="EPSG:4326", scale=resolution))  # Reproject to selected resolution
    )

    return processed_collection


# ✅ Step 3: Mosaicking by Date for Streamlit
def mosaic_by_date(collection, roi, start_date, end_date):
    """Mosaics Sentinel-1 images captured on the same date to avoid duplicate acquisitions."""
    if collection is None:
        st.error("❌ ERROR: No processed images available for mosaicking.")
        return None

    filtered_collection = collection.filterDate(start_date, end_date)

    count = filtered_collection.size().getInfo()
    if count == 0:
        st.error("❌ ERROR: No images found after filtering for mosaicking.")
        return None

    # Extract unique dates as strings
    unique_dates = (
        filtered_collection.aggregate_array('system:time_start')
        .map(lambda t: ee.Date(t).format('YYYY-MM-dd'))
        .distinct()
    )

    def mosaic_same_day(date_str):
        date = ee.Date.parse('YYYY-MM-dd', date_str)
        mosaic = (
            filtered_collection
            .filterDate(date, date.advance(1, 'day'))
            .mosaic()
            .clip(roi)
            .set('system:time_start', date.millis())
        )
        return mosaic

    mosaicked_collection = ee.ImageCollection(unique_dates.map(mosaic_same_day))

    mosaicked_count = mosaicked_collection.size().getInfo()
    if mosaicked_count == 0:
        st.error("❌ ERROR: No mosaicked images generated.")
        return None

    return mosaicked_collection


# ✅ Step 4: SigmaDiff Computation for Streamlit
def compute_sigma_diff_pixelwise(collection):
    """Computes SigmaDiff as the pixel-wise difference in VH_corrected between consecutive images."""
    if collection is None:
        st.error("❌ ERROR: No mosaicked images available for SigmaDiff computation.")
        return None

    # Sort by time
    sorted = collection.sort("system:time_start")
    image_list = sorted.toList(sorted.size())
    collection_size = sorted.size().getInfo()

    if collection_size < 2:
        st.warning("⚠️ Not enough images to compute SigmaDiff.")
        return None

    updated_images = []

    # Initialize history with first image
    first_img = ee.Image(image_list.get(0))
    history = first_img.select('VH_corrected')
    first_with_diff = first_img.addBands(ee.Image.constant(0).float().rename('SigmaDiff'))
    updated_images.append(first_with_diff)

    for i in range(1, collection_size):
        current = ee.Image(image_list.get(i))
        current_vh = current.select('VH_corrected')
        sigma_diff = current_vh.subtract(history).rename('SigmaDiff')
        current_with_diff = current.addBands(sigma_diff)
        updated_images.append(current_with_diff)
        history = current_vh  # Update history for next iteration

    result_collection = ee.ImageCollection.fromImages(updated_images)
    return result_collection


# ✅ Step 5: SigmaDiff Min/Max Computation for Streamlit
def compute_sigma_diff_extremes(collection, start_year, user_roi):
    """
    Computes SigmaDiff_min and SigmaDiff_max dynamically per pixel based on two seasonal periods:
    - Min from mid-October to end of January
    - Max from late February to late May
    """
    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ ERROR: No valid SigmaDiff images found. Cannot compute extremes.")
        return None

    # Define seasonal windows for min and max
    mid_oct_to_end_jan = collection.filterDate(f'{start_year}-10-15', f'{start_year+1}-01-31')
    end_feb_to_may = collection.filterDate(f'{start_year+1}-02-20', f'{start_year+1}-05-20')

    # Reduce to min/max SigmaDiff in the two periods
    sigma_min = (
        mid_oct_to_end_jan.select('SigmaDiff')
        .reduce(ee.Reducer.min())
        .rename('SigmaDiff_min')
        .clip(user_roi)
    )

    sigma_max = (
        end_feb_to_may.select('SigmaDiff')
        .reduce(ee.Reducer.max())
        .rename('SigmaDiff_max')
        .clip(user_roi)
    )

    # Attach constant min/max bands to each image
    def attach_min_max(img):
        return img.addBands(sigma_min).addBands(sigma_max)

    updated_collection = collection.map(attach_min_max)

#   st.success("✅ SigmaDiff Min/Max computation complete.")
    return updated_collection


# ✅ Step 6: Freeze–Thaw K Assignment for Streamlit
def assign_freeze_thaw_k(collection):
    """
    Assigns a 'K' band to each image in the collection:
    - K = 0 → Freeze Start
    - K = 1 → Thaw Start
    - Continues using the last value if not close to min/max
    """
    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ ERROR: No collection provided for K assignment.")
        return None

    # Sort collection by date
    sorted_collection = collection.sort('system:time_start')
    image_list = sorted_collection.toList(sorted_collection.size())

    # Initialize first image with K = 1 (thaw)
    first_image = ee.Image(image_list.get(0)).addBands(
        ee.Image.constant(1).byte().rename('K')
    )

    freeze_tracker = first_image.select('K')
    updated_images = [first_image]

    collection_size = image_list.size().getInfo()

    # Loop through and assign K values
    for i in range(1, collection_size):
        current_img = ee.Image(image_list.get(i))
        sigma_diff = current_img.select('SigmaDiff')
        sigma_min = current_img.select('SigmaDiff_min')
        sigma_max = current_img.select('SigmaDiff_max')

        # Set tolerance for freeze/thaw threshold detection
        tolerance = ee.Image(0.01)

        freeze_start = sigma_diff.subtract(sigma_min).abs().lt(tolerance)
        thaw_start = sigma_diff.subtract(sigma_max).abs().lt(tolerance)

        k = freeze_tracker.where(freeze_start, 0).where(thaw_start, 1).byte()
        freeze_tracker = k  # Update tracker for next step

        k_masked = k.updateMask(sigma_diff.mask())  # Preserve valid pixels only
        updated_img = current_img.addBands(k_masked.rename('K'))
        updated_images.append(updated_img)

    final_collection = ee.ImageCollection.fromImages(updated_images)
#   st.success("✅ Freeze–Thaw K Assignment complete.")
    return final_collection


# ✅ Step 7: ThawRef Calculation for Streamlit
def compute_thaw_ref_pixelwise(collection, start_year, user_roi):
    """
    Computes the ThawRef image for each pixel as the average of the top 3 VH_corrected values
    during Fall and Spring periods.

    Args:
        collection (ee.ImageCollection): Must contain 'VH_corrected' band.
        start_year (int): Processing start year.
        user_roi (ee.Geometry): Region of interest selected by the user.

    Returns:
        ee.Image: An image with the 'ThawRef' band.
    """

    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ ERROR: Input collection is empty or undefined.")
        return None

    # Define date windows
    fall_start = f'{start_year}-10-01'
    fall_end = f'{start_year}-11-30'
    spring_start = f'{start_year+1}-04-15'
    spring_end = f'{start_year+1}-06-10'

    # Filter images
    fall_collection = collection.filterDate(fall_start, fall_end)
    spring_collection = collection.filterDate(spring_start, spring_end)

    combined_collection = fall_collection.merge(spring_collection)

    # Ensure collection is not empty
    if combined_collection.size().getInfo() == 0:
        st.error("❌ No images found in Fall and Spring periods for ThawRef.")
        return None

    # Clip images and sort by VH_corrected descending
    combined_clipped = combined_collection.map(lambda img: img.clip(user_roi))
    sorted_by_vh = combined_clipped.sort('VH_corrected', False)

    # Limit to top 3 and compute mean
    top3 = sorted_by_vh.limit(3)
    thaw_ref = top3.mean().select('VH_corrected').rename('ThawRef')

#   st.success("✅ ThawRef Calculation complete.")
    return thaw_ref


# ✅ Step 8: DeltaTheta (ΔΘ) Calculation for Streamlit
def compute_delta_theta(collection, thaw_ref_image):
    """
    Computes ΔΘ (DeltaTheta) as ThawRef - VH_corrected for each image.

    Args:
        collection (ee.ImageCollection): Must contain the 'VH_corrected' band.
        thaw_ref_image (ee.Image): ThawRef image to be subtracted from each VH_corrected.

    Returns:
        ee.ImageCollection: With 'DeltaTheta' band added to each image.
    """

    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ ERROR: No input images to compute DeltaTheta.")
        return None

    if thaw_ref_image is None:
        st.error("❌ ERROR: ThawRef image is not available.")
        return None

    def add_delta_theta(img):
        vh_corrected = img.select('VH_corrected')

        # ✅ Compute ΔΘ = ThawRef - VH_corrected
        delta_theta = thaw_ref_image.subtract(vh_corrected).rename('DeltaTheta')

        # ✅ Mask DeltaTheta where VH_corrected is masked
        delta_theta = delta_theta.updateMask(vh_corrected.mask())

        return img.addBands(delta_theta)

    # Map the function over the image collection
    updated_collection = collection.map(add_delta_theta)

 #  st.success("✅ DeltaTheta Calculation complete.")
    return updated_collection



# ✅ Step 9: EFTA Calculation for Streamlit
def compute_efta(collection, resolution):
    """
    Computes EFTA dynamically using the exponential freeze-thaw algorithm.

    Args:
        collection (ee.ImageCollection): ImageCollection with bands:
            'K', 'ThawRef', 'VH_corrected', and 'DeltaTheta'.
        resolution (int): Desired spatial resolution for reprojecting output.

    Returns:
        ee.ImageCollection: With 'EFTA' band added to each image.
    """

    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ ERROR: Input collection is empty. Cannot compute EFTA.")
        return None

    def calculate_efta(img):
        k = img.select('K')
        thaw_ref = img.select('ThawRef')
        vh_corrected = img.select('VH_corrected')
        delta_theta = img.select('DeltaTheta')

        # ✅ Compute the Exponential Component
        exp_component = (ee.Image(1)
                         .add(thaw_ref.divide(vh_corrected))
                         .multiply(k.multiply(-1))
                         .exp())

        # ✅ Final EFTA = exp_component × DeltaTheta
        efta = exp_component.multiply(delta_theta).rename('EFTA')

        # ✅ Mask EFTA where VH_corrected is invalid
        efta = efta.updateMask(vh_corrected.mask())

        # ✅ Add EFTA band and reproject
        return img.addBands(efta).reproject(crs="EPSG:4326", scale=resolution)

    # Apply calculation to each image
    updated_collection = collection.map(calculate_efta)

#   st.success("✅ EFTA Calculation complete.")
    return updated_collection




# =========================================================
# ✅ ERA5 DAILY SNOW DEPTH + SNOW TEMPERATURE (Optimized)
# =========================================================
def build_era5_snow_collection(start_date, end_date, roi):
    """
    Efficient daily ERA5 snow and snow temperature extraction.
    Returns bands:
        - Snow_depth  (cm)
        - Snow_temp   (°C)
    """
    roi = ee.Geometry(roi)
    res = st.session_state.resolution

    # Load ERA5-Land hourly dataset (corrected: include filterDate)
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(start_date, end_date)          # ✅ IMPORTANT
        .filterBounds(roi)
        .select([
            "snow_depth",                     # meters
            "temperature_of_snow_layer"       # Kelvin
        ])
    )

    # Generate daily list
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n_days = end.difference(start, "day")

    def make_daily_image(day_offset):
        day_offset = ee.Number(day_offset)
        day = start.advance(day_offset, "day")

        # Daily mean ERA5
        daily = era5.filterDate(day, day.advance(1, "day")).mean()

        snow_depth_cm = daily.select("snow_depth") \
                             .multiply(100) \
                             .rename("Snow_depth")

        snow_temp_C = daily.select("temperature_of_snow_layer") \
                           .subtract(273.15) \
                           .rename("Snow_temp")

        return (
            snow_depth_cm.addBands(snow_temp_C)
            .set("system:time_start", day.millis())
            .clip(roi)
            .reproject(crs="EPSG:4326", scale=res)
        )

    return ee.ImageCollection.fromImages(
        ee.List.sequence(0, n_days.subtract(1)).map(make_daily_image)
    )

    
def attach_era5_to_efta(efta_collection, start_date, end_date, roi):
    """
    Joins daily ERA5 snow predictors to EFTA images by timestamp.
    Returns EFTA + Snow_depth + Snow_temp.
    """
    roi = ee.Geometry(roi)

    era5_daily = build_era5_snow_collection(start_date, end_date, roi)

    join_filter = ee.Filter.equals(
        leftField="system:time_start",
        rightField="system:time_start"
    )

    inner_join = ee.Join.inner().apply(efta_collection, era5_daily, join_filter)

    def merge(pair):
        return ee.Image(pair.get("primary")).addBands(
            ee.Image(pair.get("secondary"))
        )

    return ee.ImageCollection(inner_join.map(merge))






# =========================================================
# ✅ Step 10 — Freeze–Thaw Classification Using RF (Streamlit)
# =========================================================

# ========== TRAINING DATASET (NO GEOMETRY NEEDED) ==========
url = "https://raw.githubusercontent.com/Shahab-J/Freeze-Thaw-Detection/main/data/RF_FT_Snow_EFTA_training.csv"

import pandas as pd
df_train = pd.read_csv(url)

# Convert each row to an ee.Feature WITHOUT geometry
def row_to_feature(row):
    props = {
        "EFTA": float(row["EFTA"]),
        "Snow_depth": float(row["Snow_depth"]),
        "Snow_temp": float(row["Snow_temp"]),
        "label": int(row["label"])
    }
    return ee.Feature(None, props)

training_features = df_train.apply(row_to_feature, axis=1).tolist()
training_asset = ee.FeatureCollection(training_features)

# Predictors and label
bands = ['EFTA', 'Snow_depth', 'Snow_temp']
label = 'label'



# Random Forest Model
def train_rf_model():
    try:
        rf_model = ee.Classifier.smileRandomForest(
            numberOfTrees=150,
            variablesPerSplit=1,
            minLeafPopulation=3,
            seed=42
        ).train(
            features=training_asset,
            classProperty=label,
            inputProperties=bands
        )
        return rf_model

    except Exception as e:
        st.error(f"❌ Failed to train RF model: {e}")
        return None




# ✅ Classify each image using the trained model
def classify_image(img, rf_model, resolution):
    """
    Classifies an image using the trained Random Forest model.

    Args:
        img (ee.Image): Input image with EFTA + Snow_depth + Snow_temp
        rf_model (ee.Classifier): Trained RF classifier
        resolution (int): Resolution to reproject classified results

    Returns:
        ee.Image: Image with an added 'FT_State' classification band
    """

    # IMPORTANT: Select all three predictors in correct order
    predictors = img.select(['EFTA', 'Snow_depth', 'Snow_temp'])

    classified = predictors.classify(rf_model).rename('FT_State')

    return img.addBands(classified).reproject(crs="EPSG:4326", scale=resolution)




# ============================================================
# ✅ Step 11: Compute and Summarize FT Classification (UPDATED)
# ============================================================

def summarize_ft_classification(collection, user_roi, resolution):
    """
    Summarizes freeze–thaw classification (0 = Thawed, 1 = Frozen)
    for each image inside the user's ROI.
    """

    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ No classified images available for summarization.")
        return

    st.markdown("## 📊 Freeze–Thaw Classification Summary")

    img_list = collection.toList(collection.size())
    n = collection.size().getInfo()

    for i in range(n):

        try:
            img = ee.Image(img_list.get(i))

            # Extract date
            date_str = img.date().format("YYYY-MM-dd").getInfo()

            # Compute histogram of FT_State
            stats = img.select("FT_State").reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=user_roi,
                scale=resolution,
                maxPixels=1e13
            ).getInfo()

            hist = stats.get("FT_State", {})

            # Retrieve counts safely (avoid KeyError)
            thawed = int(hist.get("0", 0))     # label 0 → thawed
            frozen = int(hist.get("1", 0))     # label 1 → frozen
            total = thawed + frozen

            if total == 0:
                thawed_pct = frozen_pct = 0
            else:
                thawed_pct = thawed / total * 100
                frozen_pct = frozen / total * 100

            # Display result
            st.markdown(f"""
### 🗓️ Image {i+1} — **{date_str}**
- **Thawed (0):** {thawed:,} pixels  —  {thawed_pct:.2f}%
- **Frozen (1):** {frozen:,} pixels  —  {frozen_pct:.2f}%
---
""")

        except Exception as e:
            st.warning(f"⚠️ Could not summarize image {i+1}: {e}")

    st.success("✅ Freeze–Thaw Classification Summary Computed.")



# ======================================================
# ✅ Step 12: Visualize FT Classification for Streamlit
# ======================================================
def visualize_ft_classification(collection, user_roi, resolution):
    import tempfile
    import base64
    from PIL import Image
    import urllib.request

    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ No classification results available for visualization.")
        return

    # Convert ImageCollection to a list
    image_list = collection.toList(collection.size())
    num_images = collection.size().getInfo()

    # Get user-selected dates for display
    start_date_str = st.session_state.start_date.strftime("%Y-%m-%d")
    end_date_str = st.session_state.end_date.strftime("%Y-%m-%d")

    # UI
    st.markdown("🔽 Open the dropdown below to view all classified images.")
    with st.expander("🧊 View All Freeze–Thaw Results", expanded=False):

        # Total image count
        st.markdown(
            f"""
            🖼️ Total FT Classified Images  
            from <u>{start_date_str}</u> to <u>{end_date_str}</u>:  
            <b><span style='font-size: 26px'>{num_images}</span></b>
            """,
            unsafe_allow_html=True
        )

        # Loop through each image
        for i in range(num_images):
            try:
                img = ee.Image(image_list.get(i))
                timestamp = img.date().format("YYYY-MM-dd").getInfo()

                # Thumbnail URL for FT_State band
                url = img.select("FT_State").clip(user_roi).getThumbURL({
                    "min": 0,
                    "max": 1,
                    "dimensions": 768,
                    "palette": ["red", "blue"]
                })

                # Display image
                image = Image.open(urllib.request.urlopen(url))
                st.image(image, caption=f"🗓️ {timestamp}", use_container_width=True)

                # Offer TIFF download
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                    image.save(tmp.name)
                    with open(tmp.name, "rb") as file:
                        b64 = base64.b64encode(file.read()).decode()
                        href = (
                            f'<a href="data:file/tif;base64,{b64}" '
                            f'download="FT_{timestamp}.tif">📥 Download TIFF</a>'
                        )
                        st.markdown(href, unsafe_allow_html=True)

                # Pixel statistics
                stats = img.select("FT_State").reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=user_roi,
                    scale=resolution,
                    maxPixels=1e13
                ).getInfo()

                hist = stats.get("FT_State", {})

                thawed = int(hist.get("0", 0))
                frozen = int(hist.get("1", 0))
                total = thawed + frozen

                thawed_pct = (thawed / total * 100) if total > 0 else 0
                frozen_pct = (frozen / total * 100) if total > 0 else 0

                # Printed summary
                st.markdown(
                    f"""
                    **🧊 Freeze–Thaw Stats — {timestamp}**
                    - 🟦 Frozen: **{frozen:,}** ({frozen_pct:.1f}%)
                    - 🟥 Thawed: **{thawed:,}** ({thawed_pct:.1f}%)
                    - 📏 Resolution: **{resolution} m**
                    """
                )

                st.divider()

            except Exception as e:
                st.warning(f"⚠️ Error displaying image {i+1}: {e}")



# ========== ✅ Step 13: Submit ROI and Processing Pipeline ==========
def submit_roi():
    # Ensure ROI is selected
    if "user_roi" not in st.session_state or st.session_state.user_roi is None:
        st.error("❌ No ROI selected. Please draw an ROI before processing.")
        return

    user_roi = st.session_state.user_roi
    resolution = st.session_state.get("resolution", 30)
    clip_agriculture = st.session_state.get("clip_to_agriculture", False)

    # User dates
    user_selected_start = st.session_state.start_date.strftime("%Y-%m-%d")
    user_selected_end = st.session_state.end_date.strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")

    # Validate date range
    if user_selected_end >= today:
        st.error(f"❌ End date ({user_selected_end}) is in the future. Please select a valid range.")
        return
    if user_selected_start >= user_selected_end:
        st.error("❌ Start date must be earlier than end date.")
        return

    # Adjust season (Oct → June window)
    start_year = int(user_selected_start[:4])
    if int(user_selected_start[5:7]) < 10:
        start_year -= 1

    start_date = f"{start_year}-10-01"
    end_date = f"{start_year+1}-06-30"

    with st.spinner("⏳ Running full Freeze–Thaw processing pipeline..."):

        # 1) Sentinel-1 Processing
        processed_images = process_sentinel1(start_date, end_date, user_roi, resolution)
        if processed_images is None:
            st.warning("⚠️ Sentinel-1 processing failed.")
            return

        # 2) Daily Mosaic
        mosaicked_images = mosaic_by_date(processed_images, user_roi, start_date, end_date)
        if mosaicked_images is None:
            st.warning("⚠️ Mosaicking failed.")
            return

        # 3) SigmaDiff pixelwise
        sigma_diff_collection = compute_sigma_diff_pixelwise(mosaicked_images)
        if sigma_diff_collection is None:
            st.warning("⚠️ SigmaDiff computation failed.")
            return

        # 4) Seasonal SigmaDiff extremes
        sigma_extreme_collection = compute_sigma_diff_extremes(
            sigma_diff_collection, start_year, user_roi
        )
        if sigma_extreme_collection is None:
            st.warning("⚠️ SigmaDiff extremes failed.")
            return

        # 5) Freeze–Thaw K assignment
        final_k_collection = assign_freeze_thaw_k(sigma_extreme_collection)
        if final_k_collection is None:
            st.warning("⚠️ K assignment failed.")
            return

        # 6) ThawRef image
        thaw_ref_image = compute_thaw_ref_pixelwise(final_k_collection, start_year, user_roi)
        if thaw_ref_image is None:
            st.warning("⚠️ ThawRef computation failed.")
            return

        thaw_ref_collection = final_k_collection.map(lambda img: img.addBands(thaw_ref_image))

        # 7) DeltaTheta
        delta_theta_collection = compute_delta_theta(thaw_ref_collection, thaw_ref_image)
        if delta_theta_collection is None:
            st.warning("⚠️ ΔTheta computation failed.")
            return

        # 8) EFTA generation
        efta_collection = compute_efta(delta_theta_collection, resolution)
        if efta_collection is None:
            st.warning("⚠️ EFTA calculation failed.")
            return

        # -----------------------------------------------------
        # ⭐ NEW CRITICAL STEP — Attach ERA5 SNOW PREDICTORS ⭐
        # -----------------------------------------------------
        efta_with_snow = attach_era5_to_efta(
            efta_collection, start_date, end_date, user_roi
        )

        if efta_with_snow is None or efta_with_snow.size().getInfo() == 0:
            st.error("❌ ERA5 Snow join failed.")
            return

        st.session_state.efta_collection = efta_with_snow

        # 9) Train RF model (EFTA + Snow_depth + Snow_temp)
        rf_model = train_rf_model()
        if rf_model is None:
            st.warning("⚠️ RF training failed.")
            return

        # 10) Classify each image
        classified_images = efta_with_snow.map(
            lambda img: classify_image(img, rf_model, resolution)
        )

        # 11) Optional — clip to cropland classes
        if clip_agriculture:
            try:
                landcover = ee.Image("USGS/NLCD_RELEASES/2020_REL/NALCMS").select("landcover")

                mask = (
                    landcover.eq(9)   # Tropical/sub-tropical grassland
                    .Or(landcover.eq(10))  # Temperate/sub-polar grassland
                    .Or(landcover.eq(15))  # Cropland
                    .Or(landcover.eq(16))  # Barren land
                )

                land_cover_geom = mask.selfMask().reduceToVectors(
                    geometry=user_roi,
                    geometryType="polygon",
                    scale=30,
                    maxPixels=1e13
                )

                new_roi = user_roi.intersection(
                    land_cover_geom.geometry(), ee.ErrorMargin(30)
                )

                if new_roi.coordinates().size().getInfo() == 0:
                    st.error("❌ Whole ROI removed by land cover mask.")
                    return

                user_roi = new_roi
                classified_images = classified_images.map(lambda img: img.clip(user_roi))
                st.success("🌾 ROI clipped to cropland/grassland/barren land.")

            except Exception as e:
                st.warning(f"⚠️ Land cover clipping failed: {e}")

        # 12) Visualization (filtered to user's date selection)
        classified_visual = classified_images.filterDate(
            user_selected_start, user_selected_end
        )

        visualize_ft_classification(classified_visual, user_roi, resolution)

        st.success("🎉 Full Freeze–Thaw Pipeline Completed Successfully!")



# ========== ✅ Submit ROI Handler ==========
if submit:
    # 1) Check if user actually drew something
    if output and "all_drawings" in output and len(output["all_drawings"]) > 0:

        # Get the last drawn ROI polygon
        last_feature = output["all_drawings"][-1]
        roi_geojson = last_feature["geometry"]

        # 2) Save everything to session state
        st.session_state.user_roi = ee.Geometry(roi_geojson)
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.resolution = resolution
        st.session_state.clip_to_agriculture = clip_to_agri

        # 3) Show yellow warning message in sidebar
        st.sidebar.markdown("""
            <div style="font-size: 16px; color: #FFA500; font-weight: bold;">
                ⚠️ Please wait. Do not zoom or tap on the map after submitting the ROI until the process is completed. 
                Scroll down to view the dropdown menu of <b>"View All Freeze–Thaw Results"</b>.
            </div>
        """, unsafe_allow_html=True)

        # 4) Disable map interactions
        st.markdown(
            """
            <style>
                .folium-map {
                    pointer-events: none;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # 5) Inform user
        st.success("✅ ROI submitted and ready for processing.")

        # 6) Run the full Freeze–Thaw pipeline
        submit_roi()   # <— Your Step 13 big function is executed here

    else:
        st.warning("⚠️ Please draw an ROI before submitting.")


# ========== Sidebar Footer ==========
with st.sidebar:
    st.markdown(
        """
        <style>
        .footer-text { font-size: 12px; }
        </style>
        <div class="footer-text"><br><br><br><br><br><br><br><br><br><br><br></div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer-text">
        <strong>Developed by</strong>: Shahabeddin Taghipourjavi <br>
        <strong>Supervised by</strong>: Prof. Christophe Kinnard and Prof. Alexandre Roy <br>
        <strong>Institution</strong>: Université du Québec à Trois-Rivières (UQTR) <br>
        <strong>Address</strong>: 3351 Bd des Forges, Trois-Rivières, QC G8Z 4M3 <br>
        🔒 <strong>All rights reserved</strong> © 2025 <br>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📩 Contact Us", expanded=False):
        st.write("If you have any questions, please feel free to reach out!")
        st.markdown("[Click here to email us](mailto:Shahabeddin.taghipourjavi@uqtr.ca)")
