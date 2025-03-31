# ✅ Step 0: Must be FIRST
import streamlit as st
st.set_page_config(layout="wide")

import ee
import io
import sys
import math
import json 
import geemap
import folium
import subprocess
import numpy as np
import urllib.request
from PIL import Image
from datetime import date
import ipywidgets as widgets
import matplotlib.pyplot as plt
from folium.plugins import Draw
import geemap.foliumap as geemap
from google.auth import credentials
from streamlit_folium import st_folium
from google.oauth2 import service_account
from streamlit_folium import folium_static






# ========== ✅ Title and Setup ==========
st.title("🧊 Freeze–Thaw Mapping Tool")


# ========== ✅ Authenticate Earth Engine ==========
try:
    service_account = st.secrets["GEE_SERVICE_ACCOUNT"]
    private_key = st.secrets["GEE_PRIVATE_KEY"]
    credentials = ee.ServiceAccountCredentials(
        service_account,
        key_data=json.dumps({
            "type": "service_account",
            "client_email": service_account,
            "private_key": private_key,
            "token_uri": "https://oauth2.googleapis.com/token"
        })
    )
    ee.Initialize(credentials)
#    st.success("✅ Earth Engine initialized.")
except Exception as e:
    st.error(f"❌ EE Auth failed: {e}")
    st.stop()

# ========== ✅ Sidebar UI ==========
st.sidebar.title("Set Parameters")
def_start = date(2023, 10, 1)
def_end = date(2024, 6, 30)

start_date = st.sidebar.date_input("Start Date", value=def_start)
end_date = st.sidebar.date_input("End Date", value=def_end)
resolution = st.sidebar.selectbox("Resolution (meters)", [10, 30, 100])
clip_to_agri = st.sidebar.checkbox("🌾 Clip to Agricultural Land Only", value=True)
submit = st.sidebar.button("🚀 Submit ROI & Start Processing")



# ========== ✅ Set up map with default satellite view ==========
st.subheader("Draw your ROI below")
st.markdown(
    "<p style='font-size: 12px;'>(choose 'Satellite' or 'OpenStreetMap' for map view using the Layer Switcher in the top right of the map)</p>", 
    unsafe_allow_html=True
)



m = folium.Map(location=[46.29, -72.75], zoom_start=12, control_scale=True)

# Add Satellite basemap (default)
satellite_tile = folium.TileLayer(
    tiles="Esri.WorldImagery", attr="Esri", name="Satellite", overlay=False, control=True
).add_to(m)

# Add Layer control to switch between Satellite and OpenStreetMap (without Street)
folium.LayerControl(position="topright").add_to(m)

# Add drawing control to the map
draw = Draw(export=False)
draw.add_to(m)

# Render the map
output = st_folium(m, width=1300, height=600)





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

#   st.success(f"🔍 Found {image_count_val} Sentinel-1 images in ROI.")

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
    """
    Mosaics Sentinel-1 images captured on the same date to avoid duplicate acquisitions.
    Returns an ImageCollection of daily mosaics clipped to ROI.
    """

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

#   st.success(f"✅ Mosaicked {mosaicked_count} daily images.")
    return mosaicked_collection



# ✅ Step 4: SigmaDiff Computation for Streamlit
def compute_sigma_diff_pixelwise(collection):
    """
    Computes SigmaDiff as the pixel-wise difference in VH_corrected between consecutive images.
    Returns an ImageCollection with a new band 'SigmaDiff' added to each image.
    """

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
#   st.success("✅ SigmaDiff computation complete.")
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



# ✅ Step 10: Freeze–Thaw Classification Using RF for Streamlit
# 🔗 Load training data from GitHub (geojson hosted in your repo _ not from Earth Engine assets)
url = "https://raw.githubusercontent.com/Shahab-J/Freeze-Thaw-Detection/main/data/training_data.geojson"
with urllib.request.urlopen(url) as response:
    geojson_data = json.load(response)

# ✅ Convert to Earth Engine FeatureCollection
features = [ee.Feature(f) for f in geojson_data["features"]]
training_asset = ee.FeatureCollection(features)



# 🔤 Define features and label
bands = ['EFTA']  # Input feature(s) for classification
label = 'label'   # Class label (0 = Frozen, 1 = Thawed)

# ✅ Train Random Forest Model in GEE
def train_rf_model():
    """
    Trains a Random Forest model using EFTA values.

    Returns:
        ee.Classifier: Trained RF classifier
    """
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
        st.success("✅ RF model trained successfully in GEE.")
        return rf_model
    except Exception as e:
        st.error(f"❌ Failed to train RF model: {e}")
        return None



# ✅ Classify each image using the trained model
def classify_image(img, rf_model, resolution):
    """
    Classifies an image using the trained Random Forest model.

    Args:
        img (ee.Image): Input image with 'EFTA' band
        rf_model (ee.Classifier): Trained RF classifier
        resolution (int): Resolution to reproject classified results

    Returns:
        ee.Image: Image with an added 'FT_State' classification band
    """
    classified = img.select('EFTA').classify(rf_model).rename('FT_State')
    return img.addBands(classified).reproject(crs="EPSG:4326", scale=resolution)



#✅ Step 11: ROI Selection Before Processing for Streamlit







# ✅ Step 12: Compute and Summarize FT Classification for Streamlit
def summarize_ft_classification(collection, user_roi, resolution):
    """
    Computes and displays the percentage of Frozen vs. Thawed pixels
    in the classification results for each image.

    Args:
        collection (ee.ImageCollection): Collection with 'FT_State' classified band.
        user_roi (ee.Geometry): ROI to summarize statistics over.
        resolution (int): User-selected spatial resolution (10, 30, 100 m).
    """

    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ No classified images available for summarization.")
        return

    image_list = collection.toList(collection.size())
    num_images = collection.size().getInfo()

    st.markdown("### 📊 Freeze–Thaw Classification Summary")

    for i in range(num_images):
        try:
            img = ee.Image(image_list.get(i))

            # Extract timestamp and format it
            timestamp = img.date().format("YYYY-MM-dd").getInfo()

            # Compute histogram of Freeze (1) / Thaw (0) pixels
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

            thawed_percent = (thawed / total * 100) if total > 0 else 0
            frozen_percent = (frozen / total * 100) if total > 0 else 0

            # Output summary for this image
            st.markdown(f"""
            **🗓️ Image {i+1} — {timestamp}**
            - Thawed Pixels (0): {thawed:,} ({thawed_percent:.2f}%)
            - Frozen Pixels (1): {frozen:,} ({frozen_percent:.2f}%)
            ---
            """)
        except Exception as e:
            st.warning(f"⚠️ Could not summarize image {i+1}: {e}")

    st.success("✅ Freeze–Thaw Classification Summary Computed.")




# Footer Section (on the left side below the Submit ROI button)
# Footer Section (on the left side below the Submit ROI button)
with st.sidebar:
    # Footer information
    st.markdown(
        """
        ---
        **Developed by**: Shahabeddin Taghipourjavi   
        **Supervised by**: Prof. Christophe Kinnard and Prof. Alexandre Roy  
        **Institution**: Université du Québec à Trois-Rivières (UQTR)  
        **Address**: 3351 Bd des Forges, Trois-Rivières, QC G8Z 4M3  
        🔒 **All rights reserved** © 2025
        """
    )

    # Create collapsible section for Contact Us at the end of the sidebar
    with st.expander("📩 Contact Us", expanded=False):
        st.write("If you have any questions, please feel free to reach out!")
        st.markdown("[Click here to email us](mailto:Shahabeddin.taghipourjavi@uqtr.ca)")





# ✅ Step 13: Visualize FT Classification for Streamlit
def visualize_ft_classification(collection, user_roi, resolution):
    import tempfile
    import base64
    from PIL import Image
    import urllib.request

    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ No classification results available for visualization.")
        return

    image_list = collection.toList(collection.size())
    num_images = collection.size().getInfo()

    # Get start and end dates from user input
    start_date_str = st.session_state.start_date.strftime("%Y-%m-%d")
    end_date_str = st.session_state.end_date.strftime("%Y-%m-%d")

    # Display the total number of images for the selected date range
    with st.expander("🧊 View All Freeze–Thaw Results", expanded=False):
        st.markdown(
            f"🖼️ Total Images for visualization during the selected date range from "
            f"<u>{start_date_str}</u> to <u>{end_date_str}</u>: <b><span style='font-size: 30px'>{num_images}</span></b> FT classified images.",
            unsafe_allow_html=True
        )

    # Continue with the rest of the logic (image processing, visualization, etc.)
        # Loop through the images and display results
        for i in range(num_images):
            try:
                img = ee.Image(image_list.get(i))
                timestamp = img.date().format("YYYY-MM-dd").getInfo()

                # Generate the image URL for display
                url = img.select("FT_State").clip(user_roi).getThumbURL({
                    "min": 0,
                    "max": 1,
                    "dimensions": 768,
                    "palette": ["red", "blue"]
                })

                # Load the image into Streamlit
                image = Image.open(urllib.request.urlopen(url))
                st.image(image, caption=f"🗓️ {timestamp}", use_container_width=True)

                # Save TIFF file and provide a download link
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                    image.save(tmp.name)
                    with open(tmp.name, "rb") as file:
                        b64 = base64.b64encode(file.read()).decode()
                        href = f'<a href="data:file/tif;base64,{b64}" download="FT_{timestamp}.tif">📥 Download TIFF</a>'
                        st.markdown(href, unsafe_allow_html=True)

                # Compute Pixel Stats
                stats = img.select("FT_State").reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=user_roi,
                    scale=resolution,
                    maxPixels=1e13
                ).getInfo()

                # Extract histogram data
                hist = stats.get("FT_State", {})
                thawed = int(hist.get("0", 0))  # Thawed pixels
                frozen = int(hist.get("1", 0))  # Frozen pixels
                total = thawed + frozen
                thawed_pct = (thawed / total) * 100 if total > 0 else 0
                frozen_pct = (frozen / total) * 100 if total > 0 else 0

                # Display updated stats with colored squares for Frozen and Thawed and Resolution
                st.markdown(
                    f"**🧊 Freeze–Thaw Stats for {timestamp}**  \n"
                    f"🟦 Frozen: **{frozen} pixels** | {frozen_pct:.1f}%  \n"
                    f"🟥 Thawed: **{thawed} pixels** | {thawed_pct:.1f}%  \n"
                    f"📏 Resolution: **{resolution} meters**"
                )
                st.divider()

            except Exception as e:
                st.warning(f"⚠️ Error displaying image {i+1}: {e}")





# ========== ✅ Processing Pipeline ==========
def submit_roi():
    if "user_roi" not in st.session_state or st.session_state.user_roi is None:
        st.error("❌ No ROI selected. Please draw an ROI before processing.")
        return

    user_roi = st.session_state.user_roi
    resolution = st.session_state.get("resolution", 30)
    clip_agriculture = st.session_state.get("clip_to_agriculture", False)

    user_selected_start = st.session_state.start_date.strftime("%Y-%m-%d")
    user_selected_end = st.session_state.end_date.strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")

    if user_selected_end >= today:
        st.error(f"❌ End date ({user_selected_end}) is in the future. Please select a valid range.")
        return
    if user_selected_start >= user_selected_end:
        st.error("❌ Start date must be earlier than end date.")
        return

    start_year = int(user_selected_start[:4])
    if int(user_selected_start[5:7]) < 10:
        start_year -= 1
    start_date = f"{start_year}-10-01"
    end_date = f"{start_year+1}-06-30"

#   st.write(f"✅ Adjusted Processing Range: {start_date} to {end_date}")
    with st.spinner("⏳ Running full Freeze–Thaw processing pipeline..."):
        st.warning("""
            ⚠️ Please wait. Do not zoom or tap on the map after submitting the ROI until the process is completed, as this may cause the process to collapse. 
            Scroll down without tapping or zooming the selected ROI to see the dropdown menu of **"View All Freeze–Thaw Results"**.
        
        """)

       
        processed_images = process_sentinel1(start_date, end_date, user_roi, resolution)
        if processed_images is None:
            st.warning("⚠️ Step failed: Sentinel-1 processing.")
            return

        mosaicked_images = mosaic_by_date(processed_images, user_roi, start_date, end_date)
        if mosaicked_images is None:
            st.warning("⚠️ Step failed: Mosaicking by date.")
            return

        sigma_diff_collection = compute_sigma_diff_pixelwise(mosaicked_images)
        if sigma_diff_collection is None:
            st.warning("⚠️ Step failed: SigmaDiff computation.")
            return

        sigma_extreme_collection = compute_sigma_diff_extremes(sigma_diff_collection, start_year, user_roi)
        if sigma_extreme_collection is None:
            st.warning("⚠️ Step failed: Sigma extremes.")
            return

        final_k_collection = assign_freeze_thaw_k(sigma_extreme_collection)
        if final_k_collection is None:
            st.warning("⚠️ Step failed: K assignment.")
            return

        thaw_ref_image = compute_thaw_ref_pixelwise(final_k_collection, start_year, user_roi)
        if thaw_ref_image is None:
            st.warning("⚠️ Step failed: ThawRef image.")
            return

        thaw_ref_collection = final_k_collection.map(lambda img: img.addBands(thaw_ref_image))
        delta_theta_collection = compute_delta_theta(thaw_ref_collection, thaw_ref_image)
        if delta_theta_collection is None:
            st.warning("⚠️ Step failed: ΔTheta computation.")
            return

        efta_collection = compute_efta(delta_theta_collection, resolution)
        if efta_collection is None:
            st.warning("⚠️ Step failed: EFTA calculation.")
            return

        st.session_state.efta_collection = efta_collection

        rf_model = train_rf_model()
        if rf_model is None:
            st.warning("⚠️ RF Model training failed.")
            return

        classified_images = efta_collection.map(lambda img: classify_image(img, rf_model, resolution))

        # ✅ Optional: Clip to cropland using NALCMS class 15
        if clip_agriculture:
#           st.info("🌾 Cropland-only mode enabled. Intersecting ROI with agricultural land...")

            try:
                # Load NALCMS and mask class 15 (cropland)
                landcover = ee.Image("USGS/NLCD_RELEASES/2020_REL/NALCMS").select("landcover")
                cropland_mask = landcover.eq(15)

                cropland_geometry = cropland_mask.selfMask().reduceToVectors(
                    geometry=user_roi,
                    geometryType='polygon',
                    scale=30,
                    maxPixels=1e13
                )

                intersected_roi = user_roi.intersection(cropland_geometry.geometry(), ee.ErrorMargin(30))

                # Check validity
                if intersected_roi.coordinates().size().getInfo() == 0:
                    st.error("❌ Cropland mask removed entire ROI. Please select a different area or disable cropland-only mode.")
                    return

                user_roi = intersected_roi
                classified_images = classified_images.map(lambda img: img.clip(user_roi))

                st.success("🌾 ROI successfully clipped to cropland using NALCMS.")

            except Exception as e:
                st.warning(f"⚠️ Cropland masking failed: {e}")

        classified_collection_visual = classified_images.filterDate(user_selected_start, user_selected_end)
        visualize_ft_classification(classified_collection_visual, user_roi, resolution)

        st.success("✅ Full Freeze–Thaw pipeline finished successfully.")







# ========== ✅ Submit Handler ==========
if submit:
    if output and "all_drawings" in output and len(output["all_drawings"]) > 0:
        # Get the last drawn feature (ROI)
        last_feature = output["all_drawings"][-1]
        roi_geojson = last_feature["geometry"]
        
        # Store the drawn ROI in session state
        st.session_state.user_roi = ee.Geometry(roi_geojson)
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.resolution = resolution
        st.session_state.clip_to_agriculture = clip_to_agri

        # st.success("✅ ROI submitted and ready for processing.")
      
        # Running Freeze–Thaw processing pipeline without the spinner
        submit_roi()  

        # Display an alert to warn users not to interact with the map
        st.warning("⚠️ The process will collapse if interacted with after submitting the ROI. Please do not zoom or tap the map. Scroll down to see the visualization.")

    else:
        st.warning("⚠️ Please draw an ROI before submitting.")
