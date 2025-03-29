import ee
import sys
import math
import json 
import geemap
import folium
import subprocess
import numpy as np
from PIL import Image
import urllib.request
import streamlit as st
from datetime import date
import ipywidgets as widgets
import matplotlib.pyplot as plt
from folium.plugins import Draw
import geemap.foliumap as geemap
from google.auth import credentials
from google.oauth2 import service_account
from streamlit_folium import folium_static




st.set_page_config(layout="wide")
st.title("🧪 Startup Package Diagnostics")

# ✅ Display environment info
st.write(f"🔧 Python: {sys.version}")

def check(name, code):
    try:
        exec(code)
        st.success(f"✅ {name} OK")
    except Exception as e:
        st.error(f"❌ {name} FAILED: {e}")

# ✅ Dependency checks
check("folium", "import folium")
check("streamlit-folium", "from streamlit_folium import folium_static")
check("geemap", "import geemap")
check("earthengine-api (ee)", "import ee")
check("pandas", "import pandas as pd")
check("numpy", "import numpy as np")
check("matplotlib", "import matplotlib.pyplot as plt")
check("Pillow (PIL)", "from PIL import Image")
check("scikit-learn", "import sklearn")
check("ipywidgets", "import ipywidgets")


# ================== Actual App ==================

st.title("🧊 Freeze–Thaw Mapping Tool")
st.write("📌 Draw your ROI on the map below and click Submit.")

# ✅ Authenticate Earth Engine
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
    st.success("✅ Earth Engine initialized.")
except Exception as e:
    st.error(f"❌ EE Auth failed: {e}")


# ✅ Show Interactive Map
try:
    Map = geemap.Map(center=[46.29, -72.75], zoom=12, draw_export=True)
    Map.add_basemap('SATELLITE')

    # ✅ If an ROI exists from previous interaction, show it
    if "user_roi" in st.session_state:
        Map.addLayer(ee.FeatureCollection([ee.Feature(st.session_state.user_roi)]), {}, "Stored ROI")

    # ✅ Render map
    Map.to_streamlit(height=600)

    # ✅ If new ROI is drawn this time, store it
    if Map.user_roi is not None:
        st.session_state.user_roi = Map.user_roi
        st.success("🗂 ROI selected and stored.")

except Exception as e:
    st.error(f"❌ Map render failed: {e}")

# ✅ Handle and store ROI in session_state
if Map.user_roi is not None:
    st.session_state.user_roi = Map.user_roi
    st.info("🗂 ROI selected and saved.")
elif "user_roi" in st.session_state:
    st.info("🗂 Using stored ROI from session.")
else:
    st.warning("✏️ Please draw an ROI using the polygon tool on the map.")



# 📆 **Date Selection Widgets**
st.session_state.start_date = st.date_input(
    'Start Date',
    value=date(2023, 10, 1),
    min_value=date(2015, 1, 1),
    max_value=date(2025, 12, 31)
)

st.session_state.end_date = st.date_input(
    'End Date',
    value=date(2024, 6, 30),
    min_value=date(2015, 1, 1),
    max_value=date(2025, 12, 31)
)

# 🌍 **Resolution Selector**
st.session_state.resolution = st.selectbox(
    'Resolution (m):',
    [10, 30, 100],
    index=1  # Default is 30m
)

# 🌍 **Cropland Clipping Option**
st.session_state.clip_to_agriculture = st.checkbox(
    'Clip to Agricultural Lands Only'
)


# 🌍 Submit Button
roi_button = st.button("Submit ROI & Start Processing", key="submit_roi")

# ✅ Check if button is pressed
if roi_button:
    st.write("🚀 Starting Freeze-Thaw Detection...")

    # ✅ Check if the ROI is in session state
    st.write("✅ ROI exists in session:", "user_roi" in st.session_state)

    if "user_roi" in st.session_state:
        user_roi = st.session_state.user_roi  # ✅ Correct way to use stored ROI
        st.info("🗂 ROI found in session.")

        # ✅ Proceed with parameters and processing
        st.write(f"Start Date: {start_date_widget}, End Date: {end_date_widget}")
        st.write(f"Resolution: {resolution_widget} meters")
        st.write(f"Agricultural Clipping: {'Yes' if clip_to_agriculture_checkbox else 'No'}")

        submit_roi()  # ✅ Call the function
    else:
        st.error("❌ No ROI selected. Please draw an ROI on the map.")


# ✅ Step 2: Sentinel-1 Processing for Streamlit
def process_sentinel1(start_date, end_date, roi, resolution):
    """Loads and processes Sentinel-1 data for the selected ROI and time range."""

    # Check if ROI is selected
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

    if collection.size().getInfo() == 0:
        st.error("❌ No Sentinel-1 images found in the selected date range and ROI.")
        return None

    st.write(f"🔍 Found {collection.size().getInfo()} Sentinel-1 images in ROI.")

    # Refined Lee Filtering Function
    def RefinedLee(img):
        img_natural = ee.Image(10.0).pow(img.select('VH').divide(10.0))
        kernel = ee.Kernel.square(7)  # Window size
        mean = img_natural.reduceNeighborhood(ee.Reducer.mean(), kernel)
        variance = img_natural.reduceNeighborhood(ee.Reducer.variance(), kernel)
        sample_stats = variance.divide(mean.multiply(mean))
        b = sample_stats.divide(sample_stats.add(1.0))
        refined = mean.add(b.multiply(img_natural.subtract(mean))).float()
        refined_db = refined.log10().multiply(10.0).rename('refined_lee')
        return img.addBands(refined_db)

    # Terrain Correction Function
    def normalizeS1(image):
        srtm = ee.Image('USGS/SRTMGL1_003')
        cos2thetaref = ee.Number(40).divide(180).multiply(math.pi).cos().pow(2)
        cos2theta = image.select('angle').divide(180).multiply(math.pi).cos().pow(2)
        vh_corrected = image.select('refined_lee').multiply(cos2thetaref).divide(cos2theta).rename('VH_corrected')
        return image.addBands(vh_corrected)

    # Apply Refined Lee and Normalize to the Sentinel-1 Collection
    processed_collection = (
        collection.map(RefinedLee)
        .map(normalizeS1)
        .map(lambda img: img.reproject(crs="EPSG:4326", scale=resolution))  # Apply selected resolution
    )

    return processed_collection


# ✅ Step 3: Mosaicking by Date for Streamlit
def mosaic_by_date(collection, roi, start_date, end_date):
    """Mosaics Sentinel-1 images captured on the same date to avoid duplicate acquisitions."""

    # Check if collection is provided
    if collection is None:
        st.error("❌ ERROR: No processed images available for mosaicking.")
        return None

    filtered_collection = collection.filterDate(start_date, end_date)

    if filtered_collection.size().getInfo() == 0:
        st.error("❌ ERROR: No images found after filtering for mosaicking.")
        return None

    # Extract unique dates
    unique_dates = filtered_collection.aggregate_array('system:time_start').map(
        lambda millis: ee.Date(millis).format('YYYY-MM-dd')
    ).distinct()

    def mosaic_same_day(date_str):
        date = ee.Date.parse('YYYY-MM-dd', date_str)
        return filtered_collection.filterDate(date, date.advance(1, 'day')).mosaic().clip(roi).set('system:time_start', date.millis())

    mosaicked_collection = ee.ImageCollection(unique_dates.map(mosaic_same_day))

    if mosaicked_collection.size().getInfo() == 0:
        st.error("❌ ERROR: No mosaicked images generated.")
        return None

    st.success(f"✅ SUCCESS: Mosaicked {mosaicked_collection.size().getInfo()} images.")
    return mosaicked_collection


# ✅ Step 4: SigmaDiff Computation for Streamlit
def compute_sigma_diff_pixelwise(collection):
    """Computes SigmaDiff images step-by-step."""
    if collection is None:
        st.error("❌ ERROR: No mosaicked images available for SigmaDiff computation.")
        return None

    # Sort the collection by time
    sorted_collection = collection.sort('system:time_start')
    image_list = sorted_collection.toList(sorted_collection.size())

    # Initialize first image with a 'SigmaDiff' band
    first_image = ee.Image(image_list.get(0)).addBands(
        ee.Image.constant(0).float().rename('SigmaDiff')
    )

    # History of the first image's VH_corrected band
    history = first_image.select('VH_corrected')
    updated_images = [first_image]

    collection_size = sorted_collection.size().getInfo()

    # Compute SigmaDiff for each image in the collection
    for i in range(1, collection_size):
        current_img = ee.Image(image_list.get(i))
        sigma_diff = current_img.select('VH_corrected').subtract(history).rename('SigmaDiff')
        updated_images.append(current_img.addBands(sigma_diff))

    # Return the updated ImageCollection with SigmaDiff band
    st.success("✅ SigmaDiff computation complete.")
    return ee.ImageCollection.fromImages(updated_images)


# ✅ Step 5: SigmaDiff Min/Max Computation for Streamlit
def compute_sigma_diff_extremes(collection, start_year, user_roi):
    """Computes SigmaDiff_min and SigmaDiff_max dynamically per pixel."""
    if collection is None or collection.size().getInfo() == 0:
        st.error("❌ ERROR: No valid SigmaDiff images found. Cannot compute extremes.")
        return None

    mid_oct_to_end_jan = collection.filterDate(
        f'{start_year}-10-15', f'{start_year+1}-01-31')

    end_feb_to_may = collection.filterDate(
        f'{start_year+1}-02-20', f'{start_year+1}-05-20')

    sigma_min = mid_oct_to_end_jan.select('SigmaDiff')\
        .reduce(ee.Reducer.min())\
        .rename('SigmaDiff_min')\
        .clip(user_roi)

    sigma_max = end_feb_to_may.select('SigmaDiff')\
        .reduce(ee.Reducer.max())\
        .rename('SigmaDiff_max')\
        .clip(user_roi)

    def attach_min_max(img):
        return img.addBands(sigma_min).addBands(sigma_max)

    updated_collection = collection.map(attach_min_max)

    st.success("✅ SigmaDiff Min/Max computation complete.")
    return updated_collection


# ✅ Step 6: Freeze-Thaw K Assignment for Streamlit
def assign_freeze_thaw_k(collection):
    """Assigns Freeze-Thaw K values based on cumulative tracking."""
    sorted_collection = collection.sort('system:time_start')
    image_list = sorted_collection.toList(sorted_collection.size())

    first_image = ee.Image(image_list.get(0)).addBands(
        ee.Image.constant(1).byte().rename('K')
    )

    freeze_tracker = first_image.select('K')
    updated_images = [first_image]

    collection_size = image_list.size().getInfo()

    # Loop through the images in the collection
    for i in range(1, collection_size):
        current_img = ee.Image(image_list.get(i))
        sigma_diff = current_img.select('SigmaDiff')
        sigma_min = current_img.select('SigmaDiff_min')
        sigma_max = current_img.select('SigmaDiff_max')

        tolerance = ee.Image(0.01)
        freeze_start = sigma_diff.subtract(sigma_min).abs().lt(tolerance)
        thaw_start = sigma_diff.subtract(sigma_max).abs().lt(tolerance)

        k = freeze_tracker.where(freeze_start, 0).where(thaw_start, 1).byte()
        freeze_tracker = k

        k = k.updateMask(sigma_diff.mask())  # Ensure valid regions only
        updated_img = current_img.addBands(k)
        updated_images.append(updated_img)

    st.success("✅ Freeze-Thaw K Assignment complete.")
    return ee.ImageCollection.fromImages(updated_images)


# ✅ Step 7: ThawRef Calculation for Streamlit
def compute_thaw_ref_pixelwise(collection, start_year, user_roi):
    """
    Computes ThawRef for each pixel as the average of the three highest VH_corrected values
    from combined Fall and Spring periods within the selected ROI.

    Args:
        collection (ee.ImageCollection): ImageCollection with 'VH_corrected' band.
        start_year (int): Starting year for the analysis.
        user_roi (ee.Geometry): The user-selected ROI.

    Returns:
        ee.Image: An image where each pixel's value is the computed ThawRef.
    """
    # Define the date ranges for Fall and Spring periods
    fall_start = f'{start_year}-10-01'
    fall_end = f'{start_year}-11-30'
    spring_start = f'{start_year+1}-04-15'
    spring_end = f'{start_year+1}-06-10'

    # Filter the collection for Fall and Spring periods
    fall_collection = collection.filterDate(fall_start, fall_end)
    spring_collection = collection.filterDate(spring_start, spring_end)

    # Combine Fall and Spring collections
    combined_collection = fall_collection.merge(spring_collection)

    # Clip images to the user-defined ROI
    combined_collection = combined_collection.map(lambda img: img.clip(user_roi))

    # Sort the collection by VH_corrected in descending order
    sorted_collection = combined_collection.sort('VH_corrected', False)

    # Limit the collection to the top 3 images per pixel
    top3_collection = sorted_collection.limit(3)

    # Compute the mean of the top 3 VH_corrected values per pixel
    thaw_ref_image = top3_collection.mean().select('VH_corrected').rename('ThawRef')

    st.success("✅ ThawRef Calculation complete.")
    return thaw_ref_image


# ✅ Step 8: DeltaTheta (ΔΘ) Calculation for Streamlit
def compute_delta_theta(collection, thaw_ref_image):
    """
    Computes ΔΘ (DeltaTheta) as ThawRef - VH_corrected for each image dynamically.

    Inputs:
        collection: ImageCollection with VH_corrected band.
        thaw_ref_image: ThawRef Image (constant band computed previously).

    Outputs:
        ImageCollection with an added DeltaTheta band per image.
    """

    def add_delta_theta(img):
        vh_corrected = img.select('VH_corrected')

        # ✅ Compute ΔΘ = ThawRef - VH_corrected
        delta_theta = thaw_ref_image.subtract(vh_corrected).rename('DeltaTheta')

        # ✅ Ensure correct masking (mask ΔΘ where VH_corrected is No Data)
        delta_theta = delta_theta.updateMask(vh_corrected.mask())

        return img.addBands(delta_theta)

    # Apply the function to the collection and return
    updated_collection = collection.map(add_delta_theta)

    st.success("✅ DeltaTheta Calculation complete.")
    return updated_collection


# ✅ Step 9: EFTA Calculation for Streamlit
def compute_efta(collection, resolution):
    """
    Computes EFTA dynamically using the exponential freeze-thaw algorithm.
    Inputs:
        collection: ImageCollection with necessary bands (K, ThawRef, VH_corrected, DeltaTheta).
        resolution: User-selected resolution for reprojecting the images.
    Outputs:
        ImageCollection with EFTA band added per image.
    """

    def calculate_efta(img):
        # ✅ Retrieve Required Bands
        k = img.select('K')
        thaw_ref = img.select('ThawRef')
        vh_corrected = img.select('VH_corrected')
        delta_theta = img.select('DeltaTheta')

        # ✅ Compute the Exponential Component
        exp_component = (ee.Image(1)
                         .add(thaw_ref.divide(vh_corrected))
                         .multiply(k.multiply(-1))
                         .exp())

        # ✅ Compute EFTA
        efta = exp_component.multiply(delta_theta).rename('EFTA')

        # ✅ Ensure proper masking (Handle No Data)
        efta = efta.updateMask(vh_corrected.mask())

        return img.addBands(efta).reproject(crs="EPSG:4326", scale=resolution)  # ✅ Apply resolution

    # Apply the EFTA calculation to each image in the collection
    updated_collection = collection.map(calculate_efta)

    st.success("✅ EFTA Calculation complete.")
    return updated_collection


# ✅ Step 10: Freeze-Thaw Classification Using RF for Streamlit

# Import training data from GEE asset
training_asset = ee.FeatureCollection('projects/ee-shahabeddinj/assets/training_data')

# Define bands and label
bands = ['EFTA']  # Feature(s) used for classification
label = 'label'   # Class label (0 = Frozen, 1 = Thawed)

# Train the RF model in GEE
def train_rf_model():
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

# Classify images in efta_collection using the trained RF model
def classify_image(img, rf_model, resolution):
    """Classify image using trained RF model."""
    classified = img.select('EFTA').classify(rf_model).rename('FT_State')
    return img.addBands(classified).reproject(crs="EPSG:4326", scale=resolution)  # Apply resolution

# Call the RF model training when button is clicked
if roi_button:
    st.write("🚀 Training RF Model...")

    rf_model = train_rf_model()

    # After the model is trained, you can use it to classify the images
    st.write("✅ RF Model trained. Now you can classify the images.")

    # Assuming `efta_collection` is available, classify the images
    classified_images = efta_collection.map(lambda img: classify_image(img, rf_model, resolution_widget))

    # Proceed with further processing or visualization of the classified images
    st.write("✅ Images classified successfully.")


# ✅ Step 11: ROI Selection Before Processing for Streamlit
def submit_roi():
    """Handles the full pipeline from ROI selection to classification."""

    # ✅ Ensure ROI exists
    if "user_roi" not in st.session_state:
        st.error("❌ No ROI selected. Please draw an ROI before processing.")
        return

    user_roi = st.session_state.user_roi
    resolution = st.session_state.get("resolution", 30)  # default resolution fallback

    # ✅ Optional Cropland Clipping
    if st.session_state.get("clip_to_agriculture", False):
        st.write("🌱 Cropland-only mode enabled. Clipping ROI to agricultural areas...")

        landcover = ee.Image("USGS/NLCD_RELEASES/2020_REL/NALCMS").select("landcover")
        cropland_mask = landcover.eq(15)

        cropland_geometry = cropland_mask.selfMask().reduceToVectors(
            geometry=user_roi,
            geometryType='polygon',
            scale=30,
            maxPixels=1e13
        )

        user_roi = user_roi.intersection(cropland_geometry.geometry(), ee.ErrorMargin(30))

        empty_check = user_roi.coordinates().size().getInfo()
        if empty_check == 0:
            st.error("❌ Cropland mask removed entire ROI. Please select a different area or disable cropland-only mode.")
            return

    # ✅ Date checks
    user_selected_start = st.session_state.start_date.strftime("%Y-%m-%d")
    user_selected_end = st.session_state.end_date.strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")

    if user_selected_end >= today:
        st.error(f"❌ ERROR: Selected end date ({user_selected_end}) is in the future. Please choose a date before {today}.")
        return

    if user_selected_start >= user_selected_end:
        st.error(f"❌ ERROR: Start date ({user_selected_start}) must be earlier than end date ({user_selected_end}).")
        return

    # ✅ Adjust year logic
    start_year = int(user_selected_start[:4])
    if int(user_selected_start[5:7]) < 10:
        start_year -= 1

    start_date = f"{start_year}-10-01"
    end_date = f"{start_year+1}-06-30"

    st.write(f"✅ Adjusted Processing Range: {start_date} to {end_date}")

    # ✅ Process pipeline
    processed_images = process_sentinel1(start_date, end_date, user_roi, resolution)
    if processed_images is None:
        return

    mosaicked_images = mosaic_by_date(processed_images, user_roi, start_date, end_date)
    if mosaicked_images is None:
        return

    sigma_diff_collection = compute_sigma_diff_pixelwise(mosaicked_images)
    if sigma_diff_collection is None:
        return

    sigma_extreme_collection = compute_sigma_diff_extremes(sigma_diff_collection, start_year, user_roi)
    if sigma_extreme_collection is None:
        return

    final_k_collection = assign_freeze_thaw_k(sigma_extreme_collection)
    if final_k_collection is None:
        st.error("❌ ERROR: K computation failed. Stopping execution.")
        return

    thaw_ref_image = compute_thaw_ref_pixelwise(final_k_collection, start_year, user_roi)
    if thaw_ref_image is None:
        return

    thaw_ref_collection = final_k_collection.map(lambda img: img.addBands(thaw_ref_image))
    delta_theta_collection = compute_delta_theta(thaw_ref_collection, thaw_ref_image)
    if delta_theta_collection is None:
        return

    efta_collection = compute_efta(delta_theta_collection, resolution)
    if efta_collection is None:
        return

    st.session_state.efta_collection = efta_collection

    # ✅ Train RF
    rf_model = ee.Classifier.smileRandomForest(
        numberOfTrees=150,
        variablesPerSplit=1,
        minLeafPopulation=3,
        seed=42
    ).train(
        features=training_asset,
        classProperty='label',
        inputProperties=['EFTA']
    )

    # ✅ Classify and visualize
    classified_collection = efta_collection.map(lambda img: img.addBands(
        img.select('EFTA').classify(rf_model).rename('FT_State')
    ))

    classified_collection_visual = classified_collection.filterDate(
        user_selected_start, user_selected_end
    )

    visualize_ft_classification(classified_collection_visual, user_roi, resolution)
    st.success("✅ All Processing Completed.")

# ✅ Step 12: Compute and Summarize FT Classification for Streamlit
def summarize_ft_classification(collection, user_roi, resolution):
    """
    Computes statistical summary of Freeze-Thaw classification (Frozen vs. Thawed pixels)
    for each image in the collection.
    """
    if collection.size().getInfo() == 0:
        st.error("❌ No Freeze-Thaw classified images available for summarization.")
        return

    image_list = collection.toList(collection.size())
    num_images = collection.size().getInfo()

    st.write("\n📊 **Freeze-Thaw Classification Summary**:\n")

    for i in range(num_images):
        img = ee.Image(image_list.get(i))

        # Extract date
        timestamp = img.get("system:time_start").getInfo()
        date_str = ee.Date(timestamp).format("YYYY-MM-dd").getInfo()

        # Apply user-selected resolution dynamically
        stats = img.select("FT_State").reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=user_roi,
            scale=resolution,  # Apply selected resolution
            maxPixels=1e13
        ).getInfo()

        hist = stats.get("FT_State", {})

        thawed_count = int(hist.get("0", 0))
        frozen_count = int(hist.get("1", 0))
        total_count = thawed_count + frozen_count
        thawed_percent = (thawed_count / total_count * 100) if total_count > 0 else 0
        frozen_percent = (frozen_count / total_count * 100) if total_count > 0 else 0

        st.write(f"📅 Image {i+1} (Date: {date_str}):")
        st.write(f"   - **Thawed Pixels (0):** {thawed_count} ({thawed_percent:.2f}%)")
        st.write(f"   - **Frozen Pixels (1):** {frozen_count} ({frozen_percent:.2f}%)")
        st.write("--------------------------------------------------")

    st.success("✅ Freeze-Thaw Classification Summary Computed Successfully.")


# ✅ Step 13: Visualize FT Classification for Streamlit
def visualize_ft_classification(collection, user_roi, resolution):
    """
    Visualizes Freeze-Thaw classification images with a single-row legend.
    """
    if collection.size().getInfo() == 0:
        st.error("❌ No Freeze-Thaw classification images available for visualization.")
        return

    image_list = collection.toList(collection.size())
    num_images = collection.size().getInfo()

    st.write(f"📊 Total Freeze-Thaw Classified Images Available for Visualization: {num_images}")

    cols = 3  # Three columns for images
    rows = (num_images // cols) + (num_images % cols > 0)  # Calculate required rows

    # If the last row is full, legend goes in a new row, else it fits after the last image
    total_slots = rows * cols
    legend_needed = total_slots > num_images
    if not legend_needed:
        rows += 1  # Add a row for the legend if needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    summary_lines = []  # To store compact pixel summaries

    for i in range(num_images):
        img = ee.Image(image_list.get(i))

        # Extract the date for reference
        timestamp = img.get("system:time_start").getInfo()
        date_str = ee.Date(timestamp).format("YYYY/MM/dd").getInfo()

        # Get the thumbnail URL for FT_State
        url = img.select("FT_State").clip(user_roi).getThumbURL({
            "min": 0, "max": 1, "dimensions": 512, "palette": ["red", "blue"]
        })

        # Load image and plot
        image_array = np.array(PIL.Image.open(urllib.request.urlopen(url)))
        axes[i].imshow(image_array, cmap="bwr", vmin=0, vmax=1)
        axes[i].set_title(f"{date_str}")
        axes[i].axis("off")  # Remove grids

        # Compute pixel statistics
        stats = img.select("FT_State").reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=user_roi,
            scale=resolution,  # Use user-selected resolution
            maxPixels=1e13
        ).getInfo()

        hist = stats.get("FT_State", {})
        thawed_count = int(hist.get("0", 0))
        frozen_count = int(hist.get("1", 0))
        total_count = thawed_count + frozen_count
        thawed_percent = (thawed_count / total_count * 100) if total_count > 0 else 0
        frozen_percent = (frozen_count / total_count * 100) if total_count > 0 else 0

        # Store compact summary line
        summary_lines.append(f"{date_str}: Frozen={frozen_count} ({frozen_percent:.1f}%) | Thawed={thawed_count} ({thawed_percent:.1f}%)")

    # Place legend in the last available space
    if legend_needed:
        legend_ax = axes[num_images]  # Place legend in the first available empty slot
    else:
        legend_ax = fig.add_subplot(rows, cols, num_images + 1)  # New row for the legend

    legend_ax.axis("off")  # Remove axis

    # Create legend colors and labels in a single row, ensuring it's done only once
    legend_ax.legend(
        labels=["**Thawed**", "**Frozen**"],
        handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                 plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)],
        loc="center", ncol=2, frameon=False, fontsize=10  # Single row for the legend
    )

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Print compact pixel summary for each image
    st.write(f"\n📊 **Freeze-Thaw Pixel Summary with resolution of {resolution} meters**:")
    for line in summary_lines:
        st.write(line)

    st.success("✅ Freeze-Thaw Classification Visualization Complete.")

# ✅ Step: Attach the function to the button in Streamlit
if roi_button:
    # Trigger the processing when the button is clicked
    st.session_state.start_date = start_date_widget
    st.session_state.end_date = end_date_widget
    st.session_state.clip_to_agriculture = clip_to_agriculture_checkbox
    st.session_state.resolution = resolution_widget

    if "user_roi" in st.session_state:
        submit_roi()
    else:
        st.error("❌ No ROI selected. Please draw one on the map.")


st.write("✅ ROI exists in session:", "user_roi" in st.session_state)
