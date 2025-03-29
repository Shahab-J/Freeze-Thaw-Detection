import ee
import sys
import math
import json 
import geemap
import PIL.Image
import subprocess
import numpy as np
import urllib.request
import streamlit as st
from datetime import date
import ipywidgets as widgets
import matplotlib.pyplot as plt
from google.auth import credentials
from google.oauth2 import service_account




# Earth Engine Auth
service_account_dict = dict(st.secrets["GEE_SERVICE_ACCOUNT_JSON"])
service_account_json = json.dumps(service_account_dict)

credentials = ee.ServiceAccountCredentials(
    service_account_dict["client_email"],
    key_data=service_account_json
)

ee.Initialize(credentials)

# Create interactive map
Map = geemap.Map()
Map.add_draw_control()

# Display in Streamlit
Map.to_streamlit()



# 🌍 **Step 2: Interactive Map for ROI Selection**
def display_map():
    Map = geemap.Map()
    Map.add_basemap('SATELLITE')
    Map.centerObject(ee.Geometry.Point([-72.75, 46.29]), 12)
    Map.add_draw_control()
    return Map

# Display the interactive map
Map = display_map()
st.write("🔹 **Draw** your ROI on the map above and click **Submit**.")
st.pydeck_chart(Map)  # Display the map in Streamlit

# 📆 **Date Selection Widgets**
start_date_widget = st.date_input(
    'Start Date',
    value=date(2023, 10, 1),
    min_value=date(2015, 1, 1),
    max_value=date(2025, 12, 31)
)

end_date_widget = st.date_input(
    'End Date',
    value=date(2024, 6, 30),
    min_value=date(2015, 1, 1),
    max_value=date(2025, 12, 31)
)

# 🌍 **Resolution Selector**
resolution_widget = st.selectbox(
    'Resolution (m):',
    [10, 30, 100],
    index=1  # Default is 30m
)

# 🌍 **Cropland Clipping Option**
clip_to_agriculture_checkbox = st.checkbox(
    'Clip to Agricultural Lands Only'
)


# 🌍 **Submit Button**
roi_button = st.button('Submit ROI & Start Processing')

# Check if button is pressed
if roi_button:
    st.write("🚀 Starting Freeze-Thaw Detection...")

    # Get the ROI from the map
    roi = Map.user_roi  # Assuming the map stores the ROI after user draws it

    # Check if an ROI is selected
    if roi is None:
        st.error("❌ No ROI selected. Please draw an ROI on the map.")
    else:
        # Process the data with the selected parameters
        st.write("✅ Processing your data...")

        # Call your process_freeze_thaw function here (e.g., passing the parameters)
        # For now, we simulate a message
        st.write(f"Start Date: {start_date_widget}, End Date: {end_date_widget}")
        st.write(f"Resolution: {resolution_widget} meters")
        st.write(f"Agricultural Clipping: {'Yes' if clip_to_agriculture_checkbox else 'No'}")

        # Process and visualize the results
        # result = process_freeze_thaw(roi, start_date_widget, end_date_widget, resolution_widget)
        # st.pydeck_chart(result)  # Display processed results (could be a map or image)


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
    """
    Handles the entire pipeline from ROI selection to Freeze-Thaw classification.
    """
    # Ensure the user has drawn an ROI before proceeding
    if "user_roi" not in st.session_state:
        st.error("❌ No ROI selected. Please draw an ROI before processing.")
        return

    # Extract user-selected ROI
    user_roi = st.session_state.user_roi  # Use Streamlit's session_state to store ROI

    # Optional Cropland Masking (Clip to Agriculture Only)
    # This section is commented out, but you can enable it if needed
    # if clip_to_agriculture_checkbox.value:
    #    st.write("🌱 Cropland-only mode enabled. Clipping ROI to agricultural areas ...")

    # Load North America 2020 Land Cover (NALCMS)
    landcover = ee.Image("USGS/NLCD_RELEASES/2020_REL/NALCMS").select("landcover")

    # Create a mask for cropland only (class 15)
    cropland_mask = landcover.eq(15)

    # Extract cropland geometry within the user ROI
    cropland_geometry = cropland_mask.selfMask().reduceToVectors(
        geometry=user_roi,
        geometryType='polygon',
        scale=30,
        maxPixels=1e13
    )

    # Intersect user's ROI with cropland polygons
    user_roi = user_roi.intersection(cropland_geometry.geometry(), ee.ErrorMargin(30))

    # Check if ROI still contains valid cropland area
    empty_check = user_roi.coordinates().size().getInfo()
    if empty_check == 0:
        st.error("❌ Cropland mask removed entire ROI. Please select a different area or disable cropland-only mode.")
        return

    user_selected_start = st.session_state.start_date.strftime("%Y-%m-%d")
    user_selected_end = st.session_state.end_date.strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")  # Get today's date as string

    # Prevent user from selecting a future date beyond today
    if user_selected_end >= today:
        st.error(f"❌ ERROR: Selected end date ({user_selected_end}) is in the future. Please choose a date before {today}.")
        return

    # Ensure start_date is before end_date
    if user_selected_start >= user_selected_end:
        st.error(f"❌ ERROR: Start date ({user_selected_start}) must be earlier than end date ({user_selected_end}).")
        return

    # Determine Processing Window (Always October to June)
    start_year = int(user_selected_start[:4])

    # Rule: If start date is between October 1st and December 31st → Keep `start_year`
    # Rule: If start date is between January 1st and September 30th → Subtract 1 from `start_year`
    if int(user_selected_start[5:7]) < 10:  # If month is January (01) to September (09)
        start_year -= 1

    start_date = f"{start_year}-10-01"
    end_date = f"{start_year+1}-06-30"

    # Handle Future Data Gaps
    # If the user-selected period extends beyond available data (today), limit visualization
    if end_date >= today:
        st.warning(f"⚠️ WARNING: Some future dates are unavailable. Adjusting processing to available data.")
        end_date = today  # Clip the processing end to today

    st.write(f"✅ User Selected Inputs:\n - ROI: Defined\n - Start: {user_selected_start}\n - End: {user_selected_end}")
    st.write(f"🔹 Adjusted Processing Range: {start_date} to {end_date}")

    # Process Sentinel-1 Data
    processed_images = process_sentinel1(start_date, end_date, user_roi)
    if processed_images is None:
        return

    # Apply Mosaicking
    mosaicked_images = mosaic_by_date(processed_images, user_roi, start_date, end_date)
    if mosaicked_images is None:
        return

    # Compute SigmaDiff (No Visualization)
    sigma_diff_collection = compute_sigma_diff_pixelwise(mosaicked_images)
    if sigma_diff_collection is None:
        return

    # Compute SigmaDiff Min/Max (No Visualization)
    sigma_extreme_collection = compute_sigma_diff_extremes(sigma_diff_collection, start_year, user_roi)
    if sigma_extreme_collection is None:
        return

    # Compute Freeze-Thaw K
    final_k_collection = assign_freeze_thaw_k(sigma_extreme_collection)
    if final_k_collection is None:
        st.error("❌ ERROR: K computation failed. Stopping execution.")
        return

    # Compute ThawRef
    thaw_ref_image = compute_thaw_ref_pixelwise(final_k_collection, start_year, user_roi)
    if thaw_ref_image is None:
        return

    # Add ThawRef Band to Each Image
    thaw_ref_collection = final_k_collection.map(lambda img: img.addBands(thaw_ref_image))

    # Compute DeltaTheta
    delta_theta_collection = compute_delta_theta(thaw_ref_collection, thaw_ref_image)
    if delta_theta_collection is None:
        st.error("❌ ERROR: DeltaTheta computation failed. Stopping execution.")
        return

    # Compute EFTA
    efta_collection = compute_efta(delta_theta_collection)
    if efta_collection is None:
        st.error("❌ ERROR: EFTA computation failed. Stopping execution.")
        return

    # Train the RF model in GEE
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

    # Classify images in efta_collection using the trained RF model
    classified_collection = efta_collection.map(lambda img: img.addBands(
        img.select('EFTA').classify(rf_model).rename('FT_State')
    ))

    # Ensure Visualization Uses Only User-Selected Date Range
    classified_collection_visual = classified_collection.filterDate(user_selected_start, user_selected_end)

    # Visualize Freeze-Thaw Classification (Only User-Selected Date Range)
    visualize_ft_classification(classified_collection_visual, user_roi, resolution_widget.value)

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

# Define the button in Streamlit
roi_button = st.button("Submit ROI & Start Processing")

# Check if the button is clicked
if roi_button:
    # Trigger the processing when the button is clicked
    submit_roi()  # Call the function to process the selected ROI

