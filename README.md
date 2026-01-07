
# ❄️ Freeze-Thaw Mapping Tool

This interactive tool allows you to detect **freeze-thaw (FT) states** in agricultural land using **Sentinel-1 SAR data** and a **Random Forest classifier**, implemented in **Google Earth Engine (GEE)**.

## 🔍 Features
The Freeze-Thaw Mapping Tool offers an interactive and modular pipeline for detecting and classifying freeze-thaw conditions in agricultural areas. The key functionalities are summarized below:

- 🗺️ Interactive region selection: Allows users to define a custom Region of Interest (ROI) directly on a map interface.
- 📅 Customizable temporal range: Enables selection of a date range between October and June, with automatic adjustment to align with seasonal freeze–thaw periods.
- 🌱 Cropland, Grasslands, and Barren Lands Filter Option: Provides the option to restrict analysis to clip results to cropland, grasslands, and barren lands (Class 9: Tropical/Sub-tropical grasslands, Class 10: Temperate/Sub-polar grasslands, Class 15: Cropland, Class 16: Barren lands), based on the 2020 NALCMS dataset at 30m resolution (https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2020_REL_NALCMS).
- 🧩 User-defined spatial resolution: Supports output resolutions of 10 m, 30 m, or 100 m to match user needs and computational resources.
- ⚙️ EFTA computation: Implements the Exponential Freeze–Thaw Algorithm (EFTA) using Sentinel-1 VH backscatter for enhanced transition detection (Refer to: https://doi.org/10.3390/rs16071294).
- 🌐 Random Forest classification: Applies a pre-trained Random Forest (RF) model to classify each image pixel as either Frozen (1) or Thawed (0).
- 📊 Multi-image visualization: Displays classified results in organized image panels with date labels and a freeze-thaw color legend.
- 📉 Statistical reporting: Generates pixel-based summaries of class frequency (frozen vs. thawed) for each output image, including percentages and spatial resolution context.
- 💾 Export capability: Allows downloading classified outputs as GeoTIFF for GIS analysis or JPG/PNG for documentation and presentations.


## 🚀 How to Run

🌐 Open the web app: https://freeze-thaw-detection-kmpqcuusaqtf5ypu5h3vyg.streamlit.app/

- Draw your Region of Interest (ROI) on the map using the pentagon icon
- Click “Submit ROI” in the left sidebar
- Click “Confirm before processing” to acknowledge the processing conditions
- The processing pipeline starts automatically
- Do not interact with the map during processing — wait until execution is completed
- Visualize freeze–thaw classification results
- Export outputs as GeoTIFF

## 🛠 Requirements

- Google Colab
- Earth Engine Python API
- `geemap`, `ipywidgets`, `matplotlib`, `PIL`, `numpy`, `pillow`, `streamlit`, `earthengine-api`


## 🌱 Cropland-Only Option

You can choose to clip analysis only to **agricultural land** using the 2020 NALCMS dataset.

## 📁 Repository Files

- `Freeze_Thaw_Mapping_Tool.ipynb`: Main interactive tool
- `README.md`: This file
- `LICENSE`: Please read the full LICENSE file in the Repo

## ✨ Authors
Developed by:
- **Shahabeddin Taghipourjavi** (ORCID: [0000-0002-2036-9863](https://orcid.org/0000-0002-2036-9863))
Institution: Université du Québec à Trois-Rivières
Address: 3351 Bd des Forges, Trois-Rivières, QC G8Z 4M3
Email: Shahabeddin.taghipourjavi@uqtr.ca

📚 Supervision:
This project is carried out under the supervision of:  
- **Prof. Christophe Kinnard** (Christophe.Kinnard@uqtr.ca)  
- **Prof. Alexandre Roy** (Alexandre.roy@uqtr.ca)

Institution: Université du Québec à Trois-Rivières
Address: 3351 Bd des Forges, Trois-Rivières, QC G8Z 4M3

To reference this work:
Taghipourjavi, S., Kinnard, C., Roy, A. 2025. Freeze-Thaw Mapping Tool using Sentinel-1 SAR and Random Forest Classifier. GitHub repository. https://github.com/Shahab-J/Freeze-Thaw-Detection. 


## 🛡 License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
🔗 Please read the full LICENSE file in the Repo.

****For non-commercial use only. For other permissions, please contact: **Shahabeddin.taghipourjavi@uqtr.ca**

GitHub Repository: [Shahab-J/Freeze-Thaw-Detection](https://github.com/Shahab-J/Freeze-Thaw-Detection)  
Full License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 

## ✨ Google Colab Launch Button to README
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shahab-J/Freeze-Thaw-Detection/blob/main/Freeze_Thaw_Mapping_Tool.ipynb)
