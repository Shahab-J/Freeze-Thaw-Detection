
# ❄️ Freeze-Thaw Mapping Tool

This interactive tool allows you to detect **freeze-thaw (FT) states** in agricultural land using **Sentinel-1 SAR data** and a **Random Forest classifier**, implemented in **Google Earth Engine (GEE)**.

## 🔍 Features

- Draw your Region of Interest (ROI) on the map
- Select date range (October–June, adjusted automatically)
- Optionally limit analysis to **cropland only (Class 15)**
- Compute SigmaDiff, Freeze/Thaw thresholds, K, DeltaTheta, and EFTA
- Classify freeze/thaw conditions using a pre-trained Random Forest model
- View classified maps directly in notebook

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Authenticate with your **Google Earth Engine** account
3. Draw your ROI and press **Submit**
4. Visualize the FT classification results

## 🛠 Requirements

- Google Colab
- Earth Engine Python API
- `geemap`, `ipywidgets`, `matplotlib`, `PIL`, `numpy`

## 🌱 Cropland-Only Option

You can choose to clip analysis only to **agricultural land** using the 2020 NALCMS dataset.

## 📁 Repository Files

- `Freeze_Thaw_Mapping_Tool.ipynb`: Main interactive tool
- `README.md`: This file

## ✨ Author

Developed by [Shahabeddin Taghipourjavi](https://github.com/Shahab-J) 
Department of Environmental Sciences, University of Québec at Trois-Rivières, Trois-Rivieres, QC G8Z 4M3, Canada.

📚 Supervision  
This project is carried out under the supervision of:  
- **Prof. Christophe Kinnard** (UQTR)  
- **Prof. Alexandre Roy** (UQTR)


Repository: https://github.com/Shahab-J/Freeze-Thaw-Detection
