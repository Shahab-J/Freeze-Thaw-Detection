
# ❄️ Freeze-Thaw Mapping Tool

This interactive tool allows you to detect **freeze-thaw (FT) states** in agricultural land using **Sentinel-1 SAR data** and a **Random Forest classifier**, implemented in **Google Earth Engine (GEE)**.

## 🔍 Features

- Draw your Region of Interest (ROI) on the map
- Select date range (October–June, adjusted automatically)
- Optionally limit analysis to **cropland only **
- Compute the Exponential Freeze-Thaw Algorithm (EFTA) derived from VH radar backscatter (VHEFTA). For more info please refer to https://doi.org/10.3390/rs16071294 
- Classify Freeze-Thaw states using a pre-trained Random Forest model (Link: TBD)
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
- `LICENSE`: Please read the full LICENSE file in the Repo

## ✨ Authors
Developed by:
- Shahabeddin Taghipourjavi (ORCID: [0000-0002-2036-9863](https://orcid.org/0000-0002-2036-9863))
Institution: Université du Québec à Trois-Rivières
Address: 3351 Bd des Forges, Trois-Rivières, QC G8Z 4M3
Email: Shahabeddin.taghipourjavi@uqtr.ca

📚 Supervision
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


