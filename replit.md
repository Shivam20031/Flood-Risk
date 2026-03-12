# Flood Risk Prediction in India

## Overview
A Streamlit web application for flood risk prediction in India using machine learning (Random Forest).

## Features
- **Real-time prediction** via a sidebar with adjustable input sliders
- **Overview tab** — dataset summary and statistics
- **Data Analysis tab** — EDA charts: pie charts, correlation heatmap, histograms, flood count plots
- **Model Performance tab** — accuracy, classification report, feature importance bar chart
- **Flood Map tab** — interactive Folium map with marker clusters and heat map for flood events above a configurable rainfall threshold

## Architecture
- **Framework**: Streamlit (Python)
- **Port**: 5000
- **Entry Point**: `app.py`

## ML Model
- Algorithm: Random Forest Classifier (sklearn)
- Features: Rainfall, Temperature, Humidity, River Discharge, Water Level, Elevation, Population Density, Land Cover (one-hot), Soil Type (one-hot)
- Target: Flood Occurred (binary 0/1)
- Synthetic dataset of 500 records generated to mirror original notebook schema

## Key Libraries
- `streamlit` — web UI framework
- `scikit-learn` — Random Forest, preprocessing, metrics
- `folium` — interactive map with MarkerCluster and HeatMap
- `seaborn`, `matplotlib` — EDA charts
- `pandas`, `numpy` — data processing

## System Dependencies
- `ffmpeg`, `openjdk17` — carried from initial setup

## Running
```bash
streamlit run app.py --server.port 5000
```

## Source Notebooks
- `Project1 (1).ipynb` — Flood Prediction in India (adapted into this app)
- `Grammar_Scoring_Engine_for_Voice_Samples.ipynb` — separate notebook (not used)
