# 🛒 AI Customer Segmentation Dashboard (K-Means)

An end-to-end unsupervised machine learning solution that partitions a retail consumer database into actionable marketing archetypes. This repository contains the data preprocessing layers, K-Means modeling notebooks, and a container-ready interactive Streamlit deployment dashboard.

## 🚀 Live Demo Architecture
The project features an interactive Streamlit interface where stakeholders can manually adjust customer attributes to calculate live segment predictions.

## 🧠 Real-World Data Sourcing & Feature Context
In an enterprise production environment, this dashboard does not rely on static input text. The underlying features represent a unified pipeline compiled from two primary data channels:
1. **Spending Score (Internal Behavioral Streams):** Generated dynamically using internal point-of-sale receipt logs, tracking purchase recency, transaction frequency, and total monetary volume (RFM metrics).
2. **Annual Income (External Demographic Streams):** Sourced via loyalty sign-up applications, first-party customer surveys, or geodemographic market appends based on regional census income averages.

## 🛠️ Technical Stack & Dependencies
* **Core Modeling:** Python, Scikit-Learn (KMeans, StandardScaler), NumPy, Pandas
* **Visualizations:** Matplotlib, Seaborn
* **Interface Layer:** Streamlit
