import pandas as pd
import numpy as np
import sklearn.cluster as Kmeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib as jb
k = 3
df = pd.read_csv('customer_segmentation_data.csv')
X = df[['Annual_Income_k', 'Spending_Score']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Kmeans.KMeans(n_clusters=k, init='k-means++', random_state=42)
model.fit(X_scaled)
jb.dump(model, 'customer_segmentation_model.pkl')
jb.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")