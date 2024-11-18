import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial import cKDTree
import base64

# Load the cell tower dataset from GitHub
cell_tower_data = pd.read_csv("https://raw.githubusercontent.com/khidwiser/MSC-DSA-FINAL-PROJECT/main/keCellTowers.csv")

# Function to add engineered features for clustering
def feature_engineering(data):
    # Sample a subset of the dataset for faster visualization
    sampled_data = data.sample(n=500, random_state=42)  # Adjust size as needed
    
    # Calculate distance of each cell tower from the dataset's center
    center_point = (sampled_data['lat'].mean(), sampled_data['lon'].mean())
    sampled_data['Distance_From_Center'] = sampled_data.apply(
        lambda row: geodesic((row['lat'], row['lon']), center_point).kilometers, axis=1
    )

    # Convert timestamps to datetime and compute the active days
    sampled_data['created'] = pd.to_datetime(sampled_data['created'], unit='s')
    sampled_data['updated'] = pd.to_datetime(sampled_data['updated'], unit='s')
    sampled_data['Days_Active'] = (sampled_data['updated'] - sampled_data['created']).dt.days

    # Calculate the density of cell towers within a 10 km radius
    def calculate_density(data, radius_km=10):
        tree = cKDTree(data[['lat', 'lon']].values)
        counts = tree.query_ball_point(data[['lat', 'lon']].values, r=radius_km / 111)  # Convert km to degrees
        return [len(c) - 1 for c in counts]  # Exclude the cell tower itself

    sampled_data['Density_10km'] = calculate_density(sampled_data)
    return sampled_data

# Function to perform KMeans clustering and evaluate clusters
def perform_clustering_and_evaluation(data, num_clusters):
    clustering_features = ['lat', 'lon', 'Distance_From_Center', 'Days_Active', 'Density_10km']
    X = data[clustering_features]
    
    # Train KMeans with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X)
    
    # Compute evaluation metrics
    silhouette_avg = silhouette_score(X, data['Cluster'])
    davies_bouldin_avg = davies_bouldin_score(X, data['Cluster'])
    calinski_harabasz_avg = calinski_harabasz_score(X, data['Cluster'])
    
    return data, X, silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg

# Streamlit app interface
st.title("Interactive Cell Tower Clustering Visualization for Kenya")

# User selects the number of clusters
optimal_k = st.slider("Select the number of clusters (k)", min_value=2, max_value=10, value=3, step=1)

# Feature engineering to prepare data for clustering
sampled_data = feature_engineering(cell_tower_data)

# Perform clustering and calculate metrics
clustered_data, X, silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg = perform_clustering_and_evaluation(sampled_data, optimal_k)

# Initial map to show data points before clustering
st.subheader("Initial Distribution of Cell Tower Locations")
initial_map = folium.Map(location=[sampled_data['lat'].mean(), sampled_data['lon'].mean()], zoom_start=6)
for _, row in sampled_data.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(initial_map)
folium_static(initial_map)

# Map to show clustered data points with cluster-specific colors
st.subheader(f"Clustered Cell Towers Visualization (k = {optimal_k})")
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
clustered_map = folium.Map(location=[sampled_data['lat'].mean(), sampled_data['lon'].mean()], zoom_start=6)
marker_cluster = MarkerCluster().add_to(clustered_map)
for _, row in clustered_data.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"Cell: {row['cell']}, Cluster: {row['Cluster']}",
        icon=folium.Icon(color=colors[row['Cluster'] % len(colors)])
    ).add_to(marker_cluster)

# Add cluster centers to the clustered map
for i, center in enumerate(KMeans(n_clusters=optimal_k, random_state=42).fit(X).cluster_centers_):
    folium.Marker(
        location=[center[0], center[1]],
        popup=f"Cluster Center {i}",
        icon=folium.Icon(color='black', icon='info-sign')
    ).add_to(clustered_map)
folium_static(clustered_map)

# Display clustering evaluation metrics
st.subheader("Clustering Evaluation Metrics")
st.write(f"**Silhouette Score:** {silhouette_avg:.4f}")
st.write(f"**Davies-Bouldin Index:** {davies_bouldin_avg:.4f}")
st.write(f"**Calinski-Harabasz Index:** {calinski_harabasz_avg:.4f}")

# Download link for clustered data as a CSV file
csv = clustered_data.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="clustered_data.csv">Download Clustered Data CSV</a>'
st.markdown(href, unsafe_allow_html=True)
