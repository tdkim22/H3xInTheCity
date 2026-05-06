# Loading necessary libraries; ensure all are installed prior to running code
import os
import json
import random
import warnings
import logging
from pathlib import Path

import h3
import folium
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.colors as mcolors
import torch
import pytorch_lightning as pl_lightning

from shapely.geometry import mapping
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import hdbscan
from minisom import MiniSom

from srai.loaders import OSMPbfLoader
from srai.regionalizers import geocode_to_region_gdf, H3Regionalizer
from srai.joiners import IntersectionJoiner
from srai.embedders import Hex2VecEmbedder
from srai.neighbourhoods import H3Neighbourhood
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER


# -----------------------------------------------------
# App / environment setup
# -----------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("srai").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("medium")

st.set_page_config(page_title="Hex2Vec + Clustering Explorer", layout="wide")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------

# Setting seed to 42 for reproducibility
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl_lightning.seed_everything(seed, workers=True)


# Getting OSM loader
@st.cache_resource(show_spinner=False)
def get_loader() -> OSMPbfLoader:
    return OSMPbfLoader()


# Getting Hex2Vec embedder  
@st.cache_resource(show_spinner=False)
def get_embedder() -> Hex2VecEmbedder:
    return Hex2VecEmbedder(encoder_sizes=[256, 128, 64])


#  Obtaining geodataframe for target region
@st.cache_data(show_spinner=False)
def get_area(place_name: str) -> gpd.GeoDataFrame:
    return geocode_to_region_gdf(place_name)


# Loading OSM features for the target region from geodataframe
@st.cache_data(show_spinner=False)
def load_osm_features(place_name: str):
    area = get_area(place_name)
    loader = get_loader()
    loader_gdf = loader.load(area, HEX2VEC_FILTER)
    return area, loader_gdf


# building H3 region for target region from geodataframe
@st.cache_data(show_spinner=False)
def build_regions_from_area(place_name: str, resolution: int) -> gpd.GeoDataFrame:
    area = get_area(place_name)
    regionalizer = H3Regionalizer(resolution=resolution)
    regions = regionalizer.transform(area)
    if regions.index.name != "region_id":
        regions.index.name = "region_id"
    return regions


# Building H3 region with features 
@st.cache_data(show_spinner=False)
def build_regions_from_features(place_name: str, resolution: int):
    area, loader_gdf = load_osm_features(place_name)
    regionalizer = H3Regionalizer(resolution=resolution)
    regions_gdf = regionalizer.transform(loader_gdf)
    if regions_gdf.index.name != "region_id":
        regions_gdf.index.name = "region_id"
    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(regions_gdf, loader_gdf)
    return area, loader_gdf, regions_gdf, joint_gdf


# -----------------------------------------------------
# Polars-heavy feature engineering
# -----------------------------------------------------

# Building binary feature dataframe from loader gdf with OSM features
def build_binary_feature_frame(loader_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    feature_columns = list(HEX2VEC_FILTER.keys())

    # Keep only columns that actually exist in the loader output.
    present_cols = [c for c in feature_columns if c in loader_gdf.columns]
    binary_df = pd.DataFrame(index=loader_gdf.index)

    for col in present_cols:
        binary_df[col] = loader_gdf[col].notna().astype(np.int8)

    binary_df.index.name = "feature_id"
    return binary_df


# Aggregating the features, convert minimal columns to polars
def aggregate_features_per_hex_polars(
    joint_gdf: gpd.GeoDataFrame,
    binary_feature_df: pd.DataFrame,
    region_index: pd.Index,
) -> pd.DataFrame:
    # Convert only the minimal columns to Polars.
    joint_pd = joint_gdf.reset_index()[["region_id", "feature_id"]].copy()
    feats_pd = binary_feature_df.reset_index().copy()

    joint_pl = pl.from_pandas(joint_pd)
    feats_pl = pl.from_pandas(feats_pd)

    feature_cols = [c for c in feats_pd.columns if c != "feature_id"]

    merged_pl = joint_pl.join(feats_pl, on="feature_id", how="left")

    agg_exprs = [pl.col(col).sum().fill_null(0).alias(col) for col in feature_cols]
    grouped_pl = merged_pl.group_by("region_id").agg(agg_exprs)

    grouped_pd = grouped_pl.to_pandas()
    grouped_pd = grouped_pd.set_index("region_id")
    grouped_pd = grouped_pd.reindex(region_index, fill_value=0)

    for col in grouped_pd.columns:
        grouped_pd[col] = grouped_pd[col].fillna(0)

    return grouped_pd


# -----------------------------------------------------
# Embeddings / model prep
# -----------------------------------------------------

# Preparing data with previous function calls and filter to administrative boundary
@st.cache_data(show_spinner=False)
def prepare_data(place_name: str, resolution: int):
    area, loader_gdf, regions_gdf, joint_gdf = build_regions_from_features(place_name, resolution)

    binary_feature_df = build_binary_feature_frame(loader_gdf)
    features_per_hex = aggregate_features_per_hex_polars(
        joint_gdf=joint_gdf,
        binary_feature_df=binary_feature_df,
        region_index=regions_gdf.index,
    )

    # Filter to administrative boundary after aggregation.
    regions_gdf_filtered = gpd.sjoin(regions_gdf, area, predicate="intersects", how="inner").reset_index()

    if "region_id_left" in regions_gdf_filtered.columns:
        regions_gdf_filtered = regions_gdf_filtered.rename(columns={"region_id_left": "region_id"})
    elif "index" in regions_gdf_filtered.columns and "region_id" not in regions_gdf_filtered.columns:
        regions_gdf_filtered = regions_gdf_filtered.rename(columns={"index": "region_id"})

    regions_gdf_filtered = (
        regions_gdf_filtered
        .drop(columns=[c for c in ["index_right"] if c in regions_gdf_filtered.columns], errors="ignore")
        .drop_duplicates(subset="region_id")
        .set_index("region_id")
    )

    regions_gdf_filtered.index.name = "region_id"
    features_per_hex = features_per_hex.reindex(regions_gdf_filtered.index, fill_value=0)

    return area, loader_gdf, regions_gdf_filtered, joint_gdf, features_per_hex


# Training the embeddings with Hex2Vec on gdfs, and weighing them using scaled lat/long coords
@st.cache_data(show_spinner=False)
def train_embeddings(
    place_name: str,
    resolution: int,
    k_ring: int,
    max_epochs: int,
    learning_rate: float,
    batch_size: int,
    location_weight: float,
    seed: int,
):
    set_global_seed(seed)
    area, loader_gdf, regions_gdf, joint_gdf, features_per_hex = prepare_data(place_name, resolution)

    neighbors = H3Neighbourhood(regions_gdf, k_ring)
    embedder = get_embedder()

    embeddings = embedder.fit_transform(
        regions_gdf,
        loader_gdf,
        joint_gdf,
        neighbors,
        batch_size=batch_size,
        learning_rate=learning_rate,
        trainer_kwargs={
            "max_epochs": max_epochs,
            "accelerator": "auto",
            "enable_progress_bar": False,
            "logger": False,
            "enable_checkpointing": False,
        },
    )

    embeddings = embeddings.groupby(embeddings.index).mean()
    embeddings = embeddings.loc[regions_gdf.index]

    coords = np.array([h3.cell_to_latlng(h) for h in embeddings.index], dtype=float)
    scaled_coords = StandardScaler().fit_transform(coords)
    weighted_features = np.hstack([embeddings.to_numpy(), scaled_coords * location_weight])

    weighted_features_df = pd.DataFrame(weighted_features, index=embeddings.index)
    weighted_features_df.index.name = "region_id"

    return area, loader_gdf, regions_gdf, joint_gdf, features_per_hex, embeddings, weighted_features_df


# -----------------------------------------------------
# Clustering
# -----------------------------------------------------

# Checking labels before returning silhouette score to prevent errors
def safe_silhouette_score(X: pd.DataFrame, labels: np.ndarray):
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.nan
    if len(unique) == 2 and -1 in unique and np.sum(labels != -1) < 2:
        return np.nan
    try:
        sample_size = min(10000, len(X))
        return float(silhouette_score(X, labels, sample_size=sample_size, random_state=42))
    except Exception:
        return np.nan


# Running clustering based off of user's chosen model
@st.cache_data(show_spinner=False)
def run_clustering(model_name: str, weighted_features: pd.DataFrame, n_clusters: int):
    X = weighted_features.to_numpy()

    if model_name == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        meta = {"n_clusters": int(n_clusters)}

    elif model_name == "GMM":
        model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        meta = {"n_clusters": int(n_clusters)}

    elif model_name == "DBSCAN":
        model = DBSCAN(eps=0.75, min_samples=5)
        labels = model.fit_predict(X)
        meta = {
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "noise_points": int(np.sum(labels == -1)),
        }

    elif model_name == "HDBSCAN":
        model = hdbscan.HDBSCAN(
            min_cluster_size=100,
            min_samples=10,
            metric="euclidean",
            prediction_data=True,
        )
        labels = model.fit_predict(X)
        meta = {
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "noise_points": int(np.sum(labels == -1)),
        }

    elif model_name == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        meta = {"n_clusters": int(n_clusters)}

    elif model_name == "Spectral":
        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
        )
        labels = model.fit_predict(X)
        meta = {"n_clusters": int(n_clusters)}

    elif model_name == "SOM":
        som_grid_rows = 10
        som_grid_cols = 10
        som = MiniSom(
            som_grid_rows,
            som_grid_cols,
            X.shape[1],
            sigma=0.5,
            learning_rate=0.5,
            random_seed=42,
        )
        som.random_weights_init(X)
        som.train_random(X, num_iteration=500, verbose=False)
        labels = np.array([
            som.winner(x)[0] * som_grid_cols + som.winner(x)[1]
            for x in X
        ])
        meta = {"n_clusters": int(len(np.unique(labels)))}

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    sil = safe_silhouette_score(weighted_features, labels)
    return labels, meta, sil


# -----------------------------------------------------
# Mapping helpers
# -----------------------------------------------------

# Defining the color map used for cluster map
def make_cluster_color_map(unique_clusters):
    non_noise = [c for c in sorted(unique_clusters) if c != -1]
    cmap = plt_colors(len(non_noise))
    color_map = {}
    j = 0
    for cluster_id in sorted(unique_clusters):
        if cluster_id == -1:
            color_map[cluster_id] = "#808080"
        else:
            color_map[cluster_id] = cmap[j]
            j += 1
    return color_map


# Plotting colors to hexes
@st.cache_data(show_spinner=False)
def plt_colors(n: int):
    if n <= 0:
        return []
    import matplotlib.pyplot as plt
    colors = plt.cm.get_cmap("tab20", n).colors
    return [mcolors.to_hex(c) for c in colors]


# Creating feature map using geojson
@st.cache_data(show_spinner=False)
def make_feature_map_geojson(regions_gdf: gpd.GeoDataFrame, features_per_hex: pd.DataFrame, selected_feature: str):
    plot_gdf = regions_gdf.join(features_per_hex[[selected_feature]], how="left")
    plot_gdf[selected_feature] = plot_gdf[selected_feature].fillna(0)
    plot_gdf = plot_gdf.reset_index()
    return plot_gdf.to_json()


# Creating cluster map using geojson
@st.cache_data(show_spinner=False)
def make_cluster_map_geojson(regions_gdf: gpd.GeoDataFrame, cluster_col: str):
    plot_gdf = regions_gdf[["geometry", cluster_col]].reset_index()
    return plot_gdf.to_json()


# Rendering the feature map with Folium
def render_feature_map(regions_gdf: gpd.GeoDataFrame, features_per_hex: pd.DataFrame, feature_name: str):
    plot_gdf = regions_gdf.join(features_per_hex[[feature_name]], how="left")
    plot_gdf[feature_name] = plot_gdf[feature_name].fillna(0)
    plot_gdf = plot_gdf.reset_index()

    center = plot_gdf.unary_union.centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="CartoDB Voyager")

    folium.Choropleth(
        geo_data=plot_gdf.to_json(),
        data=plot_gdf,
        columns=["region_id", feature_name],
        key_on="feature.properties.region_id",
        fill_color="YlOrRd",
        fill_opacity=0.85,
        line_opacity=0.2,
        legend_name=f"{feature_name} count per hexagon",
        highlight=True,
    ).add_to(fmap)

    tooltip_layer = folium.GeoJson(
        plot_gdf.to_json(),
        style_function=lambda _: {
            "fillOpacity": 0.0,
            "color": "black",
            "weight": 0.2,
        },
        tooltip=folium.GeoJsonTooltip(fields=[feature_name], aliases=[f"{feature_name}: "]),
    )
    tooltip_layer.add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap


# Rendering the cluster map with Folium
def render_cluster_map(regions_gdf: gpd.GeoDataFrame, cluster_labels: np.ndarray, cluster_name: str):
    plot_gdf = regions_gdf[["geometry"]].copy()
    plot_gdf[cluster_name] = cluster_labels
    plot_gdf = plot_gdf.reset_index()

    unique_clusters = plot_gdf[cluster_name].unique()
    color_map = make_cluster_color_map(unique_clusters)

    center = plot_gdf.unary_union.centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="CartoDB Voyager")

    for cluster_id in sorted(unique_clusters):
        cluster_gdf = plot_gdf[plot_gdf[cluster_name] == cluster_id]
        label = f"Noise (-1)" if cluster_id == -1 else f"Cluster {cluster_id}"
        color = color_map[cluster_id]

        folium.GeoJson(
            cluster_gdf.to_json(),
            name=label,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": "black",
                "weight": 0.4,
                "fillOpacity": 0.75,
            },
            tooltip=folium.GeoJsonTooltip(fields=[cluster_name], aliases=[f"{cluster_name}: "]),
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap


# Converting the Folium mapping to Streamlit component
def folium_to_streamlit(fmap, height: int = 700):
    components.html(fmap._repr_html_(), height=height, scrolling=True)


# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------

# Various bits of code implementing Streamlit app features and interactive functionality

st.title("Hex2Vec + Geospatial Clustering Explorer")
st.caption("Uses Polars for tabular aggregation and Streamlit + Folium for interactive mapping.")

# Functionality allowing user input in sidebar to determine values such as target area, model name, and resolution
with st.sidebar:
    st.header("Settings")
    place_name = st.text_input("Place", value="Blacksburg, VA")
    resolution = st.slider("H3 resolution", min_value=7, max_value=12, value=11)
    k_ring = st.slider("Neighborhood k-ring", min_value=1, max_value=3, value=1)
    location_weight = st.slider("Location weight", min_value=0.0, max_value=2.0, value=0.65, step=0.05)
    max_epochs = st.slider("Hex2Vec epochs", min_value=5, max_value=30, value=20)
    learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
    batch_size = st.selectbox("Batch size", options=[256, 512, 1024, 2048], index=3)
    n_clusters = st.slider("Clusters (for cluster-count methods)", min_value=2, max_value=12, value=5)
    seed = st.number_input("Seed", min_value=0, max_value=9999, value=42)

    model_name = st.selectbox(
        "Clustering model",
        options=["KMeans", "GMM", "DBSCAN", "HDBSCAN", "Agglomerative", "Spectral", "SOM"],
    )

    run_button = st.button("Run pipeline", type="primary")


# Performing complete data accessing, vectorization, training, clustering, mapping on run button click
if run_button:
    # vectorizing data and training the embeddings with user's input
    with st.spinner("Preparing data and training embeddings..."):
        (
            area,
            loader_gdf,
            regions_gdf,
            joint_gdf,
            features_per_hex,
            embeddings,
            weighted_features,
        ) = train_embeddings(
            place_name=place_name,
            resolution=resolution,
            k_ring=k_ring,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            location_weight=location_weight,
            seed=seed,
        )

    st.success("Embeddings ready.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hexagons", len(regions_gdf))
    with col2:
        st.metric("Embedding dimensions", embeddings.shape[1])
    with col3:
        st.metric("Weighted dimensions", weighted_features.shape[1])

    # Creating feature map 
    st.subheader("OSM feature choropleth")
    selected_feature = st.selectbox(
        "Feature to visualize",
        options=list(features_per_hex.columns),
        index=list(features_per_hex.columns).index("building") if "building" in features_per_hex.columns else 0,
    )

    feature_map = render_feature_map(regions_gdf, features_per_hex, selected_feature)
    folium_to_streamlit(feature_map, height=650)

    # Running clustering with user's desired model
    st.subheader(f"{model_name} clustering")
    with st.spinner(f"Running {model_name}..."):
        labels, meta, sil = run_clustering(model_name, weighted_features, n_clusters)

    meta_cols = st.columns(3)
    meta_cols[0].metric("Unique labels", len(np.unique(labels)))
    meta_cols[1].metric("Silhouette", "N/A" if np.isnan(sil) else f"{sil:.4f}")
    meta_cols[2].metric("Noise points", meta.get("noise_points", 0))

    # Mapping clusters
    cluster_map = render_cluster_map(regions_gdf, labels, cluster_name=f"{model_name.lower()}_cluster")
    folium_to_streamlit(cluster_map, height=700)

    # Allowing user to download cluster assignments
    results_df = pd.DataFrame({
        "region_id": weighted_features.index,
        "cluster": labels,
    })
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download cluster assignments (CSV)",
        data=csv_bytes,
        file_name=f"{model_name.lower()}_{place_name.replace(',', '').replace(' ', '_')}_clusters.csv",
        mime="text/csv",
    )

    # Providing a summary of run
    with st.expander("Run summary"):
        st.json({
            "place": place_name,
            "resolution": resolution,
            "k_ring": k_ring,
            "location_weight": location_weight,
            "model": model_name,
            "model_meta": meta,
            "silhouette_score": None if np.isnan(sil) else sil,
            "n_hexagons": len(regions_gdf),
            "n_features": int(weighted_features.shape[1]),
        })
else:
    st.info("Choose parameters in the sidebar and click **Run pipeline**.")


# -----------------------------------------------------
# Notes for future optimization
# -----------------------------------------------------
# 1. Folium itself is not Polars-native. The speed gain comes from using Polars in
#    the region-feature aggregation step, then converting only the final map frame.
# 2. The biggest runtime cost remains Hex2Vec training and OSM loading, not mapping.
# 3. For even faster apps, precompute embeddings/clusters to parquet and let Streamlit
#    only handle filtering + visualization.
# 4. If you want much faster rendering for large geographies, consider pydeck/deck.gl
#    instead of Folium.
