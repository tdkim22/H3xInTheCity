!pip install h3 srai srai[osm] pytorch_lightning torch contextily --quiet

# --- 1. System & Warning Suppression ---
import os
import warnings
import logging
import random

# Filter specific annoying warnings that clutter client outputs
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore" # Suppress at system level

# Silence Pytorch Lightning & Library logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("srai").setLevel(logging.WARNING)

# --- 2. Data Manipulation & Math ---
import numpy as np
import pandas as pd
import geopandas as gpd
import itertools

# --- 3. Geospatial & H3 ---
import h3
from srai.loaders import OSMPbfLoader
from srai.regionalizers import geocode_to_region_gdf, H3Regionalizer
from srai.joiners import IntersectionJoiner
from srai.embedders import Hex2VecEmbedder
from srai.neighbourhoods import H3Neighbourhood
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER # This filter is used for loading specific OSM data
from srai.plotting import plot_regions

# --- 4. Machine Learning & Torch ---
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.preprocessing import StandardScaler

# --- 5. Visualization ---
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- 6. Configuration ---
# Set Matmul precision for Torch
torch.set_float32_matmul_precision('medium')

# Pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)

# Matplotlib inline
%matplotlib inline

print("Libraries loaded. Environment configured for clean output.")

# =====================================================
# 2.1 GLOBAL REPRODUCIBILITY SETUP
# =====================================================
# Setting seeds ensures that the Neural Network (Hex2Vec)
# and other stochastic processes produce the exact same
# results every time you run this notebook.

def set_global_seed(seed=42):
    # 1. Python's built-in random module
    random.seed(seed)

    # 2. NumPy (used for mathematical operations and arrays)
    np.random.seed(seed)

    # 3. PyTorch (used by Hex2Vec for weights initialization)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 4. PyTorch Lightning (manages the training loop)
    # This is the most important one for the SRAI library
    pl.seed_everything(seed, workers=True)

# Apply the seed
set_global_seed(42)
print("Global random seed set to 42 for reproducibility.")

# ---- Change these global variables! ----
RESOLUTION = 10
AREA = "Blacksburg, VA"
ZONING_FILE = "Town_Zoning.shp"

area = geocode_to_region_gdf(AREA)
loader = OSMPbfLoader()
loader_gdf = loader.load(area, HEX2VEC_FILTER)

regionalizer = H3Regionalizer(resolution=RESOLUTION)
regions_gdf = regionalizer.transform(loader_gdf)

joiner = IntersectionJoiner()
joint_gdf = joiner.transform(regions_gdf, loader_gdf)
joint_pd = joint_gdf.reset_index()

feature_columns = list(HEX2VEC_FILTER.keys())
present_cols = [c for c in feature_columns if c in loader_gdf.columns]
binary_feature_df = pd.DataFrame(index=loader_gdf.index)
for col in present_cols:
    binary_feature_df[col] = loader_gdf[col].notna().astype(np.int8)
binary_feature_df.index.name = "feature_id"

merged_df = joint_pd.merge(
    loader_gdf[feature_columns], # Select only the feature columns from loader_gdf
    left_on='feature_id',
    right_index=True, # Merge on loader_gdf's index ('feature_id')
    how='left'
)

features_per_hex = merged_df.groupby('region_id')[feature_columns].sum()
features_per_hex.rename(columns={'region_id_left': 'region_id'}, inplace=True)
regions_gdf = gpd.sjoin(regions_gdf, area, predicate='intersects', how='inner')

coords = []
for h3_idx in regions_gdf.index:
    lat, lng = h3.cell_to_latlng(h3_idx)
    coords.append([lat, lng])

coords_array = np.array(coords)
scaler = StandardScaler()
scaled_coords = scaler.fit_transform(coords_array)

zoning_df = gpd.read_file(ZONING_FILE)
zoning_df = zoning_df.to_crs(regions_gdf.crs)
regions_with_zones_gdf = gpd.sjoin(regions_gdf, 
                                   zoning_df, 
                                   how='inner', 
                                   predicate='intersects')
y_true = regions_with_zones_gdf['Zoning'] # This may need to be changed if you use a different file.

print("Distribution of Zoning Categories:")
print(y_true.value_counts())

regions_with_zoning_for_plot = regions_gdf.join(y_true, how='inner')

# Get unique zoning categories
unique_zones = regions_with_zoning_for_plot['Zoning'].unique()
num_zones = len(unique_zones)

# Create a color map for zoning categories
# Using a qualitative colormap from Matplotlib or generating distinct colors
colors = plt.cm.get_cmap('tab20', num_zones).colors # Use tab20 for up to 20 categories, adjust if more
color_map = {zone: mcolors.to_hex(colors[i]) for i, zone in enumerate(unique_zones)}

# Add color column to the GeoDataFrame
regions_with_zoning_for_plot['color'] = regions_with_zoning_for_plot['Zoning'].map(color_map)

# Get the centroid of the entire area for map centering
map_center = regions_with_zoning_for_plot.unary_union.centroid

# Create a Folium map
m = folium.Map(location=[map_center.y, map_center.x],
               zoom_start=12,
               zoom_control=True,
               zoom_delta=0.25,        # smaller zoom steps
               zoom_snap=0.25,
               tiles='CartoDB Voyager')

# Add choropleth layer for each zoning category
for zone in unique_zones:
    zone_gdf = regions_with_zoning_for_plot[regions_with_zoning_for_plot['Zoning'] == zone]
    folium.GeoJson(
        zone_gdf.to_json(),
        name=f'Zoning: {zone}',
        style_function=lambda feature, color=color_map[zone]: {
            'fillColor': color,
            'color': color,
            'weight': 0.5,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(fields=['Zoning'], aliases=['Zoning Category: '])
    ).add_to(m)

# Add a layer control to toggle zoning layers
folium.LayerControl().add_to(m)
display(m)

# Save folium map
m.save('zoning_map_r10.html')

zoning_clusters = regions_with_zoning_for_plot
zoning_clusters['Zoning'].unique()
zoning_clusters.replace({'Zoning': {'R-4  Low Density Residential': 0}}, inplace=True)
zoning_clusters.replace({'Zoning': {'R-5  Transitional Residential': 1}}, inplace=True)
zoning_clusters.replace({'Zoning': {'PR  Planned Residential': 2}}, inplace=True)
zoning_clusters.replace({'Zoning': {'RD  Research and Development': 3}}, inplace=True)
zoning_clusters.replace({'Zoning': {'DC  Downtown Commercial': 4}}, inplace=True)
zoning_clusters.replace({'Zoning': {'UNIV  University': 5}}, inplace=True)
zoning_clusters.replace({'Zoning': {'IN  Industrial': 6}}, inplace=True)
zoning_clusters.replace({'Zoning': {'O  Office': 7}}, inplace=True)
zoning_clusters.replace({'Zoning': {'RM-48  Medium Density Multiunit Residential': 8}}, inplace=True)
zoning_clusters.replace({'Zoning': {'PMH  Planned Manufactured Home': 9}}, inplace=True)
zoning_clusters.replace({'Zoning': {'GC  General Commercial': 10}}, inplace=True)
zoning_clusters.replace({'Zoning': {'RM-27  Low Density Multiunit Residential': 11}}, inplace=True)
zoning_clusters.replace({'Zoning': {'MXD  Mixed Use Development': 12}}, inplace=True)
zoning_clusters.replace({'Zoning': {'PC  Planned Commercial': 13}}, inplace=True)
zoning_clusters.replace({'Zoning': {'OTR  Old Town Residential': 14}}, inplace=True)
zoning_clusters.replace({'Zoning': {'RR-1  Rural Residential 1': 15}}, inplace=True)
zoning_clusters.replace({'Zoning': {'RR-2  Rural Residential 2': 16}}, inplace=True)
zoning_clusters = zoning_clusters.drop(columns=['geometry', 'region_id_right', 'color'])

zoning_clusters.to_csv(f'zoning_r{RESOLUTION}.csv', index=True)
