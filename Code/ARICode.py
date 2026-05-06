import re
import json
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import shapely.geometry
import shapely.wkt

# Function used later in this script
def wkt_to_shapely(wkt_string):
    """Converts a WKT string to a shapely geometry object"""
    try:
        return shapely.wkt.loads(wkt_string)
    except Exception:
        return None

# Global variables: Change these for different outputs
mod = 'SOM'
res = 12 
kring = 3

# Defining and processing the input file
mod_lower = mod.lower()
html_file_path = f'/content/{mod}_K{kring}_R{res}.html'
cluster_name = f'{mod_lower}_cluster'
html_cluster_data = []

try:
    with open(html_file_path, 'r') as f:
        html_content = f.read()
    json_matches = re.findall(r'geo_json_[a-f0-9]+_add\((\{.*?\})\);', html_content, re.DOTALL)
    if json_matches:
        for json_string in json_matches: 
            geo_json_data = json.loads(json_string)
            if 'features' in geo_json_data:
                for feature in geo_json_data['features']:
                    feature_id = None
                    if 'properties' in feature and 'region_id' in feature['properties']:
                        feature_id = feature['properties']['region_id']
                    elif 'id' in feature:
                        feature_id = feature['id']
                        
                    # Converts geometry of each feature to WKT (Well-Known Text)
                    feature_geometry_wkt = None
                    if 'geometry' in feature and feature['geometry']:
                        try:
                            shapely_geometry = shapely.geometry.shape(feature['geometry'])
                            feature_geometry_wkt = shapely_geometry.wkt
                        except Exception as e:
                            print(f"Warning: Error converting geometry to WKT for feature {feature_id}: {e}")

                    # Appends each hex to the DataFrame with its corresponding cluster and geometry
                    if 'properties' in feature and cluster_name in feature['properties']:
                        html_cluster_data.append({
                            'feature_id': feature_id,
                            cluster_name: feature['properties'][cluster_name],
                            'geometry_wkt': feature_geometry_wkt 
                        })
            else:
                print("Warning: GeoJSON data has no 'features' key in one of the blocks.")
    else:
        print("Could not find GeoJSON data pattern in HTML file.")

# Catching possible errors
except FileNotFoundError:
    print(f"Error: HTML file not found at {html_file_path}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from HTML: {e}")
except Exception as e:
    print(f"An unexpected error occurred during HTML processing: {e}")

# Converts the HTML data into a DataFrame
if html_cluster_data:
    df_html_clusters = pd.DataFrame(html_cluster_data)
    print(f"Successfully extracted {len(df_html_clusters)} 'hc_cluster' labels and their IDs and geometries from HTML.")
    print("First 10 HTML cluster entries:")
    display(df_html_clusters.head(10))
else:
    print("No 'hc_cluster' labels extracted from HTML.")
df_html_clusters['geometry_obj'] = df_html_clusters['geometry_wkt'].apply(wkt_to_shapely)
df_html_clusters = df_html_clusters.drop(columns=['geometry_wkt'])

# Reads in the zoning CSV
zoning = pd.read_csv(f'/content/zoning_r{res}.csv')
# Merges the HTML cluster data and zoning data based on hex
full = pd.merge(zoning, df_html_clusters, left_on='region_id', right_on='feature_id')

# Displays ARI
adjusted_rand_score(full[cluster_name], full['Zoning'])
