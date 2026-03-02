# Geospatial Classification Framework - H3 & Hex2Vec

## Setup Instructions
1. Unzip this folder.
2. Install the required Python libraries:
   pip install -r requirements.txt
3. Ensure the zoning file is in the same folder:
   - Zoning_Districts_h3_res12_20250728_104611.parquet
4. Open `Final_Code.ipynb` in Jupyter Notebook or JupyterLab.
5. Run the cells sequentially.

## Hardware Note
This pipeline runs faster with a GPU (NVIDIA RTX recommended) but will work on a standard CPU (though training will be slower).