# H3x and the City
Using the processed data from water usage metrics to make generalizations of areas and groups.

## Setup Instructions
1. Make sure all packages in `requirements.txt` are installed using:
```python
pip install -r requirements.txt
```
2. Provided all naming conventions are followed, file paths should not need to be changed. Files in `output` and `BlacksburgZoningData` are imported into some scripts and must be loaded into the workspace prior to running the code.
3. Open whichever script you want to run in Google Colab and run cells sequentially.

  Note: code will perform better on a high-performance computer, such as VT's ARC clusters. In this case, it will need to be restructured as such:
```python
def main():
  # Your code here

if __name__ == "__main__":
    main()
```
  Files will need to be saved differently, too. Instead of  `cluster_map.save(f"{model_name}_K{k_ring}_R{resolution}.html")`, use:
```python
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs") # Creates a folder in your ARC files called "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
cluster_map.save(os.path.join(OUTPUT_DIR, f"{model_name}_K{k_ring}_R{resolution}.html"))
```
  Also, you will need to change any calls to the `display("object")` function to `print("object")`.
To run your code in the terminal, navigate to the folder your script is in by typing `cd folder_name` in the command line. Then, run `python your_code.py`. To install packages permanently, run `pip install --user package_name`.

## Output Folder Description
We ran seven different clustering algorithms (Agglomerative, DBSCAN, GMM, HDBSCAN, KMeans, SOM, and Spectral) on different versions of our Hex2Vec data. Each algorithm was run on hex resolutions of 10, 11, and 12 and trained using 1, 2, and 3 k-rings (rings of hexes around the target hexagon). Some clustering algorithms would not run resolution 12, even on ARC, so resolution 12 data is unavailable for Spectral and Agglomerative clustering.
Each file in the output folder is an HTML file displaying the derived clusters of that algorithm. To view it, paste the code into an HTML viewer *or* download it, open Files, and double click the file for it to open in your browser.
The naming convention is as follows: `{AlgorithmName}_K{NumberOfK-Rings}_R{Resolution}`.
## BlacksburgZoningData Folder Description

## Code Folder Description
### ARICode.py
Computes the Adjusted Rand Index (ARI) between a clustering algorithm's cluster assignments and the Blacksburg municipal zoning data.
### FinalClusterCode.py




##Clustering Output RAG Instructions and Details
One of the RAG pipelines was built with Neo4j (the database) and Qwen2.5 (the LLM model). This runs out of a Jupyter Notebook. Prior to running, install Neo4j's desktop app. Make sure to follow instructions within the notebook for details on a database being initialized. Install Ollama and run lines in terminal for Qwen's installation when stated in the notebook. Run cells in order. Be weary of Section 9, and only run if Neo4j has be recently initialized and you are uzing a new neo4j run. (Note: if running the cell below section 9 throws errors, you likely don't need to run it).
