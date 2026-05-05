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
