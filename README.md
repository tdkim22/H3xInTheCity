# H3x and the City
Using the processed data from water usage metrics to make generalizations of areas and groups.

## Setup Instructions
1. Make sure all packages in `requirements.txt` are installed using:
```python
pip install -r requirements.txt
```
2. Provided all naming conventions are followed, file paths should not need to be changed. Files in `output` and `BlacksburgZoningData` are imported into some scripts and must be loaded into the workspace prior to running the code.
3. Open whichever script you want to run in Google Colab and run cells sequentially.

* Note: code will perform better on a high-performance computer, such as VT's ARC clusters. In this case, it will need to be restructured as such:
```python
def main():
  # Your code here

if __name__ == "__main__":
    main()
```
* Also, you will need to change any calls to the `display("object")` function to `print("object")`.
