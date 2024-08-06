# Yacht Speed and Heel Angle Prediction

This project aims to predict various speeds and heel angles of a yacht given specific wind velocities and angles of attack. The predictions are organized into a table as follows:

| Wind Velocity | 6 kt  | 8 kt  | 10 kt | 12 kt | 14 kt | 16 kt | 20 kt | 24 kt |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Beat VMG      | 885.0 | 719.0 | 651.0 | 624.9 | 610.3 | 600.3 | 589.9 | 598.0 |
| 52°           | 580.3 | 487.0 | 456.0 | 444.0 | 436.5 | 430.9 | 423.5 | 423.2 |
| 60°           | 549.9 | 470.9 | 443.2 | 430.3 | 421.8 | 415.2 | 406.7 | 404.2 |
| 75°           | 530.7 | 461.3 | 431.8 | 412.7 | 400.6 | 391.8 | 378.8 | 372.4 |
| 90°           | 544.8 | 466.5 | 432.7 | 407.1 | 387.2 | 372.2 | 351.0 | 337.0 |
| 110°          | 546.3 | 457.7 | 421.9 | 394.3 | 374.9 | 358.6 | 331.8 | 300.6 |
| 120°          | 558.3 | 459.9 | 418.5 | 388.4 | 362.8 | 339.2 | 303.9 | 282.4 |
| 135°          | 629.1 | 490.6 | 434.1 | 394.8 | 359.8 | 327.2 | 280.1 | 233.5 |
| 150°          | 748.7 | 585.1 | 489.9 | 435.7 | 400.4 | 366.4 | 298.1 | 232.0 |
| Run VMG       | 864.5 | 675.6 | 565.6 | 502.3 | 462.4 | 423.0 | 344.2 | 267.8 |

Note: The 2D table above is flattened into a 1D format for processing in the model.

## Project Structure

1. `dataAnalys.ipynb`:
   - This notebook performs an analysis of the data.
   - Cleans and processes the data.
   - Creates necessary directories and files for further processing.

2. `makeModel.ipynb`:
   - Uses PyTorch to create various neural network models.
   - Trains multiple models and selects the best settings.
   - Creates 96 models to predict the table.

3. `predict.py`:
   - Accepts input parameters and returns the predictions in Excel format.
   - Has two usage options:
     - **Default**: The program prompts for ship data in the terminal.
     - **With `-f` flag**: The program reads data from a specified Excel file. The Excel file should contain the following columns:
       ```plaintext
       ['Series date', 'Length Overall', 'Maximum Beam', 'Draft', 'Displacement', 'DLR', 'IMS Division', 'Dynamic Allowance', 'Age Allowance', 'Mainsail measured', 'Mainsail rated', 'Headsail Luffed measured', 'Headsail Luffed rated', 'Symmetric measured', 'Symmetric rated', 'Mizzen measured', 'Mizzen rated', 'Headsail Flying measured', 'Headsail Flying rated', 'Asymmetric measured', 'Asymmetric rated', 'Quad. Mainsail measured', 'Quad. Mainsail rated', 'Mizzen Staysail measured', 'Mizzen Staysail rated']
       ```

## Usage

1. Run `dataAnalys.ipynb` to download, clean, and prepare the data. This notebook will create the necessary directories and files locally.
2. Use `predict.py` to get predictions:
   - **Default Mode**:
     ```bash
     python predict.py
     ```
     The program will prompt for ship data in the terminal.
   - **File Mode**:
     ```bash
     python predict.py -f input.xlsx
     ```
     The program will read ship data from `predictions/input/input.xlsx`. The directory `predictions/input` is created by `dataAnalys.ipynb`.

3. During the execution of `predict.py`, you will be prompted to provide a name for the output file in the terminal. For example, if you enter `results`, the output will be saved as `results.xlsx` in the `predictions/output` directory.

## Required Libraries

To run this project, you'll need to install the following libraries:

- `requests`: For making HTTP requests to fetch data from web sources.
- `json`: For parsing and working with JSON data.
- `os`: For interacting with the operating system, such as file and directory manipulation.
- `numpy`: For numerical operations and array handling.
- `bs4` (BeautifulSoup): For parsing HTML and XML documents.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For creating static, animated, and interactive visualizations in Python.
- `torch` and `torch.nn`: For building and training neural network models using PyTorch.
- `re`: For regular expression operations.
- `sys`: For accessing system-specific parameters and functions.

You can install the required libraries using `pip` with the following command:

```bash
pip install requests beautifulsoup4 numpy pandas matplotlib torch
```

## Additional Resources

For more details on web scraping and collecting yacht data used for training models, visit my [web scraping project](https://github.com/Hinski2/webScraperForYachtsData).