# Yacht Speed Prediction Model Package

---

### Quick dectription
This package contains TensorFlow models capable of predicting various speeds and heel angles of a yacht given specific wind velocities and angles of attack. The package includes all necessary functions for making predictions. You can track the process of creating the models by reading the Jupyter notebooks: `dataAnalysis.ipynb`, `makeModel.ipynb`, and `modelAnalysis.ipynb`. Predictions can be made in two different ways: by providing input directly or via an Excel file.

### Package structure
```
model
├── README.md
├── __init__.py
├── core
│   ├── __init__.py
│   ├── best_nn_settings
│   │   │   ├── model_class_0.cpython-310.pyc
│   │   │   └── model_class_1.cpython-310.pyc
│   │   ├── model_class_0.py
│   │   └── model_class_1.py
│   ├── models
│   ├── predict.py
│   ├── predict_excel.py
│   └── predict_input.py
├── data
│   ├── clean
│   └── raw
│       
├── notebooks
│   ├── dataAnalys.ipynb
│   ├── makeModel.ipynb
│   └── modelAnalys.ipynb
├── plots
├── predictions
│   ├── input
│   │   └── input.xlsx
│   └── output
│       └── output.xlsx
├── requirements.txt
└── setup.py
```

##### Most important files and and folders:
* `core`: Main folder containing classes for making predictions and saved TensorFlow models.
* `data`: Contains files with raw and cleaned data.
* `notebooks`: Folder with Jupyter notebooks documenting data analysis, model creation, and analysis.
* `plots`: Contains plots of model performance.
* `predictions`: Used for making predictions with an Excel file. Place your file in the `predictions/input` folder (more about this in the "How to Use" section).

### Installation
To install all required packages, run the following command in the terminal (in `model/` directory):
```bash
pip install .
```

### How to use
The package contains two classes for making predictions:
1. `model.Predictexcel`:
    * Place your Excel file in the `model/predictions/input` folder and name it `input.xlsx`.
    * The class will return a pandas DataFrame with the model predictions and will automatically save the results in the `model/predictions/output` folder.

###### Example usage:
```python
import model

predictor = model.PredictExcel()
result_df = predictor.make_prediction()
```
2. `model.PredictInput:`
    * The class will prompt you to provide input parameters directly in the terminal.
    * It will return a pandas DataFrame with the model predictions.

###### Example usage:
```python
import model

predictor = model.PredictInput()
result_df = predictor.make_prediction()
```

### Input Data Requirements
The Excel file should contain the following headers:
```
'Series date', 'Length Overall', 'Maximum Beam', 'Draft', 'Displacement', 'DLR', 'IMS Division', 'Dynamic Allowance', 'Age Allowance', 'Mainsail measured', 'Mainsail rated', 'Headsail Luffed measured', 'Headsail Luffed rated', 'Symmetric measured', 'Symmetric rated', 'Mizzen measured', 'Mizzen rated', 'Headsail Flying measured', 'Headsail Flying rated', 'Asymmetric measured', 'Asymmetric rated', 'Quad. Mainsail measured', 'Quad. Mainsail rated', 'Mizzen Staysail measured', 'Mizzen Staysail rated'
```

You can read why those headers were chosen in `dataAnalys.ipynb` notebook

### Model performance
You can read about how the model's performance was tested in the `modelAnalysis.ipynb` notebook. Before creating the model, we split 11,000 yacht specifications into training data (80%) and test data (20%). The analysis was performed on the test data.

Our model produces a table like this, based on the provided data (our dataFrame is flattened to 1D from 2D):
| Wind Velocity | 6 kt  | 8 kt  | 10 kt | 12 kt | 14 kt | 16 kt | 20 kt | 24 kt |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Beat Angles   | 348.8 | 453.8 | 453.2 | 443.9 | 501.0 | 498.7 | 488.0 | 500.1 |
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
| Gybe Angles   | 137.5 | 157.5 | 138.6 | 178.6 | 156.5 | 123.3 | 134.1 | 148.8 | 

Average Values for Each Cell:

![Image](/model/plots/realValuesAverage.png)

Average Difference Between Real Yacht Values and Model Predictions:

![Image](/model/plots/modelVsRelity.png)


