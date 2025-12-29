# Repository Overview

This repository contains a notebook-driven LSTM project along with its supporting data and documentation. The contents are:

- `Taller_proyecto_LSTM.ipynb`: Notebook with 16 cells (12 code) that trains an LSTM model using Keras, pandas, NumPy, and Excel utilities.
- `lstm/`: Python modules extracted from the notebook so the workflow can run outside Jupyter.
- `00_DatosOriginalesLag.xlsx`: Excel dataset referenced by the notebook.
- `Proyecto_Machine_Learning.pdf`: Project report in PDF form.

## Notebook dependencies
The notebook imports the following libraries:

- Data handling: `pandas`, `numpy`, `os`, `warnings`
- Deep learning: `keras` (`Sequential`, `LSTM`, `Dense`, `Dropout`, `Bidirectional`, `Adam`, `EarlyStopping`, `keras.backend as K`)
- Visualization: `matplotlib.pyplot`
- Excel I/O: `openpyxl`, `Workbook`, `load_workbook`, `xlrd`, `xlsxwriter`, `xlwt`

These dependencies come from scanning the code cells directly in the notebook rather than executing it, which is helpful when setting up an environment for reproduction without immediately running the notebook.

## Extracted Python modules
The `lstm` package now mirrors the notebook's logic:

- `data.py` handles Excel loading and univariate sequence splitting.
- `models.py` defines the vanilla, stacked, and bidirectional LSTM builders.
- `metrics.py` computes the squared and absolute errors used for evaluation.
- `experiment.py` provides the `test_lstm` routine plus a CLI-style entry point that runs the five default experiments and writes `predicciones.xlsx` and `resultados.xlsx`.
