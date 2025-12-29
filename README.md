# End-to-end LSTM Pipeline

This repository contains a modular version of the LSTM experiment originally developed in `Taller_proyecto_LSTM.ipynb`.
## Project structure
- `lstm/` reusable code:
  - `data.py`: loads the `Nominales_B{beta}_Y` series from `00_DatosOriginalesLag.xlsx` and builds sequences.
  - `models.py`: defines model variants (vanilla, stacked, bidirectional).
  - `metrics.py`: computes squared and absolute errors at 5 and 10 days.
  - `experiment.py`: runs a single experiment (train ➜ predict ➜ measure ➜ return results).
  - `pipeline.py`: orchestrates multiple experiments and saves Excel files with metrics and predictions.
- `00_DatosOriginalesLag.xlsx`: expected data source (not modified by the pipeline).

## Requirements
- Python 3.10+ and a virtual environment (recommended).
- Key dependencies: `tensorflow`/`keras`, `pandas`, `numpy`, `xlsxwriter`.

Quick installation example:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow pandas numpy xlsxwriter
```

## How to run the pipeline
The end-to-end pipeline reads the Excel file, trains the model/activation combinations you specify, computes errors, and writes two files to the output folder: `predicciones.xlsx` (one sheet per combination) and `resultados.xlsx` (metrics summary).

Minimal command from the repo root:
```bash
python -m lstm.pipeline \
  --beta 1 \
  --iteracion 20 \
  --num-beta 30 \
  --dias-predecir 10 \
  --n-steps 6 \
  --activation relu tanh \
  --model vanilla stacked \
  --output-dir outputs
```

Main arguments:
- `--beta`: Excel sheet to use (e.g., `1` for `Nominales_B1_Y`).
- `--iteracion`: number of sliding windows to train/evaluate.
- `--num-beta`: size of the initial window that feeds the "magic" matrix.
- `--dias-predecir`: prediction horizon in days.
- `--n-steps`: input sequence length for the LSTM.
- `--activation`: list of activations to try (e.g., `relu tanh`).
- `--model`: list of architectures (`vanilla`, `stacked`, `bidirectional`).
- `--output-dir`: path where the Excel files are written.

## What the pipeline does internally
1. **Build configurations**: generates an `ExperimentConfig` for each requested combination of model and activation.
2. **Run experiments**: each configuration trains its own model over sliding windows and predicts 5- and 10-day horizons.
3. **Metrics**: computes aggregated squared and absolute errors.
4. **Export**: saves per-configuration predictions and a tabular metrics summary.

## Tips for use
- Ensure `00_DatosOriginalesLag.xlsx` is in the repo root, or pass `--data-path` with the correct location.
- Tune `--iteracion` and `--dias-predecir` to match your hardware capacity: higher values increase training time.
- You can reduce `--model` and `--activation` to a single value for quick experiments.

With these steps you can reproduce and extend the experiment without relying on the original notebook.
