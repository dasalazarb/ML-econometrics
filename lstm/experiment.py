"""Experiment runner extracted from the original notebook."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

from lstm.data import DEFAULT_DATA_PATH, open_file, split_sequence
from lstm.metrics import compute_errors
from lstm import models

ModelBuilder = Callable[[str, int, int], object]
legacy_preds: Dict[str, pd.DataFrame] = {}


@dataclass
class ExperimentConfig:
    iteracion: int
    num_beta: int
    dias_predecir: int
    n_features: int
    n_steps: int
    model_builder: ModelBuilder
    activation: str
    data_path: Path | str = DEFAULT_DATA_PATH
    beta: str = "1"


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    predictions: pd.DataFrame
    errors: Tuple[float, float, float, float]


def test_lstm(config: ExperimentConfig) -> ExperimentResult:
    """Train and evaluate a single LSTM configuration, mirroring the notebook logic."""
    dataset = open_file(file_path=config.data_path, beta=config.beta)
    matriz_magica = np.zeros((500, 500))

    yhat_5d: List[float] = []
    y_5d: List[float] = []
    yhat_10d: List[float] = []
    y_10d: List[float] = []

    model = config.model_builder(config.activation, config.n_steps, config.n_features)

    for i in range(config.iteracion):
        matriz_magica[0 : i + config.num_beta, i] = dataset[0 : i + config.num_beta]

        for j in range(config.dias_predecir):
            raw_seq = list(matriz_magica[0 : i + config.num_beta + j, i])
            X, y = split_sequence(raw_seq, config.n_steps)
            X = X.reshape((X.shape[0], X.shape[1], config.n_features))
            early_stop = EarlyStopping(monitor="loss", patience=2, verbose=0)
            model.fit(X, y, epochs=200, verbose=0, batch_size=3, callbacks=[early_stop])

            x_input = np.array(raw_seq[-config.n_steps :])
            x_input = x_input.reshape((1, config.n_steps, config.n_features))
            yhat = model.predict(x_input, verbose=0)

            matriz_magica[i + config.num_beta + j, i] = yhat[0][0]
            if j == 4:
                yhat_5d.append(yhat[0][0])
                y_5d.append(dataset[i + config.num_beta + j])
            elif j == 9:
                yhat_10d.append(yhat[0][0])
                y_10d.append(dataset[i + config.num_beta + j])

    param_model = {
        "iter": config.iteracion,
        "numBeta": config.num_beta,
        "diasPredecir": config.dias_predecir,
        "n_features": config.n_features,
        "n_steps": config.n_steps,
        "model": model,
        "activation": config.activation,
    }
    param_pred = {"yhat_5d": yhat_5d, "y_5d": y_5d, "yhat_10d": yhat_10d, "y_10d": y_10d}

    errors = compute_errors(param_model, param_pred)
    predictions = pd.DataFrame(np.array([yhat_5d, y_5d, yhat_10d, y_10d]).T, columns=["yhat_5d", "y_5d", "yhat_10d", "y_10d"])

    return ExperimentResult(config=config, predictions=predictions, errors=errors)


# Preserve the original notebook-friendly casing.
def testLSTM(iteracion: int, numBeta: int, diasPredecir: int, n_features: int, n_steps: int,
             model: ModelBuilder, activation: str, Beta: str, resultados: pd.DataFrame, num: int,
             data_path: Path | str = DEFAULT_DATA_PATH):
    config = ExperimentConfig(
        iteracion=iteracion,
        num_beta=numBeta,
        dias_predecir=diasPredecir,
        n_features=n_features,
        n_steps=n_steps,
        model_builder=model,
        activation=activation,
        data_path=data_path,
        beta=Beta,
    )
    result = test_lstm(config)
    errors = result.errors

    results_iter = pd.DataFrame([
        [
            config.iteracion,
            config.num_beta,
            config.dias_predecir,
            config.n_features,
            config.n_steps,
            config.model_builder.__name__,
            config.activation,
            errors[0],
            errors[1],
            errors[2],
            errors[3],
        ]
    ], columns=[
        "iter",
        "numBeta",
        "diasPredecir",
        "n_features",
        "n_steps",
        "model",
        "activation",
        "error_5d_2",
        "error_10d_2",
        "error_5d_abs",
        "error_10d_abs",
    ])

    legacy_preds[f"resultado_{num}"] = result.predictions
    resultados = pd.concat([resultados, results_iter], axis=1)
    num += 1
    return result.model, resultados, num


def run_experiments(configs: Iterable[ExperimentConfig]) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Execute all provided experiments and aggregate metrics/predictions."""
    results_frames = []
    preds: Dict[str, pd.DataFrame] = {}

    for idx, config in enumerate(configs):
        experiment_result = test_lstm(config)
        errors = experiment_result.errors
        results_frames.append(
            pd.DataFrame(
                [
                    [
                        config.iteracion,
                        config.num_beta,
                        config.dias_predecir,
                        config.n_features,
                        config.n_steps,
                        config.model_builder.__name__,
                        config.activation,
                        errors[0],
                        errors[1],
                        errors[2],
                        errors[3],
                    ]
                ],
                columns=[
                    "iter",
                    "numBeta",
                    "diasPredecir",
                    "n_features",
                    "n_steps",
                    "model",
                    "activation",
                    "error_5d_2",
                    "error_10d_2",
                    "error_5d_abs",
                    "error_10d_abs",
                ],
            )
        )
        preds[f"resultado_{idx}"] = experiment_result.predictions

    combined_results = pd.concat(results_frames, ignore_index=True)
    return combined_results, preds


def write_outputs(results: pd.DataFrame, preds: Dict[str, pd.DataFrame], output_dir: Path | str = Path(".")) -> None:
    """Persist predictions and summary metrics to Excel files, matching the notebook outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_dir / "predicciones.xlsx", engine="xlsxwriter") as writer:
        for sheet_name, df in preds.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    results.to_excel(output_dir / "resultados.xlsx", index=False)


DEFAULT_CONFIGS = [
    ExperimentConfig(iteracion=30, num_beta=30, dias_predecir=10, n_features=1, n_steps=5, model_builder=models.bidirectional_lstm, activation="tanh"),
    ExperimentConfig(iteracion=30, num_beta=30, dias_predecir=10, n_features=1, n_steps=10, model_builder=models.bidirectional_lstm, activation="tanh"),
    ExperimentConfig(iteracion=30, num_beta=30, dias_predecir=10, n_features=1, n_steps=15, model_builder=models.bidirectional_lstm, activation="tanh"),
    ExperimentConfig(iteracion=30, num_beta=30, dias_predecir=10, n_features=1, n_steps=20, model_builder=models.bidirectional_lstm, activation="tanh"),
    ExperimentConfig(iteracion=30, num_beta=30, dias_predecir=10, n_features=1, n_steps=25, model_builder=models.bidirectional_lstm, activation="tanh"),
]


if __name__ == "__main__":
    np.random.seed(7)
    results, preds = run_experiments(DEFAULT_CONFIGS)
    write_outputs(results, preds)
