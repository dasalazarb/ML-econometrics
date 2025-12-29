"""End-to-end runner that mirrors the original notebook workflow.

This module orchestrates data loading, model construction, experiment execution,
metrics aggregation, and Excel export. It is designed to provide a single entry
point for replicating the training/prediction routine outside of Jupyter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import argparse

import pandas as pd

from lstm import models
from lstm.data import DEFAULT_DATA_PATH
from lstm.experiment import ExperimentConfig, run_experiments, write_outputs


@dataclass
class PipelineConfig:
    """High-level parameters for the full pipeline run."""

    beta: str
    iteracion: int
    num_beta: int
    dias_predecir: int
    n_steps: int
    activations: Sequence[str]
    model_names: Sequence[str]
    output_dir: Path
    data_path: Path


def _resolve_models(model_names: Iterable[str]):
    """Map human-readable model names to builder callables."""
    available = {
        "vanilla": models.vanilla_lstm,
        "stacked": models.stacked_lstm,
        "bidirectional": models.bidirectional_lstm,
    }
    resolved = []
    for name in model_names:
        if name not in available:
            raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(available))}")
        resolved.append(available[name])
    return resolved


def build_configs(cfg: PipelineConfig) -> List[ExperimentConfig]:
    """Create one experiment configuration per (activation, model) pair."""
    configs: List[ExperimentConfig] = []
    for activation in cfg.activations:
        for builder in _resolve_models(cfg.model_names):
            configs.append(
                ExperimentConfig(
                    iteracion=cfg.iteracion,
                    num_beta=cfg.num_beta,
                    dias_predecir=cfg.dias_predecir,
                    n_features=1,
                    n_steps=cfg.n_steps,
                    model_builder=builder,
                    activation=activation,
                    data_path=cfg.data_path,
                    beta=cfg.beta,
                )
            )
    return configs


def run_pipeline(cfg: PipelineConfig) -> pd.DataFrame:
    """Execute the end-to-end process and write Excel outputs."""
    configs = build_configs(cfg)
    results, preds = run_experiments(configs)
    write_outputs(results, preds, cfg.output_dir)
    return results


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM models end-to-end.")
    parser.add_argument("--beta", default="1", help="Beta sheet identifier to load from the Excel file (e.g., '1').")
    parser.add_argument("--iteracion", type=int, default=20, help="Number of sliding-window iterations to train over.")
    parser.add_argument("--num-beta", type=int, default=30, help="Window size used to seed the magic matrix.")
    parser.add_argument("--dias-predecir", type=int, default=10, help="Days ahead to predict for each iteration.")
    parser.add_argument("--n-steps", type=int, default=6, help="Sequence length used as model input.")
    parser.add_argument(
        "--activation",
        nargs="+",
        default=["relu"],
        help="One or more activation functions to evaluate (e.g., relu tanh).",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["vanilla"],
        help="One or more model types to train: vanilla, stacked, bidirectional.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where to write predicciones.xlsx and resultados.xlsx.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to 00_DatosOriginalesLag.xlsx (or equivalent input file).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)

    pipeline_cfg = PipelineConfig(
        beta=args.beta,
        iteracion=args.iteracion,
        num_beta=args.num_beta,
        dias_predecir=args.dias_predecir,
        n_steps=args.n_steps,
        activations=args.activation,
        model_names=args.model,
        output_dir=args.output_dir,
        data_path=args.data_path,
    )

    run_pipeline(pipeline_cfg)


if __name__ == "__main__":
    main()
