import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
import argparse
import torch.nn as nn
from typing import List

from src.losses.mse import MSELoss
from src.utils.random import set_seed
from src.models.tempsal import TempSAL
from src.utils.parser import get_config
from src.losses.kl_div import KLDivLoss
from src.losses.combined import CombinedLoss
from src.utils.file import get_paths_recursive
from src.trainers.tempsal_trainer import TempSALTrainer
from src.datasets.salicon_dataset import get_dataloaders
from src.losses.correlation_coefficient import CorrelationCoefficientLoss
from src.config import (
    SEED,
    DEVICE,
    N_WORKERS,
    CONFIG_PATH,
    PROCESSED_SALICON_PATH,
)


def get_model(
    freeze_encoder: bool,
    hidden_channels_list: List[int],
) -> nn.Module:
    return TempSAL(
        freeze_encoder=freeze_encoder,
        hidden_channels_list=hidden_channels_list,
    ).to(DEVICE)


def get_criterion() -> nn.Module:
    kl_loss = KLDivLoss(temperature=1.0, eps=1e-7)
    corr_loss = CorrelationCoefficientLoss(eps=1e-7)
    mse_loss = MSELoss()
    criterion = CombinedLoss(
        {
            "kl": (kl_loss, 1.0),  # TODO: remove hardocded values
            "corr": (corr_loss, 1.0),  # TODO: remove hardocded values
            "mse": (mse_loss, 0.0), # TODO: remove hardocded values
        }
    )

    return criterion


def get_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> nn.Module:
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def get_trainer(
    model: nn.Module,
    criterion: nn.Module,
    accumulation_steps: int,
    evaluation_steps: int,
    use_scaler: bool,
) -> TempSALTrainer:
    return TempSALTrainer(
        model=model,
        criterion=criterion,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        use_scaler=use_scaler,
        name=f"tempsal",
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process dataset sequences.")

    parser.add_argument(
        "--config-file-path",
        "-config",
        "-conf",
        "-c",
        type=str,
        default=f"{CONFIG_PATH}/tempsal/temporal.yml",
        help="The path to the config file.",
    )

    return parser.parse_args()


def main() -> None:
    set_seed(SEED)

    # Parse arguments
    args = parse_arguments()
    config_file_path = args.config_file_path

    # Get config parameters
    config = get_config(config_file_path)
    n_epochs = int(config["n_epochs"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    batch_size = int(config["batch_size"])
    accumulation_steps = int(config["accumulation_steps"])
    evaluation_steps = int(config["evaluation_steps"])
    splits = tuple(map(float, config["splits"]))
    save_model = bool(config["save_model"])
    use_scaler = bool(config["use_scaler"])
    with_transforms = bool(config["with_transforms"])
    freeze_encoder = bool(config["freeze_encoder"])
    hidden_channels_list = list(map(int, config["hidden_channels_list"]))
    print(f"✅ Using config file at {Path(config_file_path).resolve()}")

    # Get dataloaders, model, criterion, optimizer, and trainer
    sample_folder_paths = get_paths_recursive(
        folder_path=PROCESSED_SALICON_PATH, match_pattern="*", path_type="d"
    )
    train_loader, val_loader, _ = get_dataloaders(
        sample_folder_paths=sample_folder_paths,
        with_transforms=with_transforms,
        batch_size=batch_size,
        train_split=splits[0],
        val_split=splits[1],
        test_split=splits[2],
        train_shuffle=True,
        n_workers=N_WORKERS,
        seed=SEED,
    )
    model = get_model(
        freeze_encoder=freeze_encoder,
        hidden_channels_list=hidden_channels_list,
    )
    criterion = get_criterion()
    optimizer = get_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    trainer = get_trainer(
        model,
        criterion=criterion,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        use_scaler=use_scaler,
    )

    # Train the model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        save_model=save_model,
    )


if __name__ == "__main__":
    main()
