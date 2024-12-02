import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import time
import torch
import argparse
import platform
import multiprocessing
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils.random import set_seed
from src.models.livesal import LiveSAL
from src.utils.parser import get_config
from src.utils.file import get_paths_recursive
from src.datasets.dhf1k_dataset import DHF1KDataModule
from src.datasets.salicon_dataset import SaliconDataModule
from src.lightning_models.lightning_model import LightningModel
from src.config import (
    SEED,
    N_WORKERS,
    MODELS_PATH,
    CONFIG_PATH,
    CHECKPOINTS_PATH,
    PROCESSED_DHF1K_PATH,
    PROCESSED_SALICON_PATH,
)


def _get_data_module(
    dataset: str,
    batch_size: int,
    train_split: float,
    val_split: float,
    test_split: float,
    use_challenge_split: bool,
    with_transforms: bool,
) -> SaliconDataModule | DHF1KDataModule:
    """
    Get the data module for the dataset.

    Args:
        dataset (str): The dataset to use.
        batch_size (int): The batch size.
        train_split (float): The train split.
        val_split (float): The validation split.
        test_split (float): The test split.
        with_transforms (bool): Whether to use transforms.

    Returns:
        Any: The data module.
    """
    if dataset == "salicon":
        sample_folder_paths = get_paths_recursive(
            folder_path=PROCESSED_SALICON_PATH, match_pattern="*", path_type="d"
        )
        data_module = SaliconDataModule(
            sample_folder_paths=sample_folder_paths,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            use_challenge_split=use_challenge_split,
            with_transforms=with_transforms,
            n_workers=N_WORKERS,
            seed=SEED,
        )
    elif dataset == "dhf1k":
        sample_folder_paths = get_paths_recursive(
            folder_path=PROCESSED_DHF1K_PATH, match_pattern="*", path_type="d"
        )
        data_module = DHF1KDataModule(
            sample_folder_paths=sample_folder_paths,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            with_transforms=with_transforms,
            n_workers=N_WORKERS,
            seed=SEED,
        )
    else:
        raise ValueError(f"❌ Unknown dataset {dataset}.")

    return data_module


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
        default=f"{CONFIG_PATH}/livesal/default.yml",
        help="The path to the config file.",
    )

    parser.add_argument(
        "--n-nodes",
        "-n",
        type=int,
        default=1,
        help="The number of nodes to use for distributed training.",
    )

    return parser.parse_args()


def main() -> None:
    """
    The main function to train the LiveSAL model.
    """
    if platform.system() != "Windows":
        multiprocessing.set_start_method("forkserver", force=True)
    set_seed(SEED)

    # Parse arguments
    args = parse_arguments()
    config_file_path = args.config_file_path
    n_nodes = args.n_nodes

    # Get config parameters
    config = get_config(config_file_path)
    dataset = str(config["dataset"])
    n_epochs = int(config["n_epochs"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    batch_size = int(config["batch_size"])
    evaluation_steps = int(config["evaluation_steps"])
    splits = tuple(map(float, config["splits"]))
    use_challenge_split = bool(config["use_challenge_split"])
    save_model = bool(config["save_model"])
    with_transforms = bool(config["with_transforms"])
    image_n_levels = int(config["image_n_levels"])
    hidden_channels = int(config["hidden_channels"])
    neighbor_radius = int(config["neighbor_radius"])
    n_iterations = int(config["n_iterations"])
    with_graph_processing = bool(config["with_graph_processing"])
    freeze_encoder = bool(config["freeze_encoder"])
    freeze_temporal_pipeline = bool(config["freeze_temporal_pipeline"])
    depth_integration = str(config["depth_integration"])
    output_type = str(config["output_type"])
    dropout_rate = float(config["dropout_rate"])
    with_graph_edge_features = bool(config["with_graph_edge_features"])
    with_graph_positional_embeddings = bool(config["with_graph_positional_embeddings"])
    with_graph_directional_kernels = bool(config["with_graph_directional_kernels"])
    with_depth_information = bool(config["with_depth_information"])
    with_checkpoint = bool(config["with_checkpoint"])
    print(f"✅ Using config file at {Path(config_file_path).resolve()}")

    # Get dataset
    data_module = _get_data_module(
        dataset=dataset,
        batch_size=batch_size,
        train_split=splits[0],
        val_split=splits[1],
        test_split=splits[2],
        use_challenge_split=use_challenge_split,
        with_transforms=with_transforms,
    )

    # Get model
    model = LiveSAL(
        image_n_levels=image_n_levels,
        freeze_encoder=freeze_encoder,
        freeze_temporal_pipeline=freeze_temporal_pipeline,
        hidden_channels=hidden_channels,
        neighbor_radius=neighbor_radius,
        n_iterations=n_iterations,
        depth_integration=depth_integration,
        output_type=output_type,
        dropout_rate=dropout_rate,
        with_graph_processing=with_graph_processing,
        with_graph_edge_features=with_graph_edge_features,
        with_graph_positional_embeddings=with_graph_positional_embeddings,
        with_graph_directional_kernels=with_graph_directional_kernels,
        with_depth_information=with_depth_information,
    )
    if with_checkpoint:
        checkpoint_file_path = f"{CHECKPOINTS_PATH}/livesal_temporal.ckpt"
        if not os.path.exists(checkpoint_file_path):
            raise FileNotFoundError(f"❌ File {Path(checkpoint_file_path).resolve()} not found.")
        lightning_model = LightningModel.load_from_checkpoint(
            checkpoint_path=checkpoint_file_path,
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name="livesal",
            dataset=dataset,
        )
    else:
        lightning_model = LightningModel(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name="livesal",
            dataset=dataset,
        )

    # Get trainer and train
    wandb_name = f"{time.strftime('%Y%m%d-%H%M%S')}_livesal"
    wandb_logger = WandbLogger(
        project="thesis",
        name=wandb_name,
        config=config,
    )

    if save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{MODELS_PATH}/livesal/{wandb_name}",
            filename="{epoch}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        callbacks = [checkpoint_callback]
    else:
        callbacks = []

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="gpu",
        devices=-1,
        num_nodes=n_nodes,
        precision=32,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        val_check_interval=evaluation_steps,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lightning_model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    main()
