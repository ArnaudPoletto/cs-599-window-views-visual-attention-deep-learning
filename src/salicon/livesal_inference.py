import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import argparse
import platform
from PIL import Image
import multiprocessing
import lightning.pytorch as pl

from src.utils.random import set_seed
from src.models.livesal import LiveSAL
from src.utils.parser import get_config
from src.utils.file import get_paths_recursive
from src.datasets.salicon_dataset import SaliconDataModule
from src.lightning_models.lightning_model import LightningModel
from src.config import (
    SEED,
    N_WORKERS,
    CONFIG_PATH,
    CHECKPOINTS_PATH,
    TEST_SALICON_PATH,
    PROCESSED_SALICON_PATH,
    FINAL_HEIGHT,
    FINAL_WIDTH,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inferring saliency maps on the SALICON test set."
    )

    parser.add_argument(
        "--config-file-path",
        "-config",
        "-conf",
        "-c",
        type=str,
        default=f"{CONFIG_PATH}/livesal/global_salicon_salicon_challenge.yml",
        help="The path to the config file.",
    )

    parser.add_argument(
        "--checkpoint-file-path",
        "-checkpoint",
        "-cp",
        type=str,
        default=f"{CHECKPOINTS_PATH}/livesal_global_salicon_challenge.ckpt",
        help="The path to the checkpoint file.",
    )

    return parser.parse_args()


def main() -> None:
    if platform.system() != "Windows":
        multiprocessing.set_start_method("forkserver", force=True)
    set_seed(SEED)

    # Parse arguments
    args = parse_arguments()
    config_file_path = args.config_file_path
    checkpoint_file_path = args.checkpoint_file_path

    # Get config parameters
    config = get_config(config_file_path)
    batch_size = int(config["batch_size"])
    splits = tuple(map(float, config["splits"]))
    use_challenge_split = bool(config["use_challenge_split"])
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
    data_module = SaliconDataModule(
        batch_size=batch_size,
        train_split=splits[0],
        val_split=splits[1],
        test_split=splits[2],
        use_challenge_split=use_challenge_split,
        with_transforms=with_transforms,
        n_workers=N_WORKERS,
        seed=SEED,
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
    if not os.path.exists(checkpoint_file_path):
        raise FileNotFoundError(
            f"❌ File {Path(checkpoint_file_path).resolve()} not found."
        )
    lightning_model = LightningModel.load_from_checkpoint(
        checkpoint_path=checkpoint_file_path,
        model=model,
        name="livesal",
        dataset="salicon",
    )
    print(f"✅ Loaded temporal model from {Path(checkpoint_file_path).resolve()}")

    # Get trainer and predict
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
    )

    predictions = trainer.predict(lightning_model, datamodule=data_module)

    output_folder_path = f"{TEST_SALICON_PATH}/predictions"
    for pred in predictions:
        _, global_output, sample_id = pred
        output_path = f"{output_folder_path}/COCO_test2014_{sample_id:012d}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        global_output = global_output.squeeze(0).cpu().numpy()
        global_output = (global_output * 255).astype("uint8")
        global_output = Image.fromarray(global_output)
        global_output = global_output.resize(
            (FINAL_WIDTH, FINAL_HEIGHT)
        )
        global_output.save(output_path)
    print(f"✅ Saved predictions to {Path(output_folder_path).resolve()}")


if __name__ == "__main__":
    main()
