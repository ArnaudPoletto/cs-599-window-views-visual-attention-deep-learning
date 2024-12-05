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
from src.models.disjoint_simple_net import DisjointSimpleNet
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
        default=f"{CONFIG_PATH}/disjoint_simple_net/global_salicon_challenge.yml",
        help="The path to the config file.",
    )

    parser.add_argument(
        "--checkpoint-file-path",
        "-checkpoint",
        "-cp",
        type=str,
        default=f"{CHECKPOINTS_PATH}/disjoint_simple_net_global_salicon_challenge.ckpt",
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
    freeze_encoder = bool(config["freeze_encoder"])
    freeze_temporal_pipeline = bool(config["freeze_temporal_pipeline"])
    hidden_channels_list = list(map(int, config["hidden_channels_list"]))
    output_type = str(config["output_type"])
    dropout_rate = float(config["dropout_rate"])
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
    model = DisjointSimpleNet(
        freeze_encoder=freeze_encoder,
        freeze_temporal_pipeline=freeze_temporal_pipeline,
        hidden_channels_list=hidden_channels_list,
        dropout_rate=dropout_rate,
        output_type=output_type,
    )

    if not os.path.exists(checkpoint_file_path):
        raise FileNotFoundError(
            f"❌ File {Path(checkpoint_file_path).resolve()} not found."
        )
    lightning_model = LightningModel.load_from_checkpoint(
        checkpoint_path=checkpoint_file_path,
        model=model,
        name="disjoint_simple_net",
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
        global_output = global_output.resize((FINAL_WIDTH, FINAL_HEIGHT))
        global_output.save(output_path)
    print(f"✅ Saved predictions to {Path(output_folder_path).resolve()}")


if __name__ == "__main__":
    main()
