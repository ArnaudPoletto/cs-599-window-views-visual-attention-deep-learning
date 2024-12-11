import os
import torch
from torch import nn
from typing import List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models.image_encoder import ImageEncoder
from src.models.image_decoder import ImageDecoder
from src.models.spatio_temporal_mixing_module import SpatioTemporalMixingModule
from src.config import SEQUENCE_LENGTH


class TempSAL(nn.Module):
    """
    A spatio-temporal saliency prediction model estimating both temporal and global saliency maps using
    a shared image encoder and separate image decoders. A final spatio-temporal mixing module combines
    the temporal and global features to predict the final global saliency map.
    From: https://ivrl.github.io/Tempsal/

    Args:
        freeze_encoder (bool): Whether to freeze the encoder's parameters.
        hidden_channels_list (List[int]): The number of hidden channels to use in the decoders.

    Returns:
        torch.Tensor: The temporal and global saliency maps.
    """

    def __init__(
        self,
        freeze_encoder: bool,
        freeze_temporal_pipeline: bool,
        hidden_channels_list: List[int],
        output_type: str,
        dropout_rate: float,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize the TempSAL model.

        Args:
            freeze_encoder (bool): Whether to freeze the encoder's parameters.
            hidden_channels_list (List[int]): The number of hidden channels to use in the decoders.
        """
        if output_type not in ["temporal", "global"]:
            raise ValueError(f"❌ Invalid output type: {output_type}")
        if freeze_temporal_pipeline and output_type == "temporal":
            raise ValueError(
                "❌ Cannot freeze the temporal pipeline when output type is temporal."
            )

        super(TempSAL, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.freeze_temporal_pipeline = freeze_temporal_pipeline
        self.hidden_channels_list = hidden_channels_list
        self.output_type = output_type
        self.dropout_rate = dropout_rate
        self.eps = eps

        # Get normalization parameters for encoder inputs
        self.register_buffer(
            "image_mean",
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.image_encoder = ImageEncoder(
            freeze=freeze_encoder or freeze_temporal_pipeline,
        )

        self.temporal_decoder = ImageDecoder(
            features_channels_list=self.image_encoder.feature_channels_list,
            hidden_channels_list=hidden_channels_list,
            output_channels=SEQUENCE_LENGTH,
            dropout_rate=dropout_rate,
            with_final_sigmoid=False,
        )
        self.global_decoder = ImageDecoder(
            features_channels_list=self.image_encoder.feature_channels_list,
            hidden_channels_list=hidden_channels_list,
            output_channels=1,
            dropout_rate=dropout_rate,
            with_final_sigmoid=False,
        )
        self.spatio_temporal_mixing_module = SpatioTemporalMixingModule(
            hidden_channels_list=hidden_channels_list,
            feature_channels_list=self.image_encoder.feature_channels_list,
            dropout_rate=dropout_rate,
        )

        self.sigmoid = nn.Sigmoid()

        if freeze_temporal_pipeline:
            for param in self.temporal_decoder.parameters():
                param.requires_grad = False

        if output_type == "temporal":
            for param in self.global_decoder.parameters():
                param.requires_grad = False
            for param in self.spatio_temporal_mixing_module.parameters():
                param.requires_grad = False

    def _normalize_input(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        x = x.clone()
        normalized_x = (x - mean) / (std + self.eps)

        return normalized_x

    def _normalize_spatial_dimensions(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1)
        x = x / (x.max(dim=2, keepdim=True)[0] + self.eps)
        x = x.view(batch_size, channels, height, width)

        return x

    def _forward_temporal_pipeline(self, x: torch.Tensor) -> torch.Tensor:
        # Encode the input image
        x_image = self._normalize_input(x, self.image_mean, self.image_std)
        encoded_features_list = self.image_encoder(x_image)

        # Decode temporal features and get temporal output
        temporal_features = self.temporal_decoder(encoded_features_list)
        temporal_output = self.sigmoid(temporal_features)
        temporal_output = self._normalize_spatial_dimensions(temporal_output)

        return encoded_features_list, temporal_features, temporal_output

    def _forward_global_pipeline(
        self, encoded_features_list: torch.Tensor, temporal_features: torch.Tensor
    ) -> torch.Tensor:
        global_features = self.global_decoder(encoded_features_list)
        global_output = self.spatio_temporal_mixing_module(
            encoded_features_list=encoded_features_list,
            temporal_features=temporal_features,
            global_features=global_features,
        )
        global_output = self._normalize_spatial_dimensions(global_output).squeeze(1)

        return global_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TempSAL model.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The temporal and global saliency maps.
        """
        if self.output_type == "global":
            encoded_features_list, temporal_features, temporal_output = (
                self._forward_temporal_pipeline(x)
            )
            global_output = self._forward_global_pipeline(
                encoded_features_list, temporal_features
            )
            return None, global_output
        else:
            _, _, temporal_output = self._forward_temporal_pipeline(x)
            return temporal_output, None
