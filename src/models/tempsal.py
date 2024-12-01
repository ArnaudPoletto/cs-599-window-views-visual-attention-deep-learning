import torch
from torch import nn
from typing import List

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
        with_global_output: bool,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize the TempSAL model.

        Args:
            freeze_encoder (bool): Whether to freeze the encoder's parameters.
            hidden_channels_list (List[int]): The number of hidden channels to use in the decoders.
        """
        super(TempSAL, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.freeze_temporal_pipeline = freeze_temporal_pipeline
        self.hidden_channels_list = hidden_channels_list
        self.with_global_output = with_global_output
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
            features_sizes=self.image_encoder.feature_sizes,
            output_channels=SEQUENCE_LENGTH,
            with_final_sigmoid=False,
        )
        if with_global_output:
            self.global_decoder = ImageDecoder(
                features_channels_list=self.image_encoder.feature_channels_list,
                hidden_channels_list=hidden_channels_list,
                features_sizes=self.image_encoder.feature_sizes,
                output_channels=1,
                with_final_sigmoid=False,
            )
            self.spatio_temporal_mixing_module = SpatioTemporalMixingModule(
                hidden_channels_list=hidden_channels_list,
                feature_channels_list=self.image_encoder.feature_channels_list,
            )

        self.sigmoid = nn.Sigmoid()

        if freeze_temporal_pipeline:
            for param in self.temporal_decoder.parameters():
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
        batch_size, channels , height, width = x.size()
        x = x.view(batch_size, channels, -1)
        x = x / (x.max(dim=2, keepdim=True)[0] + self.eps)
        x = x.view(batch_size, channels, height, width)

        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TempSAL model.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The temporal and global saliency maps.
        """
        # Encode the input image
        x_image = self._normalize_input(x, self.image_mean, self.image_std)
        encoded_features_list = self.image_encoder(x_image)

        # Decode temporal features and get temporal output
        temporal_features = self.temporal_decoder(encoded_features_list)
        temporal_output = self.sigmoid(temporal_features)
        temporal_output = self._normalize_spatial_dimensions(temporal_output)

        # Decode global features and get global output with spatio-temporal mixing module if needed
        if self.with_global_output:
            global_features = self.global_decoder(encoded_features_list)
            global_output = self.spatio_temporal_mixing_module(
                encoded_features_list=encoded_features_list,
                temporal_features=temporal_features,
                global_features=global_features,
            )
            global_output = self._normalize_spatial_dimensions(global_output).squeeze(1)
        else:
            global_output = None

        return temporal_output, global_output
