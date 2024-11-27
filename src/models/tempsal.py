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
        hidden_channels_list: List[int],
    ) -> None:
        """
        Initialize the TempSAL model.
        
        Args:
            freeze_encoder (bool): Whether to freeze the encoder's parameters.
            hidden_channels_list (List[int]): The number of hidden channels to use in the decoders.
        """
        super(TempSAL, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.hidden_channels_list = hidden_channels_list

        self.image_encoder = ImageEncoder(
            freeze=freeze_encoder,
        )

        self.temporal_decoder = ImageDecoder(
            features_channels_list=self.image_encoder.feature_channels_list,
            hidden_channels_list=hidden_channels_list,
            features_sizes=self.image_encoder.feature_sizes,
            output_channels=SEQUENCE_LENGTH,
        )
        self.global_decoder = ImageDecoder(
            features_channels_list=self.image_encoder.feature_channels_list,
            hidden_channels_list=hidden_channels_list,
            features_sizes=self.image_encoder.feature_sizes,
            output_channels=1,
        )
        self.spatio_temporal_mixing_module = SpatioTemporalMixingModule(
            hidden_channels_list=hidden_channels_list,
            feature_channels_list=self.image_encoder.feature_channels_list,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TempSAL model.
        
        Args:
            x (torch.Tensor): The input image.
            
        Returns:
            torch.Tensor: The temporal and global saliency maps.
        """
        # Encode the input image
        encoded_features_list = self.image_encoder(x)

        # Decode the temporal and global features
        temporal_features = self.temporal_decoder(encoded_features_list)
        global_features = self.global_decoder(encoded_features_list)

        # The temporal output is the temporal features
        temporal_output = temporal_features

        # The global output is the output of the spatio-temporal mixing module
        global_output = self.spatio_temporal_mixing_module(
            encoded_features_list=encoded_features_list,
            temporal_features=temporal_features,
            global_features=global_features,
        ).squeeze(1)

        return temporal_output, global_output
