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
        with_global_output: bool = True,
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
        self.with_global_output = with_global_output

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
            freeze=freeze_encoder,
        )

        self.temporal_decoder = ImageDecoder(
            features_channels_list=self.image_encoder.feature_channels_list,
            hidden_channels_list=hidden_channels_list,
            features_sizes=self.image_encoder.feature_sizes,
            output_channels=SEQUENCE_LENGTH,
        )
        if with_global_output:
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

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _normalize_input(
        self, 
        x: torch.Tensor, 
        mean: torch.Tensor, 
        std: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        x = x.clone()
        normalized_x = (x - mean) / (std + eps)

        return normalized_x

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

        # Decode the temporal and global features
        temporal_features = self.temporal_decoder(encoded_features_list)
        temporal_output = temporal_features

        # Estimate global saliency map using spatio-temporal mixing module if required
        if self.with_global_output:
            global_features = self.global_decoder(encoded_features_list)
            global_output = self.spatio_temporal_mixing_module(
                encoded_features_list=encoded_features_list,
                temporal_features=temporal_features,
                global_features=global_features,
            ).squeeze(1)
        else: 
            global_output = None

        return temporal_output, global_output
