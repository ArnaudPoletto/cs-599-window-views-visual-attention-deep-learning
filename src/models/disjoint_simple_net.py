import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from src.models.image_encoder import ImageEncoder
from src.models.image_decoder import ImageDecoder
from src.config import SEQUENCE_LENGTH

class DisjointSimpleNet(nn.Module):
    def __init__(
        self, 
        freeze_encoder: bool,
        freeze_temporal_pipeline: bool,
        hidden_channels_list: List[int],
        output_type: str,
        eps: float = 1e-6
    ) -> None:
        if output_type not in ["temporal", "global"]:
            raise ValueError(f"❌ Invalid output type: {output_type}")
        if freeze_temporal_pipeline and output_type == "temporal":
            raise ValueError("❌ Cannot freeze the temporal pipeline when output type is temporal.")
        
        super(DisjointSimpleNet, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.freeze_temporal_pipeline = freeze_temporal_pipeline
        self.hidden_channels_list = hidden_channels_list
        self.output_type = output_type
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

        self.image_encoders = nn.ModuleList([
            ImageEncoder(freeze=freeze_encoder or freeze_temporal_pipeline)
            for _ in range(SEQUENCE_LENGTH)
        ])
        self.image_decoders = nn.ModuleList([
            ImageDecoder(
                features_channels_list=self.image_encoders[i].feature_channels_list,
                hidden_channels_list=hidden_channels_list,
                features_sizes=self.image_encoders[i].feature_sizes,
                output_channels=1,
                with_final_sigmoid=False,
            )
            for i in range(SEQUENCE_LENGTH)
        ])

        self.final_global_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=SEQUENCE_LENGTH,
                out_channels=SEQUENCE_LENGTH,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=SEQUENCE_LENGTH,
                out_channels=1,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

        if freeze_temporal_pipeline:
            for image_decoder in self.image_decoders:
                for param in image_decoder.parameters():
                    param.requires_grad = False

        if output_type == "temporal":
            for param in self.final_global_layer.parameters():
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

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Normalize input and resize the tensor accordingly
        is_image = x.dim() == 4
        if not is_image:
            batch_size, sequence_length, channels, height, width = x.shape
            x = x.view(-1, channels, height, width)
        x_image = self._normalize_input(x, self.image_mean, self.image_std)
        if is_image:
            x_image = x_image.unsqueeze(1).repeat(1, SEQUENCE_LENGTH, 1, 1, 1)
        else:
            x_image = x_image.view(batch_size, sequence_length, channels, height, width)

        # Encode and decode each image in the sequence
        decoded_features_list = []
        for i, (image_encoder, image_decoder) in enumerate(zip(self.image_encoders, self.image_decoders)):
            x_image_i = x_image[:, i]
            encoded_features_list = image_encoder(x_image_i)
            decoded_features = image_decoder(encoded_features_list)
            decoded_features_list.append(decoded_features)
        temporal_features = torch.stack(decoded_features_list, dim=1)

        # Compute the output
        if self.output_type == "global":
            global_output = self.final_global_layer(temporal_features)
            return global_output, None
        else:
            temporal_output = self.sigmoid(temporal_features)
            return temporal_output, None