from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms

from .models.clip_models import CLIPModel


class Detector(torch.nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self._transform = transforms.Compose([
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        self._model = CLIPModel(name="ViT-L/14", num_classes=1)

    def load_pretrained(self, weights_path: Path) -> None:
        state_dict = torch.load(weights_path, map_location='cpu')
        self._model.fc.load_state_dict(state_dict)

    def configure(self, device: Optional[str], training: Optional[bool] = None, **kwargs) -> None:
        if device is not None:
            self.to(device)
            self._model.to(device)

        if training is None:
            return

        if training:
            self.train()
            self._model.train()
        else:
            self.eval()
            self._model.eval()

    def forward(self, img_batch: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        img_batch = self._transform(img_batch)
        sig = self._model(img_batch).sigmoid()
        label = torch.round(sig).to(torch.int)
        return label, sig
