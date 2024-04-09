from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg19


class PerceptionLoss(nn.Module):
    def __init__(self, feature_layer: int = 35):
        super(PerceptionLoss, self).__init__()
        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer]).eval()
        
        #отключаем подсчет градиентов внутри vgg
        for name, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        perception_loss = F.mse_loss(self.features(input), self.features(target))

        return perception_loss
