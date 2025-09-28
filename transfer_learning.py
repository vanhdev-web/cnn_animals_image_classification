from torchvision.models import ResNet50_Weights, resnet50
import torch.nn as nn
import torch
from torchsummary import summary

class MyResnet(nn.Module):
    def __init__(self, num_classes) :
        super().__init__()
        self.model = resnet50(weights = ResNet50_Weights.DEFAULT)
        del self.model.fc

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        for name, param in self.model.named_parameters():
            if not "fc" in name:
                param.requires_grad = False
            if "layer4" in name:
                param.requires_grad = True

    def _forward_impl(self, x) :
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x) :
        return self._forward_impl(x)