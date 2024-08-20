from config import *
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn



def build_model(pretrained=True, fine_tune=True, num_classes=10):

    
    class EfficientNet(nn.Module):
        def __init__(self,num_classes):
            super(EfficientNet, self).__init__()

            self.backbone = nn.Sequential(*list(models.efficientnet_b0(weights= torchvision.models.EfficientNet_B0_Weights.DEFAULT).children())[:-2])
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(1280, num_classes)
            self.rel = nn.ReLU()
            
        def forward(self, x):
            
            x = self.backbone(x)
            x = self.avg_pool(x)

            x = x.view(x.size(0), -1)
            x = self.linear(x) 
            return x

    model = EfficientNet(num_classes)
    print(model(torch.randn(1,3,128,224)))
   
    return model