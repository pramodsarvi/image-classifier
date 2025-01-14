import torchvision.models as models
import torch.nn as nn

def build_model(pretrained=True, fine_tune=True, num_classes=3):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(pretrained=True)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
       
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
        
    model.classifier[1] =nn.Sequential(nn.Linear(1280,512),nn.Linear(in_features=512, out_features=num_classes))
    
    return model
# print(build_model())