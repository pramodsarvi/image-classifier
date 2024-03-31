
from torchvision import transforms

def get_train_transform(IMAGE_DIMS, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_DIMS[0],IMAGE_DIMS[1])),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform

T = transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(degrees=(350, 370)),
        transforms.RandomPerspective(distortion_scale=0.15, p=1.0),
        transforms.ColorJitter(brightness=.60, hue=.15),
        transforms.RandomAutocontrast(p=0.45)
])


def get_valid_transform(IMAGE_DIMS, pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_DIMS[0],IMAGE_DIMS[1])),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform

def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]                                                                                                                                                                                                                                                                
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize

normalize_pretrained = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
