from torch import nn
import torch

class quantized_model(nn.Module):
    def __init__(self, model):
        super(quantized_model, self).__init__()
        self.model_fp32 = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x
    
