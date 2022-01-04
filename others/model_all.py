import torch
from thop import profile
from torchvision import models


def get_shufflenet_v2_x0_5_Model():
    model = models.shufflenet_v2_x0_5(pretrained=True)
    model.requires_grad_(True)
    model.fc = torch.nn.Sequential(torch.nn.Linear(1024, 2))
    return model

def get_shufflenet_v2_x1_0_Model():
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.requires_grad_(True)
    model.fc = torch.nn.Sequential(torch.nn.Linear(1024, 2))
    return model

def get_squeezenet1_1():
    model = models.squeezenet1_1(pretrained=True)
    model.requires_grad_(True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=False),
        torch.nn.Conv2d(512,2,kernel_size=(1,1),stride=(1,1)),
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveMaxPool2d(output_size=(1,1))
    )
    return model

def get_mobilenet_v2():
    model = models.mobilenet_v2(pretrained=True)
    model.requires_grad_(True)
    model.classifier[1] = torch.nn.Linear(in_features=1280,out_features=2,bias=True)
    return model

def get_mobilenet_v3():
    model = models.mobilenet_v3_small(pretrained=True)
    model.requires_grad_(True)
    model.classifier[3] = torch.nn.Linear(in_features=1024,out_features=2,bias=True)
    return model

def get_all_parameters(model):
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("FLOPs:", flops)
    print("params:", params)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2f" % (total))
