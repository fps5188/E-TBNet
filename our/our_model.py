from math import log
import torch.nn as nn
import torch
from thop import profile

class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1,include_pool=True,**kwargs):
        super(BasicBlock, self).__init__()
        self.include_pool = include_pool
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,
                          stride=stride, bias=False),
                ECALayer(c1=out_channel),
                nn.BatchNorm2d(out_channel))
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2)

    def forward(self, x):

        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        if self.include_pool:
            out = self.max_pool(out)
        return out

class ECALayer(nn.Module):
    #ECAblock
    def __init__(self,c1,gamma=1,b=1):
        super(ECALayer, self).__init__()
        t = int(abs((log(c1, 2)+b)/gamma))
        k = t if t%2 else t+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=1,out_channels=1,
                              kernel_size=k,padding=int(k/2),bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        y = x * y.expand_as(x)
        return y

class Model_eca(nn.Module):
    def __init__(self):
        super(Model_eca, self).__init__()
        self.layer1 = BasicBlock(in_channel=3,out_channel=16,stride=2)

        self.layer2 = BasicBlock(in_channel=16,out_channel=32)

        self.layer3 = BasicBlock(in_channel=32,out_channel=48)

        self.layer4 = BasicBlock(in_channel=48,out_channel=64)

        self.layer5 = BasicBlock(in_channel=64,out_channel=128)

        self.eca = ECALayer(c1=128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=128,out_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=2)
        self.soft = nn.Softmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print('conv')
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')

            if isinstance(m, nn.Linear):  # 判断是否是线性层
                print('linear')
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.eca(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_all_parameters(model):
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("FLOPs:", flops)
    print("params:", params)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2f" % (total))
