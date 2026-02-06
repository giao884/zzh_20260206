import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(32 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNet18(ResNet):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__(BasicBlock, [2,2,2,2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=100):
            super(ResNet18, self).__init__()
            self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, num_classes)  

    def forward(self, x):
        return self.resnet(x)

class TexasPurchaseMLP(nn.Module):
    def __init__(self, input_dim=600, num_classes=100, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(TexasPurchaseMLP, self).__init__()

        layers = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, num_classes)
        self._initialize_weights()

    def forward(self, x):
        
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def texas_purchase_mlp():
    return TexasPurchaseMLP(
        input_dim=600,
        num_classes=100,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.3
    )
