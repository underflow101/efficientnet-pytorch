from model import EfficientNet
from torchsummary import summary

print("Efficient B0 Summary")
net = EfficientNet(1, 1)
summary(net.cuda(), (3, 224, 224))