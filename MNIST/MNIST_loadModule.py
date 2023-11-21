import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

# 指定训练设备 gpu优先
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 准备MNIST测试数据集
test_data = datasets.MNIST('./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                           download=True)
# 设置数据加载器
test_loader = DataLoader(test_data, batch_size=15, shuffle=True)


# 网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平输入数据
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return torch.nn.functional.log_softmax(self.fc4(x), dim=1)


# 实例化网络
model = torch.load('model.pt')
# 指定训练设备
model.to(device)


def test():
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(test_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)
            correct += (output.argmax(1) == targets).sum()
        return correct/total


if __name__ == '__main__':
    Accuracy = test()
    print(f'Model Accuracy: {Accuracy}')
