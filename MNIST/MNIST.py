import torch
import torchvision.transforms
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader

# 指定训练设备 gpu优先
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 准备MNIST训练和测试数据集
train_data = datasets.MNIST('./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                            download=True)
test_data = datasets.MNIST('./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                           download=True)
# 设置数据加载器
train_loader = DataLoader(train_data, batch_size=15, shuffle=True)
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
model = Net()
# 指定训练设备
model.to(device)
# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 设置损失函数并指定训练设备 交叉熵
Cross_loss = nn.CrossEntropyLoss()
Cross_loss.to(device)


def train(epoch):
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = Cross_loss(output, targets)
        loss.backward()
        optimizer.step()
    print(f'epoch: {epoch} finish')


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
    print(f'Initial Accuracy: {Accuracy}')
    for epoch in range(10):
        train(epoch)
        Accuracy = test()
        print(f'epoch: {epoch + 1}, Accuracy: {Accuracy}')
    # 模型保存方式一
    torch.save(model, 'model.pt')
    # 保存方式二
    # torch.save(model.state_dict(), 'model.pt')
