import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import StepLR
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import copy


def quantize_model(model: nn.Module) -> nn.Module:
    quantized_model = copy.deepcopy(model)
    for param in quantized_model.parameters():
        # 将权重量化为[-1, 1]范围内的8位整数
        param.data = torch.round(param.data * 127) / 127
    return quantized_model


def dequantize_model(model: nn.Module) -> nn.Module:
    dequantized_model = copy.deepcopy(model)
    for param in dequantized_model.parameters():
        # 将量化后的权重恢复为浮点数
        param.data = param.data / 127
    return dequantized_model


"""
使用例
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.argmax(dim=1, keepdim=True)  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

"""


"""
def main():
    # 设置设备，利用CPU计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义原始网络模型
    original_model = Net()

    # 加载训练数据和测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 原始模型转移到设备上
    original_model = original_model.to(device)

    # 进行量化
    quantized_model = quantize_model(original_model)
    # quantized_model = original_model

    # 定义优化器和学习率调度器
    optimizer = optim.Adam(quantized_model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练和测试量化模型
    for epoch in range(1, 11):
        train(quantized_model, device, trainloader, optimizer, epoch)
        test(quantized_model, device, testloader)
        scheduler.step()

    # 解量化模型
    dequantized_model = dequantize_model(quantized_model)
    # dequantized_model = quantized_model

    # 测试解量化模型
    test(dequantized_model, device, testloader)


if __name__ == '__main__':
    main()
"""