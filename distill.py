# 导入必要包文件
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from torch import softmax


class DistillModel:
    def __init__(self, source_data, temperature: int = 5, lr: float = 0.001, momentum: float = 0.9):
        """
        蒸馏模型，减少计算量。
        :param source_data: 输入的数据集，原始数据
        :param temperature: 温度控制，控制输出平滑度
        :param lr: 初始学习率
        :param momentum: 使用SGD优化器会用到，目前使用Adam优化器，暂时不需考虑此参数
        """
        self.source_data = source_data
        self.temperature = temperature
        self.momentum = momentum
        # 定义老师与学生模型，resnet可以用其他网络替换
        self.TeacherModel = nn.Sequential(resnet18(pretrained=True))
        self.StudentModel = nn.Sequential(resnet18(pretrained=True))
        # 定义损失函数：交叉熵损失；以及优化器：SGD优化。
        # 关于为什么选择交叉熵，因为后续TeacherModel会对StudentModel作用，即软硬标签部分
        self.criterion = nn.CrossEntropyLoss()
        # 定义优化器
        self.optimizer = optim.Adam(self.StudentModel.parameters(), lr)
        # optimizer = optim.SGD(self.StudentModel, lr=self.lr, momentum=self.momentum)

    def train(self, hard_labels, epoch: int = 100) -> None:
        """
        训练过程
        :param hard_labels: 需要输入固定在数据集中的标签进行训练
        :param epoch: 训练步数
        """
        for epoch in range(epoch):
            # 输入数据
            t_output = self.TeacherModel(self.source_data)
            s_output = self.StudentModel(self.source_data)
            # 计算教师模型的软标签（Soft Label）
            soft_labels = softmax(t_output / self.temperature, dim=1)
            # 计算损失函数,预测的损失求和
            loss = self.criterion(s_output, hard_labels) + self.criterion(s_output, soft_labels)
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# 使用范例
# source_data = torch.randn(10, 3, 224, 224)  # 示例数据
# hard_labels = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2, 1])  # 示例硬标签
# in_features = 3 # 创建DistillModel对象
# model = DistillModel(source_data)
# model.train(hard_labels, epoch=10) # 训练模型
