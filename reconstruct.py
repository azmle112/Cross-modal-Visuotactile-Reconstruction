"""重建网络-GAN"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import time
from torch.autograd import Variable
import torch.nn.parallel as parallel
import matplotlib.pyplot as plt


# from torch.utils.tensorboard import SummaryWriter
# from visdom import Visdom

# # 创建SummaryWriter对象，指定日志目录
# log_dir = "logs/"
# writer = SummaryWriter(log_dir)

# 新建名为'demo'的环境
# viz = Visdom(env='demo')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.bn1 = nn.BatchNorm1d(512)

        self.fc = nn.Linear(512, 2 * 40 * 128)  # FC层，将输入向量转换为适应卷积层输入大小的特征图

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 2*40
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),  # Upsample层，8*160
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),  #
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv_block3 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='nearest'),  # Upsample层，64*1280
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.Up = nn.Upsample(size=(101, 2001), mode='bilinear')

    def forward(self, z):
        batch_size = z.size(0)
        z = self.bn1(z)
        x = self.fc(z)
        x = x.view(batch_size, 128, 2, 40)  # 将FC层输出的特征图进行reshape
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.Up(x)
        return x


# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.x_shape = [2, 101, 2001]

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.x_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


# 超参数设置
epochs = 100
batch_size = 32
lr = 0.0001
adam_b1 = 0.5
adam_b2 = 0.99
latent_dim = 512

# 转换NumPy数组为PyTorch张量
features = torch.from_numpy(np.load('./generated_data/features.npy')).float().to('cuda')
spec = torch.from_numpy(np.load('./generated_data/spec1.npy')).float().to('cuda')

# 设置数据集
train_set = TensorDataset(features, spec)
indices = list(range(len(train_set)))
np.random.shuffle(indices)
train_sampler = SubsetRandomSampler(indices)
train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)

# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()

# 定义损失函数和优化器
criterion = nn.BCELoss()
criterion_2 = nn.MSELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(adam_b1, adam_b2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(adam_b1, adam_b2))

cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()
    criterion_2.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if torch.cuda.device_count() > 1:
    generator = parallel.DataParallel(generator)
    discriminator = parallel.DataParallel(discriminator)
    # criterion = parallel.DataParallel(criterion)

# 开始训练
j = 0
G_loss = []
D_loss = []
MSE_loss = []
for epoch in range(epochs):
    for i, (batch_feature, batch_spec) in enumerate(train_loader):
        start = time.time()
        # 对抗性的真实和虚假标签
        # real_labels = torch.ones(batch_size, 1).to("cuda:0")
        # fake_labels = torch.zeros(batch_size, 1).to("cuda:0")
        size = len(batch_feature)
        real_labels = Variable(Tensor(size, 1).fill_(1.0), requires_grad=False)
        fake_labels = Variable(Tensor(size, 1).fill_(0.0), requires_grad=False)

        # 生成器生成虚假的spec数据
        generated_spec = generator(batch_feature)
        fake_output = discriminator(generated_spec)

        # 训练G，计算生成器的损失函数并进行反向传播和优化
        generator_optimizer.zero_grad()
        # 生成器的损失度量其欺骗判别器的能力
        generator_loss = criterion(fake_output, real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # 训练D，计算判别器的损失函数并进行反向传播和优化
        discriminator_optimizer.zero_grad()
        real_output = discriminator(batch_spec)
        # 判别器的损失度量其对真实样本和生成样本进行分类的能力
        discriminator_loss = criterion(real_output, real_labels) + \
                             criterion(discriminator(generated_spec.detach()), fake_labels)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # MSE损失
        mse_loss = criterion_2(batch_spec, generati8or(batch_feature))
        mse_loss.backward()

        end = time.time()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f][MSE: %f] [Time: %f]"
            % (epoch, epochs, i + 1, len(train_loader), discriminator_loss.item(), generator_loss.item(),
               mse_loss.item(), end - start)
        )

        j += 1
        G_loss.append(generator_loss.item())
        D_loss.append(discriminator_loss.item())
        MSE_loss.append(mse_loss.item())

        # # 将损失写入SummaryWriter
        # writer.add_scalar('G-Loss', generator_loss.item(), j)
        #
        # # 可以记录其他指标，如准确率、学习率等
        # writer.add_scalar('D-Loss', discriminator_loss.item(), j)
        # viz.line(
        #     X=np.array([j]),
        #     Y=np.array([discriminator_loss.item()]),
        #     win='D-loss',
        #     update='append')
        # viz.line(
        #     X=np.array([j]),
        #     Y=np.array([generator_loss.item()]),
        #     win='G-loss',
        #     update='append')
        # viz.line(
        #     X=np.array([j]),
        #     Y=np.array([generator_loss.item()]),
        #     win='GD-loss',
        #     update='append')
        # viz.line(
        #     X=np.array([j]),
        #     Y=np.array([discriminator_loss.item()]),
        #     win='GD-loss',
        #     update='append')
    if epoch == 99:
        np.save("./generated_data/generated_spec_without_mse.npy", generated_spec.detach().cpu().numpy())
        np.save("./generated_data/batch_spec_without_mse.npy", batch_spec.cpu().numpy())

# writer.close()
# 使用生成器对features进行重建
j = range(j)
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('G_loss')
ax.plot(j, G_loss)
plt.savefig("./three_devide_loss/G_loss.png")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('D_loss')
ax.plot(j, D_loss)
plt.savefig("./three_devide_loss/D_loss.png")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE_loss')
ax.plot(j, MSE_loss)
plt.savefig("./three_devide_loss/MSE_loss.png")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('loss')
ax.plot(j, G_loss, label='G_loss')
ax.plot(j, D_loss, label='D_loss')
ax.legend()
plt.savefig("./three_devide_loss/GD_loss_without_mse.png")
