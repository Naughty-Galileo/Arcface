import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn, optim
import torchvision
from torch.utils.data import DataLoader
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ArcLoss(nn.Module):
    def __init__(self, feature_num, cls_num):
        """
        :param feature_num: 特征数
        :param cls_num:     类别数
        """
        super(ArcLoss, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_num, cls_num).cuda())
        nn.init.xavier_uniform_(self.w)
        self.func = nn.Softmax()

    def forward(self, x, s=64, m=0.5):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)

        cosa = torch.matmul(x_norm, w_norm)/10  # 权重W和特征向量X的夹角余弦值
        a = torch.acos(cosa)  # 角度
        arcsoftmax = torch.log(torch.exp(s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) -
                                                                  torch.exp(s * cosa) + torch.exp(
                    s * torch.cos(a + m))))
        return arcsoftmax


class ClsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.PReLU(),
                                        nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.PReLU(),
                                        nn.MaxPool2d(3, 2))
        self.feature_layer = nn.Sequential(nn.Linear(11 * 11 * 64, 256), nn.BatchNorm1d(256), nn.PReLU(),
                                           nn.Linear(256, 128), nn.BatchNorm1d(128), nn.PReLU(),
                                           nn.Linear(128, 2), nn.PReLU())
        self.arcsoftmax = ArcLoss(2, 10)
        self.loss_fn = nn.NLLLoss()

    def forward(self, x, s, m):
        conv = self.conv_layer(x)
        conv = conv.reshape(x.size(0), -1)
        feature = self.feature_layer(conv)
        out = self.arcsoftmax(feature, s, m)
        return feature, out

    def get_loss(self, out, ys):
        return self.loss_fn(out, ys)


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    train = DataLoader(dataset=train_data, batch_size=1024, shuffle=True, drop_last=True)

    net = ClsNet().cuda()

    # path = r'params/weightnet2.pt'
    # if os.path.exists(path):
    #     net.load_state_dict(torch.load(path))
    #     net.eval()
    #     print('load susseful')
    # else:
    #     print('load fail')

    # optimism = optim.SGD(net.parameters(), lr=1e-3)  # 存为pic, weightnet
    optimism = optim.Adam(net.parameters(), lr=5e-4)  # 优化器  效果更好--2

    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    epoch = 1000

    for i in range(epoch):
        print('epoch: {}'.format(i))
        # print(len(train))
        tar = []
        out = []
        for j, (input, target) in enumerate(train):
            input = input.cuda()
            target = target.cuda()
            feature, output = net(input, 10, 0.5)  # 将图像数据扔进网络
            # net(input, 10, 0.5) --1/2
            # net(input, 64, 0.5) --3

            loss = net.get_loss(output, target)

            label = torch.argmax(output, dim=1)  # 选出最大值的索引作为标签

            # 清空梯度 反向传播 更新梯度
            optimism.zero_grad()
            loss.backward()
            optimism.step()

            feature = feature.cpu().detach().numpy()
            # print(output)
            target = target.cpu().detach()
            # print(target)
            out.extend(feature)  # 加载画图数据
            tar.extend(target)

            print('[epochs - {} - {} / {}] loss: {} '.format(i + 1, j, len(train), loss.float()))
            outstack = np.stack(out)
            tarstack = torch.stack(tar)

            plt.ion()
            if j == 3:
                for m in range(10):
                    index = torch.as_tensor(torch.nonzero(tarstack == m))
                    # print(index)
                    plt.scatter(outstack[:, 0][index[:, 0]], outstack[:, 1][index[:, 0]], c=c[m], marker='.')
                    plt.legend(cls, loc="upper right")
                plt.title("epoch={}".format(str(i + 1)))
                plt.savefig('pic4/picture{0}.jpg'.format(i + 1))
                # plt.show()
                # plt.pause(10)
            plt.close()
        torch.save(net.state_dict(), r'params/weightnet4.pt')
