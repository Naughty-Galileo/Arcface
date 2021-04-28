from PIL import Image
from torchvision import transforms
from main import ClsNet
import torch
import torchvision
import os
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def test_mydata():
    # 调取图片
    path = 'test/'
    for i in range(10):
        file = path+str(i)+'.png'
        images = Image.open(file)
        images = images.convert('L')

        transform = transforms.ToTensor()
        images = transform(images)
        images = images.resize(1, 1, 28, 28)
        images = images.cuda()

        # 加载网络和参数
        model = ClsNet().cuda()
        model.load_state_dict(torch.load(r'params/weightnet2.pt'))
        model.eval()
        feature, outputs = model(images, 10, 0.5)

        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().detach().numpy()
        print('{} --> 预测值：{}'.format(i, predicted[0]))


def test_MNISTdata():
    test_set = torchvision.datasets.MNIST(
        root='./mnist',
        train=False,
        download=False,
        transform=torchvision.transforms.ToTensor()
    )
    test = DataLoader(dataset=test_set, batch_size=100, shuffle=True, drop_last=True)

    # # 可视化
    # batch = next(iter(test))
    # images, labels = batch
    # images = images.numpy()
    #
    # fig = plt.figure(figsize=(25, 25))
    # for idx in np.arange(36):
    #     ax = fig.add_subplot(6, 6, idx + 1, xticks=[], yticks=[])
    #     ax.imshow(np.squeeze(images[idx]), cmap='gray')
    #
    #     ax.set_title(str(labels[idx].item()))
    # plt.show()
    # plt.pause(1)
    # plt.close()

    net = ClsNet().cuda()
    net.load_state_dict(torch.load(r'params/weightnet2.pt'))
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 不更新梯度
        for data in test:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            feature, outputs = net(images, 10, 0.5)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 预测正确的数目

    print('测试集上整体的准确率： {:.2f}%'.format(100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            feature, outputs = net(images, 10, 0.5)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print('10个类别的准确率:')
    for i in range(10):
        print('{}的准确率 : {:.2f}%'.format(i, 100 * class_correct[i] / class_total[i]))


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_MNISTdata()
test_mydata()



