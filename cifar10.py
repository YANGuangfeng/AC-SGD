import torch
import torch.nn as nn
from torchvision import datasets, transforms
import math
import numpy as np
import torch.nn.functional as F
import pandas as pd


def cifar10(batch_size, num_users):
    root = '../input/cifar10-python'
    trainset = datasets.CIFAR10(root=root, train=True, download=False,
                                transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ]))

    testset = datasets.CIFAR10(root=root, train=False, download=False,
                               transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]))
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size * num_users, shuffle=True, drop_last=True)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size * num_users, shuffle=False, drop_last=True)
    return train_loader, test_loader


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(**kwargs):
    return ResNet(ResidualBlock, [2, 2, 2, 2], **kwargs)


class IdenticalCompressor(object):
    def __init__(self):
        pass

    @staticmethod
    def compress(vec):
        return vec.clone()

    @staticmethod
    def decompress(signature):
        return signature


class TopKSparsificationCompressor(object):
    def __init__(self, size, shape, cr):
        self.size = size
        self.shape = shape
        self.cr = cr
        self.k = math.ceil(size * cr)

    def compress(self, vec):
        vec = vec.view(-1, self.size)
        ind = torch.zeros_like(vec)
        idx = torch.topk(torch.abs(vec), k=self.k, dim=1)[1]
        ind.scatter_(1, idx, 1)
        return vec * ind

    def decompress(self, signature):
        return signature.view(self.shape)


class RandKSparsificationCompressor(object):
    def __init__(self, size, shape, cr):
        self.cr = cr
        self.size = size
        self.shape = shape
        self.spar = torch.nn.Dropout(p=1 - cr)

    def compress(self, vec):
        vec = vec.view(-1, self.size)
        vec = self.spar(vec)
        return vec

    def decompress(self, signature):
        return signature.view(self.shape)


class SIGNCompressor(object):
    def __init__(self, size, shape):
        pass

    @staticmethod
    def compress(vec):
        return torch.sign(vec)

    @staticmethod
    def decompress(signature):
        return signature


class QSparCompressor(object):
    def __init__(self, size, shape, cr, n_bit):
        self.bit = n_bit
        self.s = pow(2, self.bit - 1) - 1
        self.spar = torch.nn.Dropout(p=1 - cr)
        self.size = size
        self.shape = shape
        self.code_dtype = torch.int32

    def compress(self, vec):
        vec = vec.view(-1, self.size)
        vec = self.spar(vec)
        norm = torch.norm(vec, dim=1, keepdim=True)
        # norm = torch.max(torch.abs(vec), dim=1, keepdim=True)[0]
        normalized_vec = vec / norm
        scaled_vec = torch.abs(normalized_vec) * self.s
        l = scaled_vec.type(self.code_dtype)
        probabilities = scaled_vec - l.type(torch.float32)
        r = torch.rand(l.size()).cuda()
        l[:] += (probabilities > r).type(self.code_dtype)

        signs = torch.sign(vec)
        return [norm, signs.view(self.shape), l.view(self.shape)]

    def decompress(self, signature):
        [norm, signs, l] = signature
        assert l.shape == signs.shape
        scaled_vec = l.type(torch.float32) * signs
        compressed = (scaled_vec.view((-1, self.size))) * norm / self.s
        return compressed.view(self.shape)


class QSGD(object):
    def __init__(self, size, shape, n_bit):
        self.bit = n_bit
        self.s = pow(2, self.bit - 1) - 1
        self.size = size
        self.shape = shape
        self.code_dtype = torch.int32

    def compress(self, vec):
        vec = vec.view(-1, self.size)
        norm = torch.norm(vec, dim=1, keepdim=True)
        # norm = torch.max(torch.abs(vec), dim=1, keepdim=True)[0]
        normalized_vec = vec / norm
        scaled_vec = torch.abs(normalized_vec) * self.s
        l = scaled_vec.type(self.code_dtype)
        probabilities = scaled_vec - l.type(torch.float32)
        r = torch.rand(l.size()).cuda()
        l[:] += (probabilities > r).type(self.code_dtype)
        signs = torch.sign(vec)
        return [norm, signs.view(self.shape), l.view(self.shape)]

    def decompress(self, signature):
        [norm, signs, l] = signature
        assert l.shape == signs.shape
        scaled_vec = l.type(torch.float32) * signs
        compressed = (scaled_vec.view((-1, self.size))) * norm / self.s
        return compressed.view(self.shape)


class Compressor(object):
    def __init__(self, parameters, cr, n_bit, alg):
        self.parameters = list(parameters)
        self.num_layers = len(self.parameters)
        self.compressors = list()
        self.compressed_gradients = [list() for _ in range(self.num_layers)]
        for param in self.parameters:
            param_size = param.flatten().shape[0]
            if alg == 'topk':
                self.compressors.append(TopKSparsificationCompressor(param_size, param.shape, cr))
            elif alg == 'sign':
                self.compressors.append(SIGNCompressor(param_size, param.shape))
            elif alg == 'ac':
                self.compressors.append(QSparCompressor(param_size, param.shape, cr, n_bit))
            elif alg == 'sgd':
                self.compressors.append(IdenticalCompressor())
            elif alg == 'randk':
                self.compressors.append(RandKSparsificationCompressor(param_size, param.shape, cr))
            else:
                self.compressors.append(QSGD(param_size, param.shape, n_bit))

    def record(self):
        for i, param in enumerate(self.parameters):
            decompressed_g = self.compressors[i].decompress(
                self.compressors[i].compress(param.grad.data)
            )
            self.compressed_gradients[i].append(decompressed_g)

    def apply(self):
        for i, param in enumerate(self.parameters):
            param.grad.data = torch.stack(self.compressed_gradients[i], dim=0).mean(dim=0)
        for compressed in self.compressed_gradients:
            compressed.clear()


def test(device, model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return correct / len(test_loader.dataset)


def csgd(model, optimizer, loss_func, train_loader, test_loader, device, r, EPOCH, loopNum, NUM_USER, alg):
    all_loss = []
    all_bit = []
    all_acc = []
    all_norm = []
    all_k = []
    all_Cn = []
    alpha = 0.999
    d = 11173962
    Cn = d * 32
    gap = 100
    train_data = list()
    cr = 1
    n_bit = 2
    k1 =  1 ## adaqs
    if alg == 'randk' or 'topk':
        cr = r / (32 + math.log2(d))
    if alg =='qsgd':
        n_bit = r
    # training...
    for epoch in range(EPOCH):
        print(epoch)
        for step, (data, target) in enumerate(train_loader):
            model.train()
            user_batch_size = len(data) // NUM_USER
            train_data.clear()
            for user_id in range(NUM_USER - 1):
                train_data.append((data[user_id * user_batch_size:(user_id + 1) * user_batch_size],
                                   target[user_id * user_batch_size:(user_id + 1) * user_batch_size]))
            train_data.append((data[(NUM_USER - 1) * user_batch_size:],
                               target[(NUM_USER - 1) * user_batch_size:]))

            if alg == 'ac' and len(all_loss) % gap == 0:
                if len(all_norm) < gap:
                    Cn = r * d * pow(alpha, 3000) * 50
                else:
                    Cn = r * d * pow(alpha, 0.5 * (6000 - len(all_norm))) * np.mean(all_norm[len(all_norm) - 50:])
                n_bit = math.floor(0.5 * math.log2(Cn) + 0.25)
                k = (Cn - 32) / (n_bit + math.log2(d))
                cr = k / d
            # elif alg == 'qsgd' and len(all_loss) % gap == 0:
            #     if len(all_norm) < gap:
            #         n_bit = r * pow(alpha, 3000) * 60
            #     else:
            #         n_bit = np.ceil(np.log2(np.mean(all_norm[len(all_norm) - 50:]) * r + 1)) + 1
            quantizer = Compressor(model.parameters(), cr, n_bit, alg)
            for user_id in range(NUM_USER):
                optimizer.zero_grad()
                _x, _y = train_data[user_id]
                x = _x.to(device)
                y = _y.to(device)
                output = model(x)  # cnn output
                loss = loss_func(output, y)  # cross entropy loss
                loss.backward()  # backpropagation, compute gradients
                quantizer.record()
            quantizer.apply()
            optimizer.step()
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            norm = total_norm ** (1. / 2)
            all_norm.append(norm)
            all_loss.append(loss.item())
            all_bit.append(n_bit)
            all_k.append(k)
            all_Cn.append(Cn)
            if len(all_loss) % 20 == 0:
                acc = test(device, model, test_loader)
                all_acc.append(acc)
            if len(all_loss) == loopNum:
                return all_loss, all_norm, all_acc, all_bit, all_k, all_Cn
    return all_loss, all_norm, all_acc, all_bit, all_k, all_Cn


def main():
    # Hyperparameters
    BATCH_SIZE = 32
    LR = 0.01
    EPOCH = 50
    NUM_REP = 1
    loopNum = 300
    NUM_USER = 8
    alg = 'ac'  #{'topk','sign','ac','sgd','qsgd','randk'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RAT = [0.5]
    Perform = pd.DataFrame(pd.DataFrame(columns=RAT))
    Accrucy = pd.DataFrame(pd.DataFrame(columns=RAT))
    NUM_BIT = pd.DataFrame(pd.DataFrame(columns=RAT))
    NUM_NORM = pd.DataFrame(pd.DataFrame(columns=RAT))
    NUM_K = pd.DataFrame(pd.DataFrame(columns=RAT))
    NUM_CN = pd.DataFrame(pd.DataFrame(columns=RAT))

    for r in RAT:
        loss_func = torch.nn.CrossEntropyLoss()
        Loss = np.zeros(loopNum)
        Acc = np.zeros(int(loopNum/20))
        for i in range(NUM_REP):
            # Loading data
            train_loader, test_loader = cifar10(BATCH_SIZE, NUM_USER)

            model = ResNet18().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            all_loss, all_norm, all_acc, all_bit, all_k, all_Cn = csgd(model, optimizer, loss_func, train_loader, test_loader, device,
                                                        r, EPOCH, loopNum, NUM_USER, alg)

            Loss = Loss + np.array(all_loss)
            Acc = Acc + np.array(all_acc)
        Loss = Loss / NUM_REP
        Acc = Acc / NUM_REP

        Perform.loc[:, r] = Loss
        Accrucy.loc[:, r] = Acc
        NUM_BIT.loc[:, r] = all_bit
        NUM_NORM.loc[:, r] = all_norm
        NUM_K[:, r] = all_k.values
        NUM_CN[:, r] = all_Cn.values

    filename1 = '/kaggle/working/loss1.csv'
    filename2 = '/kaggle/working/acc1.csv'
    filename3 = '/kaggle/working/bit1.csv'
    filename4 = '/kaggle/working/norm1.csv'
    filename5 = '/kaggle/working/k1.csv'
    filename6 = '/kaggle/working/cn1.csv'
    Perform.to_csv(filename1)
    Accrucy.to_csv(filename2)
    NUM_BIT.to_csv(filename3)
    NUM_NORM.to_csv(filename4)
    NUM_K.to_csv(filename5)
    NUM_CN.to_csv(filename6)


if __name__ == '__main__':
    main()