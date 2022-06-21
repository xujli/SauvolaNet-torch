import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module
from pytorch_edition.layerUtils import *

class Conv_block(Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size=(3, 3),
                 dilation_rate=(1, 1),
                 strides=(1, 1),
                 norm_type='inorm',
                 activation='relu',
                 **conv_kwargs,
                 ):
        super(Conv_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding='same',
                              dilation=dilation_rate, stride=strides)
        # nn.init._no_grad_fill_(self.conv.weight, 0.5)
        # nn.init._no_grad_fill_(self.conv.bias, 0.5)
        if norm_type == 'bnorm':
            self.norm = nn.BatchNorm2d(filters, eps=1e-5, momentum=0, affine=False, track_running_stats=False)
        elif norm_type == 'inorm':
            self.norm = nn.InstanceNorm2d(filters, eps=1e-5, momentum=0, affine=False, track_running_stats=False)
        else:
            raise NotImplementedError(f"ERROR: unknown normalization type {norm_type}")

        self.activation = nn.ReLU()

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        conv = self.conv(x)
        norm = self.norm(conv)
        act = self.activation(norm)
        return conv, act

class Multiscale_sauvola(Module):
    def __init__(self, window_size_list=[7, 15, 23, 31, 39, 47, 55, 63],
                 train_k=False, # test
                 train_R=False,
                 train_alpha=False,
                 norm_type='inorm',
                 base_filters=8,
                 img_range=(0., 1.)):

        super(Multiscale_sauvola, self).__init__()

        # attention branch
        n = len(window_size_list)
        filters = base_filters
        t = int(np.ceil(np.log2(max(window_size_list)))) - 1
        # 1st block
        self.conv1 = Conv_block(1, filters, norm_type=norm_type)
        # later blocks
        convs = []

        self.convs1 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2),  norm_type=norm_type)
        filters += base_filters
        self.convs2 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2),  norm_type=norm_type)
        filters += base_filters
        self.convs3 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2),  norm_type=norm_type)
        filters += base_filters
        self.convs4 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2),  norm_type=norm_type)
        filters += base_filters
        self.convs5 = Conv_block(filters, filters + base_filters, dilation_rate=(2, 2),  norm_type=norm_type)
        filters += base_filters

        self.conv2 = nn.Conv2d(filters, n, (3, 3), padding='same')
        nn.init.constant(self.conv2.weight, 0.5)
        nn.init.constant(self.conv2.bias, 0.5)
        self.softmax = nn.Softmax(dim=1)

        self.th = SauvolaMultiWindow(window_size_list=window_size_list,
                            train_k=train_k,
                            train_R=train_R)
        self.diff = DifferenceThresh(img_min=img_range[0],
                            img_max=img_range[1],
                            init_alpha=16.,
                            train_alpha=train_alpha)

    def forward(self, inputs):
        conv1, x2 = self.conv1(inputs)
        _, x = self.convs1(x2)
        _, x = self.convs2(x)
        _, x = self.convs3(x)
        _, x = self.convs4(x)
        _, x = self.convs5(x)
        f = self.conv2(x)
        f = self.softmax(f)

        x1 = torch.permute(inputs, (0, 2, 3, 1))
        th = self.th(x1)
        th1 = torch.sum(torch.unsqueeze(f, -1) * th, dim=1)
        diff = self.diff([x1, th1])
        return diff

if __name__ == '__main__':
    model = Multiscale_sauvola()
    data = torch.ones([1, 1, 256, 256]) * 0.5
    output = model(data)
    print(output[0].mean(), output[0].std())