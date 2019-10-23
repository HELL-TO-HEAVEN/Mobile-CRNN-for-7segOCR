#-*- coding:utf-8 -*-

# Original MobileNet code -> https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
# blockRNN
import torch.nn.functional as F
import torch.nn as nn
import torch

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class blockRNN(nn.Module):
    def __init__(self, in_size, hidden_size, bidirectional, dropout_rate=0.1):
        super(blockRNN, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(in_size, hidden_size, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch, add=False):
        batch_size = batch.size(1)
        batch = self.dropout(batch)
        outputs, hidden = self.gru(batch)
        if add:
            out_size = int(outputs.size(2) / 2)
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs

class MobileCRNN(nn.Module):
    def __init__(self, hidden_dim, vocab_size, dropout_rate=0.1, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = hidden_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 1],
                [6, 96, 3, 1],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
            ]
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        #RNN block
        self.linear1 = nn.Linear(self.last_channel, hidden_dim)
        self.gru1 = blockRNN(hidden_dim, hidden_dim, dropout_rate=dropout_rate, bidirectional=True)
        self.gru2 = blockRNN(hidden_dim, hidden_dim, dropout_rate=dropout_rate, bidirectional=True)
        self.linear2 = nn.Linear(hidden_dim * 2, vocab_size)
        self.softmax = nn.LogSoftmax(-1)

        # weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        h, w = x.shape[2:]
        if h < w:
            x = x.permute(0, 3, 1, 2).mean([-1])
        else:
            x = x.permute(0, 2, 1, 3).mean([-1])
        x = self.linear1(x)
        x = self.gru1(x, add=True)
        x = self.gru2(x)
        x = self.linear2(x).permute(1, 0, 2)
        return self.softmax(x)

if __name__ == '__main__':
    model = MobileCRNN(128, 13)
    x = torch.rand(4, 3, 90, 32)
    y = model(x)
    print(y.shape)
