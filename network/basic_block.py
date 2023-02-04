import torch
import torch.nn as nn
import math


def weights_init(m):
    #  Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def convbnlrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=False))


def convbnsig(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channels), nn.Sigmoid())


def convbn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channels))


def makePad(gks):
    pad = []
    for i in range(gks):
        for j in range(gks):
            top = i
            bottom = gks - 1 - i
            left = j
            right = gks - 1 - j
            pad.append(torch.nn.ZeroPad2d((left, right, top, bottom)))
    return pad


class DySPN(nn.Module):

    def __init__(self, in_channels, kernel_size=7, iter_times=6):
        super(DySPN, self).__init__()
        assert kernel_size == 7, 'now only support 7'
        self.kernel_size = kernel_size
        self.iter_times = iter_times
        self.affinity7 = convbn(in_channels, 7**2 - 5**2, kernel_size=3, stride=1, padding=1)
        self.affinity5 = convbn(in_channels, 5**2 - 3**2, kernel_size=3, stride=1, padding=1)
        self.affinity3 = convbn(in_channels, 3**2 - 1**2, kernel_size=3, stride=1, padding=1)

        self.attention7 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)
        self.attention5 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)
        self.attention3 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)
        self.attention1 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)
        self.attention0 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)

    def forward(self, feature, d0, d00):
        affinity7 = self.affinity7(feature)
        affinity5 = self.affinity5(feature)
        affinity3 = self.affinity3(feature)

        attention7 = self.attention7(feature)
        attention5 = self.attention5(feature)
        attention3 = self.attention3(feature)
        attention1 = self.attention1(feature)

        zero_pad = makePad(self.kernel_size)

        weightmask_7 = [
            1, 1, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1
        ]
        weightmask_5 = [
            0, 0, 0, 0, 0, 0, 0, 
            0, 1, 1, 1, 1, 1, 0, 
            0, 1, 0, 0, 0, 1, 0, 
            0, 1, 0, 0, 0, 1, 0, 
            0, 1, 0, 0, 0, 1, 0, 
            0, 1, 1, 1, 1, 1, 0, 
            0, 0, 0, 0, 0, 0, 0
        ]
        weightmask_3 = [
            0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 
            0, 0, 1, 1, 1, 0, 0, 
            0, 0, 1, 0, 1, 0, 0, 
            0, 0, 1, 1, 1, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0
        ]

        dt = d0
        for iters in range(self.iter_times):
            # normalization
            guide7 = attention7[:, iters:iters + 1, ...] * affinity7
            guide5 = attention5[:, iters:iters + 1, ...] * affinity5
            guide3 = attention3[:, iters:iters + 1, ...] * affinity3
            guide1 = attention1[:, iters:iters + 1, ...]
            
            guide7abs = attention7[:, iters:iters + 1, ...] * affinity7.abs()
            guide5abs = attention5[:, iters:iters + 1, ...] * affinity5.abs()
            guide3abs = attention3[:, iters:iters + 1, ...] * affinity3.abs()
            guide1abs = attention1[:, iters:iters + 1, ...]

            guide_abssum = torch.sum(guide7abs, dim=1).unsqueeze(1)
            guide_abssum += torch.sum(guide5abs, dim=1).unsqueeze(1)
            guide_abssum += torch.sum(guide3abs, dim=1).unsqueeze(1)
            guide_abssum += torch.sum(guide1abs, dim=1).unsqueeze(1)

            guide_sum = torch.sum(guide7, dim=1).unsqueeze(1)
            guide_sum += torch.sum(guide5, dim=1).unsqueeze(1)
            guide_sum += torch.sum(guide3, dim=1).unsqueeze(1)
            guide_sum += torch.sum(guide1, dim=1).unsqueeze(1)
            
            guide7 = torch.div(guide7, guide_abssum)
            guide5 = torch.div(guide5, guide_abssum)
            guide3 = torch.div(guide3, guide_abssum)
            guide1 = torch.div(guide1, guide_abssum)

            guide0 = 1 - guide_sum / guide_abssum

            # guidance
            weight_pad = []
            guide7_idx = guide5_idx = guide3_idx = 0
            for t in range(self.kernel_size * self.kernel_size):
                if weightmask_7[t]:
                    weight_pad.append(zero_pad[t](guide7[:, guide7_idx:guide7_idx + 1, :, :]))
                    guide7_idx += 1
                elif weightmask_5[t]:
                    weight_pad.append(zero_pad[t](guide5[:, guide5_idx:guide5_idx + 1, :, :]))
                    guide5_idx += 1
                elif weightmask_3[t]:
                    weight_pad.append(zero_pad[t](guide3[:, guide3_idx:guide3_idx + 1, :, :]))
                    guide3_idx += 1
                else:
                    weight_pad.append(zero_pad[t](guide1[:, 0:1, :, :]))
            weight_pad.append(zero_pad[self.kernel_size**2 // 2](guide0))

            guide_weight = torch.cat([weight_pad[t] for t in range(self.kernel_size * self.kernel_size + 1)], dim=1)

            # refine
            depth_pad = []
            for t in range(self.kernel_size * self.kernel_size):
                depth_pad.append(zero_pad[t](dt))
            depth_pad.append(zero_pad[self.kernel_size**2 // 2](d00))

            depth_all = torch.cat([depth_pad[t] for t in range(self.kernel_size * self.kernel_size + 1)], dim=1)
            refined_result = torch.sum((guide_weight.mul(depth_all)), dim=1)
            refined_output = refined_result[:, (self.kernel_size - 1) // 2:-(self.kernel_size - 1) // 2,
                              (self.kernel_size - 1) // 2:-(self.kernel_size - 1) // 2].unsqueeze(dim=1)
        return refined_output
