from network.basic_block import DySPN, weights_init
from torchvision.models.resnet import resnet34
from network.base_model import LightningBaseModel
import torch.nn as nn
import torch
import pytorch_lightning as pl


class decoder_layer(pl.LightningModule):

    def __init__(self, in_channel, mid_channel, out_channel):
        super(decoder_layer, self).__init__()
        self.decoder = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=3, padding=1), nn.ReLU(),
                                     nn.BatchNorm2d(mid_channel),
                                     nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=3, padding=1), nn.ReLU(),
                                     nn.BatchNorm2d(out_channel))

    def forward(self, de_feature, en_feature=None):
        de_feature = nn.functional.interpolate(de_feature, scale_factor=2, mode='bilinear', align_corners=True)
        if en_feature is not None:
            out = torch.cat([de_feature, en_feature], dim=1)
            out = self.decoder(out)
        else:
            out = self.decoder(de_feature)
        return out


class get_model(LightningBaseModel):

    def __init__(self, args, pretrained=True):
        super(get_model, self).__init__(args)
        net = resnet34(pretrained)
        self.hidden_layer = args['model_params']['hidden_layer']
        self.conv1 = self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.bottlenet = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(1024),
                                       nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(1024),
                                       nn.MaxPool2d(kernel_size=2, stride=2))

        self.decoder4 = decoder_layer(1024 + 512, 512, 512)
        self.decoder3 = decoder_layer(512 + 256, 256, 256)
        self.decoder2 = decoder_layer(256 + 128, 128, 128)
        self.decoder1 = decoder_layer(128 + 64, 64, 64)
        self.decoder0 = decoder_layer(64, 32, 32)

        self.coarse_output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )
        self.dyspn = DySPN(in_channels=32)
        weights_init(self)

    def forward(self, input):
        d = input['d']
        rgb = input['rgb']
        rgbd = torch.cat([rgb, d], dim=1)

        encoder_init = self.maxpool(self.relu(self.bn1(self.conv1(rgbd))))
        encoder1 = self.layer1(encoder_init)
        encoder2 = self.layer2(encoder1)
        encoder3 = self.layer3(encoder2)
        encoder4 = self.layer4(encoder3)

        bottlenet = self.bottlenet(encoder4)

        decoder4 = self.decoder4(bottlenet, encoder4)
        decoder3 = self.decoder3(decoder4, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)

        decoder_feature = self.decoder0(decoder1)

        coarse_output = self.coarse_output(decoder_feature)
        refined_output = self.dyspn(decoder_feature, coarse_output, d)

        input['coarse_output'] = coarse_output
        input['refined_output'] = refined_output

        return input
