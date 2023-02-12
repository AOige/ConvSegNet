import torch
import torch.nn as nn
from resnet import resnet50

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.network = nn.Sequential(
            Conv2D(in_c, out_c),
            Conv2D(out_c, out_c, kernel_size=1, padding=0, act=False)

        )
        self.shortcut = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_init):
        x = self.network(x_init)
        s = self.shortcut(x_init)
        x = self.relu(x+s)
        return x


class encoder(nn.Module):
    def __init__(self, ch):
        super().__init__()

        """ ResNet50 """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        """ Reduce feature channels """
        self.c1 = Conv2D(64, ch)
        self.c2 = Conv2D(256, ch)
        self.c3 = Conv2D(512, ch)
        self.c4 = Conv2D(1024, ch)

    def forward(self, x):
        """ Backbone: ResNet50 """
        x0 = x
        x1 = self.layer0(x0)    ## [-1, 64, h/2, w/2]
        x2 = self.layer1(x1)    ## [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)    ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)    ## [-1, 1024, h/16, w/16]

        c1 = self.c1(x1)
        c2 = self.c2(x2)
        c3 = self.c3(x3)
        c4 = self.c4(x4)

        return c1, c2, c3, c4

class context_feature_refinement(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.d1 = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.d2 = Conv2D(in_c, out_c, kernel_size=3, padding=1)
        self.d3 = Conv2D(in_c, out_c, kernel_size=7, padding=3)
        self.d4 = Conv2D(in_c, out_c, kernel_size=11, padding=5)
        self.c1 = Conv2D(out_c*4, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        x4 = self.d4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.c1(x)
        return x

class ConvSegNet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.encoder = encoder(64)

        """ Decoder """
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = context_feature_refinement(128, 64)
        self.c2 = context_feature_refinement(128, 64)
        self.c3 = context_feature_refinement(128, 64)
        self.s0 = Conv2D(3, 64)
        self.c4 = context_feature_refinement(128, 64)

        self.output = Conv2D(64, 1, kernel_size=1, padding=0)

    def forward(self, image):
        s0 = image
        s1, s2, s3, s4 = self.encoder(image)

        x = self.up(s4)
        x = torch.cat([x, s3], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c3(x)

        x = self.up(x)
        s0 = self.s0(s0)
        x = torch.cat([x, s0], axis=1)
        x = self.c4(x)

        y = self.output(x)
        return y

if __name__ == "__main__":
    inputs = torch.randn((2, 3, 256, 256))
    model = ConvSegNet()
    # y = model(inputs)
    # print(y.shape)
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
