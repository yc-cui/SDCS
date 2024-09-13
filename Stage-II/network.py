import torch
import torch.nn as nn



CHANNELS = (16, 32, 64, 128, 128) # CIA 
# CHANNELS = (16, 32, 64, 128, 256) # LGC Tianjin Daxing AHB

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.ReflectionPad2d(padding),
                                  nn.Conv2d(input_size, output_size, kernel_size, stride, 0, bias=bias))

        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)

        return self.act(out)

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)

class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super(ResBlock, self).__init__()
        residual = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels,kernel_size, stride, 0, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels,kernel_size, stride, 0, bias=bias),
        ]
        self.residual = nn.Sequential(*residual)


    def forward(self, inputs):
        trunk = self.residual(inputs)
        return trunk + inputs


class FeatureExtract(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtract, self).__init__()
        channels = CHANNELS
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 1, 3),
            ResBlock(channels[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1),
            ResBlock(channels[1]),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            ResBlock(channels[2]),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),
            ResBlock(channels[3]),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, 2, 1),
            ResBlock(channels[4]),
        )

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        return [l1, l2, l3, l4, l5]


class RFLU(nn.Module):
    def __init__(self):
        super(RFLU, self).__init__()

    def forward(self, c, l, a, d):
        leaka=0.1*a
        z = torch.where(a >= 0, torch.where(d >= 0, a+l, leaka+l), torch.where(d >= 0, leaka, leaka+c) )
          
        return z

def calc_mean_std(features):
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.rflu = RFLU()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels))
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels))


    def forward(self, inputs):
        # csbs lr_t2 hr_t1
        
        # LGC AHb Tianjin Daxing
        l1 = 0.5 * inputs[0] + 0.5 * inputs[2]
        l1 = self.conv1(l1)
        # CIA
        # l1 = self.conv1(inputs[0]) 
        
        a1 = adain(l1, inputs[1])
        d1 = adain(inputs[2],  inputs[1])
        rf1= self.rflu(inputs[1], l1, a1, d1)
        result = self.conv2(rf1) 
        
        return result


class SDCSNet(nn.Module):
    def __init__(self, bands):
        super(SDCSNet, self).__init__()

        self.CSBSNet = FeatureExtract(bands)
        self.LSNet = FeatureExtract(bands)
        self.HSNet = FeatureExtract(bands)

        channels = CHANNELS
        self.fusion_blocks = nn.ModuleList()
        for i in range(len(channels)):
            self.fusion_blocks.append(FeatureFusion(in_channels=channels[i]))

        self.conv1 = nn.Sequential(
            DeconvBlock(channels[4] * 2, channels[3], 4, 2, 1, bias=True),
            ResBlock(channels[3]),
        )
        self.conv2 = nn.Sequential(
            DeconvBlock(channels[3] * 2, channels[2], 4, 2, 1, bias=True),
            ResBlock(channels[2]),
        )
        self.conv3 = nn.Sequential(
            DeconvBlock(channels[2] * 2, channels[1], 4, 2, 1, bias=True),
            ResBlock(channels[1]),
        )
        self.conv4 = nn.Sequential(
            DeconvBlock(channels[1] * 2, channels[0], 4, 2, 1, bias=True),
            ResBlock(channels[0]),
        )
        self.conv5 = nn.Sequential(
            ResBlock(channels[0] * 2),
            nn.Conv2d(channels[0] * 2, channels[0], 1, 1, 0),
            ResBlock(channels[0]),
            nn.Conv2d(channels[0], bands, 1, 1, 0),
        )

    def forward(self, CSBS, LR_t2, HR_t1):

        CSBS_fearures = self.CSBSNet(CSBS)
        LR_features = self.LSNet(LR_t2)
        HR_features = self.HSNet(HR_t1)

        features = []
        for block, HS, CSBS, LS in zip(self.fusion_blocks, HR_features, CSBS_fearures, LR_features):
            features.append(block([CSBS, LS, HS]))

        l5 = self.conv1(torch.cat((features[4], CSBS_fearures[4]), dim=1))
        l4 = self.conv2(torch.cat((features[3], l5), dim=1))
        l3 = self.conv3(torch.cat((features[2], l4), dim=1))
        l2 = self.conv4(torch.cat((features[1], l3), dim=1))
        l1 = self.conv5(torch.cat((features[0], l2), dim=1))

        return l1