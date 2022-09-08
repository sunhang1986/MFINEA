import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResnetGlobalAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ResnetGlobalAttention, self).__init__()

        self.feature_channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.soft = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        zx = y.squeeze(-1)
        zy = zx.permute(0, 2, 1)
        zg = torch.matmul(zy, zx)

        batch = zg.shape[0]
        v = zg.squeeze(-1).permute(1, 0).expand((self.feature_channel, batch))
        v = v.unsqueeze_(-1).permute(1, 2, 0)

        atten = self.conv(y.squeeze(-1).transpose(-1, -2))
        atten = atten + v
        atten = self.conv_end(atten)
        atten = atten.permute(0,2,1).unsqueeze(-1)

        atten_score = self.soft(atten)

        return x * atten_score


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)


    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)


        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class MFF(nn.Module):

    def __init__(self, feature_channel=64, gamma=2, b=1):
        super(MFF, self).__init__()


        if feature_channel == 128:
            self.conv1 = nn.Sequential(nn.Conv2d(feature_channel // 2, feature_channel, kernel_size=3, stride=2, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv2 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.ConvTranspose2d(feature_channel*2, feature_channel, kernel_size=3, stride=2, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))

            self.feature_channel = feature_channel


        elif feature_channel == 64:
            self.conv1 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))
            self.conv2 = nn.Sequential(nn.ConvTranspose2d(feature_channel * 2, feature_channel, kernel_size=3, stride=2, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.ConvTranspose2d(feature_channel * 4, feature_channel, kernel_size=5, stride=4, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))

            self.feature_channel = feature_channel

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(feature_channel // 4, feature_channel, kernel_size=5, stride=4, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv2 = nn.Sequential(nn.Conv2d(feature_channel // 2, feature_channel, kernel_size=3, stride=2, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))

            self.feature_channel = feature_channel


        t = int(abs((math.log(feature_channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.con_1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.con_1_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.con_2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.con_2_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)


        self.con_3 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.con_3_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)


        self.avg_pool = nn.AdaptiveAvgPool2d(1)


        self.softmax = nn.Softmax(dim=1)

    def forward(self, f1, f2, f3):

        feature1 = self.conv1(f1).unsqueeze_(dim=1)
        feature2 = self.conv2(f2).unsqueeze_(dim=1)
        feature3 = self.conv3(f3).unsqueeze_(dim=1)

        feature12 = torch.cat([feature1, feature2],dim=1)
        feature123 = torch.cat([feature12, feature3],dim=1)

        fea_U = torch.sum(feature123, dim=1)
        u = self.avg_pool(fea_U)

        zx = u.squeeze(-1)

        zy = zx.permute(0,2,1)
        zg = torch.matmul(zy, zx)
        batch = zg.shape[0]
        v = zg.squeeze(-1).permute(1,0).expand((self.feature_channel,batch))
        v = v.unsqueeze_(-1).permute(1,2,0)

        vector1 = self.con_1(u.squeeze(-1).transpose(-1, -2)) + v
        vector1 = self.con_1_end(vector1)

        vector2 = self.con_2(u.squeeze(-1).transpose(-1, -2)) + v
        vector2 = self.con_2_end(vector2)

        vector3 = self.con_3(u.squeeze(-1).transpose(-1, -2)) + v
        vector3 = self.con_3_end(vector3)


        vector12 = torch.cat([vector1, vector2], dim=1)
        vector123 = torch.cat([vector12, vector3], dim=1)

        attention_vectors = self.softmax(vector123)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feature123 * attention_vectors).sum(dim=1)

        return fea_v

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y



class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res



class Base_Model(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect', n_blocks=6):
        super(Base_Model,self).__init__()


        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.InstanceNorm2d(ngf),
                                   nn.ReLU(True))

        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(ngf*2),
                                   nn.ReLU(True))

        self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(ngf * 4),
                                   nn.ReLU(True))

        self.mff1 = MFF(64)
        self.mff2 = MFF(128)
        self.mff3 = MFF(256)

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU(True)
        model_res = []
        for i in range(n_blocks):
            model_res += [ResnetBlock(ngf * 4, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.model_res = nn.Sequential(*model_res)


        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf*2),
                                 nn.ReLU(True))


        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf),
                                 nn.ReLU(True))

        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())


        self.pa1 = PALayer(256)
        self.pa2 = PALayer(128)
        self.pa3 = PALayer(64)

        self.ca1 = ResnetGlobalAttention(256)
        self.ca2 = ResnetGlobalAttention(128)
        self.ca3 = ResnetGlobalAttention(64)


    def forward(self, input):


        x_down1 = self.down1(input)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        mff_fa1 = self.mff1(x_down1, x_down2, x_down3)
        mff_fa2 = self.mff2(x_down1, x_down2, x_down3)
        mff_fa3 = self.mff3(x_down1, x_down2, x_down3)

        x6 = self.model_res(x_down3)

        x6 = self.ca1(x6)
        x6 = self.pa1(x6)


        x_up1 = self.up1(x6 + mff_fa3)
        x_up1 = self.ca2(x_up1)
        x_up1 = self.pa2(x_up1)

        x_up2 = self.up2(x_up1 + mff_fa2)
        x_up2 = self.ca3(x_up2)
        x_up2 = self.pa3(x_up2)

        x_up3 = self.up3(x_up2 + mff_fa1)


        return x_up3


