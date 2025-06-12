import torch
import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LCNet"]

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                            stride=stride, padding=padding,
                            dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.SELU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output,max_pool], 1)

        output = self.bn_prelu(output)

        return output


def Split(x,p):
    c = int(x.size()[1])
    c1 = round(c * (1-p))
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


class TCA(nn.Module):
    def __init__(self, c, d=1, dropout=0, kSize=3, dkSize=3):
        super().__init__()
    
        self.conv3x3 = Conv(c, c, kSize, 1, padding=1, bn_acti=True)
    
        self.dconv3x3=Conv(c, c, (dkSize, dkSize), 1,
                            padding=(1, 1), groups=c, bn_acti=True)

        self.ddconv3x3 = Conv(c, c, (dkSize, dkSize), 1,
                              padding=(1 * d, 1 * d), groups=c, dilation=(d, d), bn_acti=True)
    
        self.bp = BNPReLU(c)

    def forward(self, input):
        br = self.conv3x3(input)
        
        br1 = self.dconv3x3(br)
        br2 = self.ddconv3x3(br)
        br = br + br1 + br2
        
        output = self.bp(br)
        return output


class PCT(nn.Module):
    def __init__(self, nIn, d=1, dropout=0, p = 0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1-p))
    
        self.TCA = TCA(c,d)
        
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=True)
    
    def forward(self, input):
        output1, output2 = Split(input,self.p)
        output2 = self.TCA(output2)
        
        output = torch.cat([output1, output2], dim=1)
        output = self.conv1x1(output)
        return output


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, d=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes//4)
        
        self.TCA = TCA(planes//4,2)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.TCA(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# class LCNet(nn.Module):
class submission_2023220397(nn.Module):
    # in_channels=1, num_classes=1): # 수업 조교님께서 제시한 코드 기준
    # def __init__(self, classes=19, block_1=3, block_2=7, C = 32, P=0.5): # LCNet 공식 깃허브 코드 기준
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        
        C = 32  # 고정값 사용
        P = 0.5
        block_1 = 3
        block_2 = 7

        self.Init_Block = nn.Sequential(

            # 변경 전
            # Conv(3, C, 3, 2, padding=1, bn_acti=True),

            # 변경 후
            Conv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            Conv(C, C, 3, 1, padding=1, bn_acti=True),
            Conv(C, C, 3, 1, padding=1, bn_acti=True)
        )
        
        dilation_block_1 = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        dilation_block_2 = [2, 4, 8, 16, 20, 24, 32]  # 점진적 증가 + 넓은 수용 영역 확보
        # dilation_block_2 = [2, 4, 8, 16, 24, 32, 48]  # 점진적 증가 + 넓은 수용 영역 확보
        # dilation_block_2 = [2, 4, 8, 12, 20, 28, 32]
        
        # Original 
        # dilation_block_2 = [4, 4, 8, 8,16,16,32,32,32,32,32,32]
        
        #Block 1
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C*2))
        
        for i in range(0, block_1):        
            #self.LC_Block_1.add_module("LC_Module_1_" + str(i), Bottleneck(C*2, C*2,d = dilation_block_1[i]))
            self.LC_Block_1.add_module("LC_Module_1_" + str(i), PCT(nIn = C*2, d = dilation_block_1[i], p = P ))
    
        #Block 2
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C*2, C*4))
        for i in range(0, block_2):
            #self.LC_Block_2.add_module("LC_Module_2_" + str(i), Bottleneck(C*4, C*4,d = dilation_block_2[i]))
            self.LC_Block_2.add_module("LC_Module_2_" + str(i), PCT(nIn = C*4, d = dilation_block_2[i], p = P ))

        self.DAD = DAD(C*4, C*2, num_classes) # num_classes = 1

    def forward(self, input):

        output0 = self.Init_Block(input)
        output1 = self.LC_Block_1(output0)
        output2 = self.LC_Block_2(output1)
    
        out = self.DAD(output1,output2)

        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)
        # out = TF.resize(out, size=input.shape[2:], interpolation=TF.InterpolationMode.BILINEAR)
        
        return out



# class DAD(nn.Module):
#     def __init__(self,c2, c1, classes):
#         super().__init__()
#         self.conv1x1_c = Conv(c2, c1, 1, 1, padding=0, bn_acti=True)
#         self.conv1x1_neg = Conv(c1, c1, 1, 1, padding=0, bn_acti=True)
        
#         self.conv3x3 = Conv(c1, c1, (3, 3), 1, padding=(1, 1), groups=c1, bn_acti=True)
#         self.conv1x1 = Conv(c1, classes, 1, 1, padding=0, bn_acti=True)

#     def forward(self, X, Y):
#         X_map = torch.sigmoid(X)
#         F_sg =  X_map
        
#         Yc = self.conv1x1_c(Y)
#         Yc_map = torch.sigmoid(Yc)
#         Neg_map = self.conv1x1_neg(-Yc_map)
#         F_rg = Neg_map*Yc_map + Yc
#         F_rg =  F.interpolate(F_rg, F_sg.size()[2:], mode='bilinear', align_corners=False)
        
#         output =  F_sg * F_rg
#         output = self.conv3x3(output)
#         output = self.conv1x1(output)
#         return output

# Modified 2
class DAD(nn.Module):
    """🔀 CBAM 적용 + Concat 기반 정보 융합"""
    def __init__(self, c2, c1, classes):
        super().__init__()
        self.conv1x1_c = Conv(c2, c1, 1, 1, padding=0, bn_acti=True)
        self.conv1x1_neg = Conv(c1, c1, 1, 1, padding=0, bn_acti=True)

        # self.cbam = CBAM(c1)  # 🔧 CBAM으로 채널+공간 강화
        self.attn = SEBlock(c1, reduction= 8)

        self.conv_fusion = Conv(c1 * 2, c1, 1, 1, padding=0, bn_acti=True)  # 🔁 1x1 conv 융합
        self.conv3x3 = Conv(c1, c1, (3, 3), 1, padding=(1, 1), groups=c1, bn_acti=True)
        self.conv1x1_out = Conv(c1, classes, 1, 1, padding=0, bn_acti=True)

    def forward(self, X, Y):
        X_map = torch.sigmoid(X)
        F_sg = X_map

        Yc = self.conv1x1_c(Y)
        Yc_map = torch.sigmoid(Yc)
        Neg_map = self.conv1x1_neg(-Yc_map)
        F_rg = Neg_map * Yc_map + Yc
        F_rg = F.interpolate(F_rg, F_sg.size()[2:], mode='bilinear', align_corners=False)

        # F_rg = self.cbam(F_rg)  # 🔧 CBAM 적용
        F_rg = self.attn(F_rg)

        fused = torch.cat([F_sg, F_rg], dim=1)  # 🔧 곱셈 대신 concat
        fused = self.conv_fusion(fused)         # 🔁 1x1 conv로 결합
        fused = self.conv3x3(fused)
        out = self.conv1x1_out(fused)
        return out



# ==== Modified 1 ======
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(self.avg_pool(x))  # (B, C, 1, 1)
        return x * scale



# 예시
# class submission_2023110214(nn.Module): 
#     def __init__(self, in_channels=1, num_classes=1): 
#         # - in_channels: 입력 영상의 채널 수를 지정합니다. 반드시 함수 인자로 받아야 합니다.
#         # - num_classes: 클래스(출력 채널) 수를 지정합니다. 반드시 함수 인자로 받아야 합니다.
#         #                바이너리 세그멘테이션의 경우 노드를 1개 갖도록 하므로 num_classes를 1로 가정합니다.
#         super().__init__()
#         init_features=32
#         features = init_features
#         self.encoder1 = self.block(in_channels, features)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder2 = self.block(features, features * 2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder3 = self.block(features * 2, features * 4)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder4 = self.block(features * 4, features * 8)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.bottleneck = self.block(features * 8, features * 16)

#         self.upconv4 = nn.ConvTranspose2d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = self.block((features * 8) * 2, features * 8)
#         self.upconv3 = nn.ConvTranspose2d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = self.block((features * 4) * 2, features * 4)
#         self.upconv2 = nn.ConvTranspose2d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = self.block((features * 2) * 2, features * 2)
#         self.upconv1 = nn.ConvTranspose2d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = self.block(features * 2, features)

#         self.conv = nn.Conv2d(
#             in_channels=features, out_channels=num_classes, kernel_size=1
#         )
#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))

#         bottleneck = self.bottleneck(self.pool4(enc4))

#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         dec1 = self.conv(dec1)       
#         return dec1

#     def block(self, in_channels, features):
#         return nn.Sequential(nn.Conv2d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,),
#                             nn.BatchNorm2d(num_features=features), 
#                              nn.ReLU(inplace=True),
#                              nn.Conv2d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,), nn.BatchNorm2d(num_features=features),nn.ReLU(inplace=True),)