import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import ChannelGate
from .cbam import SpatialGate
import torch
import torch.nn as nn


class ASPP(nn.Module):
    def __init__(self,dim_in,dim_out,rate=1,bn_mom=0.1):
        super(ASPP,self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        [b, c, row, col] = x.size()
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return x1,x2,x3,x4

class CSAM_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(CSAM_ASPP, self).__init__()
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 4, dim_out, 1, 1, padding=0, bias=True),

        )


        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=dim_out * 8, out_channels=dim_out * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out * 4, momentum=bn_mom),
            nn.ReLU(),
        )

        self.branch1 = ASPP(dim_in, dim_out)
        self.branch2 = ASPP(dim_in, dim_out)
        self.branch3 = ASPP(dim_in, dim_out)
        self.channel = ChannelGate(dim_in * 4)
        self.spatial1 = SpatialGate()


    def forward(self, cf,cc,c2):

        cc1, cc2, cc3, cc4 = self.branch1(cc)
        feature_cat = torch.cat([cc1, cc2, cc3, cc4], dim=1)
        cha = self.channel(feature_cat)
        spa = self.spatial1(feature_cat)
        csa=cha*spa


        c1, c2, c3, c4 = self.branch2(c2)
        feature_cat1 = torch.cat([c1, c2, c3, c4], dim=1)



        cf1, cf2, cf3, cf4 = self.branch3(cf)
        feature_cat2 = torch.cat([cf1, cf2, cf3, cf4], dim=1)

        result1 = feature_cat2 * csa
        result2=self.conv2(torch.cat([result1,feature_cat1],dim=1))
        result = self.conv_cat(result2+feature_cat2)

        return result+cf



