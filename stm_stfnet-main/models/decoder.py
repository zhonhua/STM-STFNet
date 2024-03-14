import torch
import torch.nn as nn

from .swin_transformer import BasicLayer

class FineUp(nn.Module):
    def __init__(self, in_dim=64, down_scale=2, depths=(2, 2, 6, 2)):
        super(FineUp, self).__init__()
        self.down_scale = down_scale
        self.up1 = FineUpBlock(in_dim * 8*2 , in_dim * 4, 16 // down_scale, depths[3], 1)
        self.up2 = FineUpBlock(in_dim * 4*2 , in_dim * 2, 32 // down_scale, depths[2], 2)
        self.up3 = FineUpBlock(in_dim * 2*2 , in_dim, 64 // down_scale, depths[1], 4)
        self.up4 = FineUpBlock(in_dim*2 , in_dim, 128 // down_scale, depths[0], 8)

        self.outc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim, 6, 1, 1),
        )


    def forward(self, x,residual):


        x1=self.up1(x,residual[3])
        x2=self.up2(x1,residual[2])
        x3=self.up3(x2,residual[1])
        x4=self.up4(x3,residual[0])

        B, L, C = x4.shape

        x4 = x4.transpose(1, 2).view(B, C, 128 // self.down_scale, 128 // self.down_scale)

        output_fine = self.outc(x4)

        return output_fine


class FineUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, cur_depth, gaussian_kernel_size):
        super(FineUpBlock, self).__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.out_channels=out_channels
        self.up = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2 * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )

        self.layer2 = BasicLayer(dim=out_channels, input_resolution=(resolution, resolution),
                                 depth=cur_depth, num_heads=out_channels // 32, window_size=8, mlp_ratio=1,
                                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                                 norm_layer=nn.LayerNorm)

        self.proj1 = nn.Linear(in_channels//2, out_channels)
        self.proj2=nn.Linear(in_channels //2 , out_channels)

    def forward(self, x,residual):


        B, L, C = x.shape



        x= x.transpose(1, 2).view(B, C, self.resolution//2, self.resolution//2)

        x= self.up(x).flatten(2).transpose(1, 2)

        x = self.proj1(x)

        residual=self.proj2(residual)

        x = self.layer2(x)
        x=x+residual


        return x



