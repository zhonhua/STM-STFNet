import torch.nn as nn
import torch

from .encoder import Down
from .decoder import FineUp

from .aspp import CSAM_ASPP
class STMSTFNet(nn.Module):
    def __init__(self):
        super(STMSTFNet, self).__init__()

        self.residual_down = Down()
        self.diff_down=Down()
        self.fine_up = FineUp()

        self.ca1=CSAM_ASPP(6,6)
        self.ca2=CSAM_ASPP(6,6)


    def forward(self, c1, f1,c3,f3, c2):
        diff = torch.cat([c2, c1, f1], dim=1)

        residual = self.residual_down(diff)  # [8,16,512]

        diff1 = self.diff_down(torch.cat([c2, c3, f3], dim=1))

        cf12 = self.ca1(f1 - c1, c2 - c1, c2)
        f21 = cf12 + c2

        cf23 = self.ca2(f3 - c3, c3 - c2, c2)
        f32 = cf23 + c2

        f12 = self.fine_up(residual[4], residual)
        f12 = f12 + f1
        ff2 = f12 * 0.5 + f21 * 0.5

        f232 = self.fine_up(diff1[4], diff1)
        f23 = f3 - f232
        fb2 = f23 * 0.5 + f32 * 0.5

        output = ff2 * 0.5 + fb2 * 0.5

        return output,ff2,fb2



