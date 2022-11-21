import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

class DP3DF(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(DP3DF, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_BN_f = functools.partial(arch_util.ResidualBlock_IN2, nf=256)
        ResidualBlock_BN_f1 = functools.partial(arch_util.ResidualBlock_IN2, nf=128)
        ResidualBlock_BN_f2 = functools.partial(arch_util.ResidualBlock_IN2, nf=64)
        self.conv_first = nn.Conv2d(3 * 3, 64, 3, 1, 1, bias=True)
        self.conv_first_re = arch_util.make_layer(ResidualBlock_BN_f2, 2)
        self.conv_second = nn.Conv2d(64, 128, 4, 2, 1, bias=True)
        self.conv_second_re = arch_util.make_layer(ResidualBlock_BN_f1, 2)
        self.conv_third = nn.Conv2d(128, 256, 4, 2, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_BN_f, front_RBs)

        self.scale = 4 * 4
        self.recon_trunk = arch_util.make_layer(ResidualBlock_BN_f, back_RBs)
        self.upconv1 = nn.Conv2d(256, 128 * 4, 3, 1, 1, bias=True)
        self.upconv1_re = arch_util.make_layer(ResidualBlock_BN_f1, 2)
        self.upconv2 = nn.Conv2d(128, 64 * 4, 3, 1, 1, bias=True)
        self.upconv2_re = arch_util.make_layer(ResidualBlock_BN_f2, 2)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv2 = nn.Conv2d(64, 256, 3, 1, 1, bias=True)
        self.HRconv3 = nn.Conv2d(256, 512, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(512, (3 * 3 * 3 * 3 + 3) * self.scale, 3, 1, 1, bias=True)
        self.HRconv_residual = nn.Conv2d(64, 3 * 4 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle_final = nn.PixelShuffle(4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.reset_params()

    @staticmethod
    def weight_init(m, init_type='kaiming', gain=0.02, scale=0.1):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.weight.data *= scale
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames

        x_center = x[:, self.center-1:self.center+2, :, :, :].contiguous()
        x_center = x_center.view(B, 3*C, H, W)

        L1_fea1 = self.lrelu(self.conv_first(x_center))
        L1_fea1 = self.conv_first_re(L1_fea1)
        L1_fea2 = self.lrelu(self.conv_second(L1_fea1))
        L1_fea2 = self.conv_second_re(L1_fea2)
        L1_fea3 = self.lrelu(self.conv_third(L1_fea2))
        L1_fea = self.feature_extraction(L1_fea3)
        ##########################################################################################
        out_ill_s1_o = self.recon_trunk(L1_fea)
        #############################################################################################################
        out_ill_s2_o = self.lrelu(self.pixel_shuffle(self.upconv1(out_ill_s1_o)))
        out_ill_s2_o = self.upconv1_re(out_ill_s2_o)
        #############################################################################################################
        out_ill_s3_o = self.lrelu(self.pixel_shuffle(self.upconv2(out_ill_s2_o)))
        out_ill_s3_o = self.upconv2_re(out_ill_s3_o)
        #############################################################################################################
        out_ill_raw1 = self.lrelu(self.HRconv2(out_ill_s3_o))
        out_ill_raw2 = self.lrelu(self.HRconv3(out_ill_raw1))
        out_ill_raw = self.conv_last(out_ill_raw2)

        out_ill_residual = self.HRconv_residual(out_ill_s3_o)
        out_ill_residual = self.pixel_shuffle_final(out_ill_residual)
        out_ill_residual = nn.Tanh()(out_ill_residual)
        ##########################################################################################
        source_image = F.unfold(x_center, kernel_size=3, dilation=1, stride=1, padding=1)
        B, kh_kw, L = source_image.size()
        source_image = source_image.permute(0, 2, 1)
        source_image = source_image.view(B, H, W, 3, 3, 9)
        source_image = source_image.permute(0, 1, 2, 4, 3, 5).contiguous()
        source_image = source_image.view(B, H, W, 3, 3*9)
        source_image = source_image.unsqueeze(dim=-1)
        source_image = source_image.repeat(1, 1, 1, 1, 1, self.scale)
        out_ill_raw = out_ill_raw.view(B, -1, H*W)
        out_ill_raw = out_ill_raw.permute(0, 2, 1)
        out_ill_raw = out_ill_raw.view(B, H, W, -1)
        out_ill1 = out_ill_raw[:, :, :, 0:(3 * 3 * 3 * 3)*self.scale]
        out_ill2 = out_ill_raw[:, :, :, (3 * 3 * 3 * 3)*self.scale:(3 * 3 * 3 * 3 + 3)*self.scale]

        out_ill1 = out_ill1.view(B, H, W, 3, 3*9, self.scale)
        out_ill1 = nn.Softmax(dim=4)(out_ill1)

        out_ill2 = out_ill2.view(B, H, W, 3, self.scale)
        out_ill2 = nn.Sigmoid()(out_ill2)

        out_ill3 = out_ill2.view(B, H, W, -1)
        out_ill3 = out_ill3.permute(0, 3, 1, 2)
        out_ill3 = self.pixel_shuffle_final(out_ill3)

        out_ill4 = out_ill1.view(B, H, W, -1)
        out_ill4 = out_ill4.permute(0, 3, 1, 2)
        out_ill4 = self.pixel_shuffle_final(out_ill4)

        out_ill2 = 1.0 / torch.clamp(out_ill2, min=0.0000000001, max=1.0)

        base_ill = torch.sum(source_image * out_ill1, dim=4)
        base_ill = base_ill * out_ill2
        base_ill = base_ill.view(B, H, W, -1)
        base_ill = base_ill.permute(0, 3, 1, 2)
        base_ill1 = self.pixel_shuffle_final(base_ill)
        base_ill1 = torch.clamp(base_ill1, min=0.0, max=1.0)

        base_ill2 = base_ill1 + out_ill_residual
        base_ill2 = torch.clamp(base_ill2, min=0.0, max=1.0)
        return base_ill2, base_ill1, out_ill3, out_ill4
