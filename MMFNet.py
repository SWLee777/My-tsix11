import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = []
        if transpose:
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class DSCResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DSCResBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=1):
        super(EBlock, self).__init__()
        self.layers = nn.Sequential(*[ResBlock(out_channel, out_channel) for _ in range(num_res)])

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=1):
        super(DBlock, self).__init__()
        self.layers = nn.Sequential(*[ResBlock(channel, channel) for _ in range(num_res)])

    def forward(self, x):
        return self.layers(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):
    def __init__(self, num_res=1):
        super(Encoder, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])
        self.feat_extract = nn.ModuleList([
            BasicConv(1, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=1),
        ])
        self.down12 = DownSample(base_channel * 2, base_channel * 2)
        self.down23 = DownSample(base_channel * 4, base_channel * 4)

    def forward(self, x):
        en_outputs = []
        en_1 = self.feat_extract[0](x)
        res1 = self.Encoder[0](en_1)
        en_outputs.append(res1)
        en_2 = self.feat_extract[1](res1)
        res2 = self.Encoder[1](en_2)
        x_12 = self.down12(res2)
        en_outputs.append(x_12)
        en_3 = self.feat_extract[2](x_12)
        res3 = self.Encoder[2](en_3)
        x_23 = self.down23(res3)
        en_outputs.append(x_23)
        return en_outputs


class Skip(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Skip, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3):
        return self.conv(torch.cat([x1, x2, x3], dim=1))


class Decoder(nn.Module):
    def __init__(self, num_res=1, is_seg=False, num_classes=6):
        super(Decoder, self).__init__()
        base_channel = 32
        self.is_seg = is_seg
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])
        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=3, relu=True, stride=1, transpose=True),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1)
        ])
        self.up32 = UpSample(base_channel * 2, base_channel * 2)
        self.up21 = UpSample(base_channel, base_channel)
        self.Skips = nn.ModuleList([
            Skip(base_channel * 7, base_channel * 1),
            Skip(base_channel * 7, base_channel * 2)
        ])
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])
        self.classifier = nn.Conv2d(base_channel, num_classes, kernel_size=1)

    def forward(self, inputs):
        inputs_12 = F.interpolate(inputs[0], scale_factor=0.5, mode='bilinear', align_corners=False)
        inputs_32 = F.interpolate(inputs[2], scale_factor=2, mode='bilinear', align_corners=False)
        inputs_21 = F.interpolate(inputs[1], scale_factor=2, mode='bilinear', align_corners=False)
        inputs_31 = F.interpolate(inputs_32, scale_factor=2, mode='bilinear', align_corners=False)
        skip_2 = self.Skips[1](inputs_12, inputs[1], inputs_32)
        skip_1 = self.Skips[0](inputs[0], inputs_21, inputs_31)
        res1 = self.Decoder[0](inputs[2])
        de_3 = self.feat_extract[0](res1)
        x_32 = self.up32(de_3)
        x_2 = self.Convs[0](torch.cat([x_32, skip_2], dim=1))
        res2 = self.Decoder[1](x_2)
        de_2 = self.feat_extract[1](res2)
        x_21 = self.up21(de_2)
        x_1 = self.Convs[1](torch.cat([x_21, skip_1], dim=1))
        res3 = self.Decoder[2](x_1)
        if self.is_seg:
            return self.classifier(res3)
        return self.feat_extract[2](res3)


class HaarDWT(nn.Module):
    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh, (h, w)


class HaarIDWT(nn.Module):
    def forward(self, ll, lh, hl, hh, original_hw):
        B, C, H, W = ll.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        out[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        out[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
        oh, ow = original_hw
        return out[:, :, :oh, :ow]


class WaveletFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ir_low_ratio=0.75):
        super(WaveletFusionBlock, self).__init__()
        self.ir_align = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.vi_align = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.ir_low_ratio = ir_low_ratio
        c = out_channels
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.high_att = nn.Sequential(
            nn.Conv2d(c * 2, c, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.high_guide = nn.Sequential(
            nn.Conv2d(c * 3, c, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.low_att = nn.Sequential(
            nn.Conv2d(c * 3, c, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.out_refine = nn.Sequential(
            BasicConv(c, c, kernel_size=3, stride=1, relu=True),
            BasicConv(c, c, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, feat_ir, feat_vi):
        feat_ir = self.ir_align(feat_ir)
        feat_vi = self.vi_align(feat_vi)
        ll_ir, lh_ir, hl_ir, hh_ir, ori_hw = self.dwt(feat_ir)
        ll_vi, lh_vi, hl_vi, hh_vi, _ = self.dwt(feat_vi)
        e_ir = torch.abs(lh_ir) + torch.abs(hl_ir) + torch.abs(hh_ir)
        e_vi = torch.abs(lh_vi) + torch.abs(hl_vi) + torch.abs(hh_vi)
        wh = self.high_att(torch.cat([e_ir, e_vi], dim=1))
        lh_f = wh * lh_vi + (1.0 - wh) * lh_ir
        hl_f = wh * hl_vi + (1.0 - wh) * hl_ir
        hh_f = wh * hh_vi + (1.0 - wh) * hh_ir
        h_f = torch.cat([lh_f, hl_f, hh_f], dim=1)
        h_g = self.high_guide(h_f)
        wl = self.low_att(torch.cat([ll_ir, ll_vi, h_g], dim=1))
        ll_base = self.ir_low_ratio * ll_ir + (1.0 - self.ir_low_ratio) * ll_vi
        ll_cross = wl * ll_vi + (1.0 - wl) * ll_ir
        ll_f = 0.7 * ll_base + 0.3 * ll_cross
        out = self.idwt(ll_f, lh_f, hl_f, hh_f, ori_hw)
        return self.out_refine(out) + 0.05 * (feat_ir + feat_vi)


class FFBlock(WaveletFusionBlock):
    pass


class SFBlock(nn.Module):
    def __init__(self):
        super(SFBlock, self).__init__()
        self.body_1 = DSCResBlock(32, 32)
        self.body_2 = DSCResBlock(64, 64)
        self.body_3 = DSCResBlock(128, 128)

    def forward(self, inp_feats):
        return [self.body_1(inp_feats[0]), self.body_2(inp_feats[1]), self.body_3(inp_feats[2])]


class SALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv_du(x)


class FDBlock(nn.Module):
    def __init__(self, in_channels=[32, 64, 128]):
        super(FDBlock, self).__init__()
        self.sa_0 = SALayer(in_channels[0], reduction=4, bias=False)
        self.sa_1 = SALayer(in_channels[1], reduction=4, bias=False)
        self.sa_2 = SALayer(in_channels[2], reduction=4, bias=False)
        self.sa_3 = SALayer(in_channels[0], reduction=4, bias=False)
        self.sa_4 = SALayer(in_channels[1], reduction=4, bias=False)
        self.sa_5 = SALayer(in_channels[2], reduction=4, bias=False)

    def forward(self, fus_fea, modality_fea):
        x_0 = fus_fea[0] - self.sa_0(modality_fea[0]) + 0.5*self.sa_3(modality_fea[0])
        x_1 = fus_fea[1] - self.sa_1(modality_fea[1]) + 0.5*self.sa_4(modality_fea[1])
        x_2 = fus_fea[2] - self.sa_2(modality_fea[2]) + 0.5*self.sa_5(modality_fea[2])
        return [x_0, x_1, x_2]


class SelfAttention(nn.Module):
    def __init__(self, in_channels, h):
        super(SelfAttention, self).__init__()
        assert in_channels % h == 0
        self.head_channels = in_channels // h
        self.h = h
        self.que_proj = nn.Linear(in_channels, self.h * self.head_channels)
        self.key_proj = nn.Linear(in_channels, self.h * self.head_channels)
        self.val_proj = nn.Linear(in_channels, self.h * self.head_channels)
        self.att_gate = nn.Sequential(
            nn.Linear(self.h * self.head_channels, self.h),
            nn.Dropout(0.4),
            nn.Linear(self.h, self.h * self.head_channels),
            nn.SiLU()
        )
        self.out_proj = nn.Linear(self.h * self.head_channels, in_channels)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B, N = x.shape[:2]
        q = self.que_proj(x).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)
        k = self.key_proj(x).view(B, N, self.h, self.head_channels).permute(0, 2, 3, 1)
        v = self.val_proj(x).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)
        att = torch.softmax(torch.matmul(q, k) / np.sqrt(self.head_channels), -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, N, self.h * self.head_channels)
        return self.out_proj(self.att_gate(out) * out)


class CrossTaskModule(nn.Module):
    def __init__(self, in_channels, high=32, width=32):
        super(CrossTaskModule, self).__init__()
        self.in_channels = in_channels
        self.high = high
        self.width = width
        self.avgpool = nn.AdaptiveAvgPool2d((self.high, self.width))
        self.ln_output = nn.LayerNorm(in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.sa = SelfAttention(in_channels, 8)
        self.mlp1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.ReLU(), nn.Conv2d(in_channels, in_channels, 1))
        self.mlp2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.ReLU(), nn.Conv2d(in_channels, in_channels, 1))

    def forward(self, fus, seg):
        B, C, H, W = fus.shape
        fus_fea = self.bn1(self.avgpool(fus))
        seg_fea = self.bn2(self.avgpool(seg))
        fus_flat = fus_fea.view(B, C, -1)
        seg_flat = seg_fea.view(B, C, -1)
        mt_flat = self.sa(torch.cat((fus_flat, seg_flat), dim=2).permute(0, 2, 1))
        mt_flat = self.ln_output(mt_flat)
        mt_flat = mt_flat.view(B, 2, self.high, self.width, self.in_channels)
        mt = mt_flat.permute(0, 1, 4, 2, 3)
        return self.mlp1(mt[:, 0]), self.mlp2(mt[:, 1])


class CrossAttention(nn.Module):
    def __init__(self, in_channels, h):
        super(CrossAttention, self).__init__()
        assert in_channels % h == 0
        self.head_channels = in_channels // h
        self.h = h
        self.que_proj = nn.Linear(in_channels, self.h * self.head_channels)
        self.key_proj = nn.Linear(in_channels, self.h * self.head_channels)
        self.val_proj = nn.Linear(in_channels, self.h * self.head_channels)
        self.out_proj = nn.Linear(self.h * self.head_channels, in_channels)
        self.att_gate = nn.Sequential(
            nn.Linear(self.h * self.head_channels, self.h),
            nn.Dropout(0.4),
            nn.Linear(self.h, self.h * self.head_channels),
            nn.SiLU()
        )

    def forward(self, ctf, st):
        B, N = st.shape[:2]
        st_q = self.que_proj(ctf).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)
        ctf_k = self.key_proj(ctf).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)
        ctf_v = self.val_proj(ctf).view(B, N, self.h, self.head_channels).permute(0, 2, 3, 1)
        att = torch.softmax(torch.matmul(ctf_k, ctf_v) / np.sqrt(self.head_channels), -1)
        query_res = torch.matmul(att, st_q).permute(0, 2, 1, 3).contiguous().view(B, N, self.h * self.head_channels)
        return self.out_proj(query_res * self.att_gate(query_res))


class CTQModule(nn.Module):
    def __init__(self, in_channels, high=32, width=32):
        super(CTQModule, self).__init__()
        self.in_channels = in_channels
        self.high = high
        self.width = width
        self.avgpool = nn.AdaptiveAvgPool2d((self.high, self.width))
        self.ln_output = nn.LayerNorm(self.in_channels)
        self.cs = CrossAttention(self.in_channels, 8)

    def forward(self, ctf, st):
        B, C, H, W = st.shape
        st_fea = self.avgpool(st)
        ctf_flat = ctf.view(B, C, -1).permute(0, 2, 1)
        st_flat = st_fea.view(B, C, -1).permute(0, 2, 1)
        query_out = self.cs(ctf_flat, st_flat)
        query_out = self.ln_output(query_out)
        query_out = query_out.permute(0, 2, 1).contiguous().view(B, C, self.high, self.width)
        query_out = F.interpolate(query_out, size=([H, W]), mode='bilinear', align_corners=False)
        return st + query_out


class MMFNet(nn.Module):
    def __init__(self, num_classes):
        super(MMFNet, self).__init__()
        self.encoder_1 = Encoder()
        self.encoder_2 = Encoder()
        self.decoder = Decoder(is_seg=False)
        self.ffb_1 = FFBlock(in_channels=32, out_channels=32)
        self.ffb_2 = FFBlock(in_channels=64, out_channels=64)
        self.ffb_3 = FFBlock(in_channels=128, out_channels=128)
        self.sfb = SFBlock()
        self.ctf_1 = CrossTaskModule(in_channels=32, high=32, width=32)
        self.ctf_2 = CrossTaskModule(in_channels=64, high=16, width=16)
        self.ctf_3 = CrossTaskModule(in_channels=128, high=8, width=8)
        self.ctq_fus_1 = CTQModule(in_channels=32, high=32, width=32)
        self.ctq_seg_1 = CTQModule(in_channels=32, high=32, width=32)
        self.ctq_fus_2 = CTQModule(in_channels=64, high=16, width=16)
        self.ctq_seg_2 = CTQModule(in_channels=64, high=16, width=16)
        self.ctq_fus_3 = CTQModule(in_channels=128, high=8, width=8)
        self.ctq_seg_3 = CTQModule(in_channels=128, high=8, width=8)
        self.fdb_1 = FDBlock()
        self.fdb_2 = FDBlock()
        self.seghead = Decoder(is_seg=True, num_classes=num_classes)

    def channel_relative_probability(self, prob_batch):
        total_prob = torch.sum(prob_batch, dim=1, keepdim=True)
        sum_others = total_prob - prob_batch
        relative_prob = torch.sigmoid(100 * (prob_batch - sum_others))
        relative_prob = torch.sum(relative_prob, dim=[0, 2, 3])
        return F.normalize(relative_prob.float().to(prob_batch.device), p=2, dim=-1, eps=1e-8)

    def forward(self, ir, rgb):
        ir_feas = self.encoder_1(ir)
        rgb_feas = self.encoder_2(rgb)
        recon_ir = self.decoder(ir_feas)
        recon_rgb = self.decoder(rgb_feas)
        fus_feas = [
            self.ffb_1(ir_feas[0], rgb_feas[0]),
            self.ffb_2(ir_feas[1], rgb_feas[1]),
            self.ffb_3(ir_feas[2], rgb_feas[2]),
        ]
        seg_feas = self.sfb(fus_feas)
        ctf_fus_1, ctf_seg_1 = self.ctf_1(fus_feas[0], seg_feas[0])
        ctf_fus_2, ctf_seg_2 = self.ctf_2(fus_feas[1], seg_feas[1])
        ctf_fus_3, ctf_seg_3 = self.ctf_3(fus_feas[2], seg_feas[2])
        query_fus = [
            self.ctq_fus_1(ctf_fus_1, fus_feas[0]),
            self.ctq_fus_2(ctf_fus_2, fus_feas[1]),
            self.ctq_fus_3(ctf_fus_3, fus_feas[2]),
        ]
        query_seg = [
            self.ctq_seg_1(ctf_seg_1, seg_feas[0]),
            self.ctq_seg_2(ctf_seg_2, seg_feas[1]),
            self.ctq_seg_3(ctf_seg_3, seg_feas[2]),
        ]
        dec_ir = self.fdb_1(query_fus, rgb_feas)
        dec_rgb = self.fdb_2(query_fus, ir_feas)
        recon_dec_ir = self.decoder(dec_ir)
        recon_dec_rgb = self.decoder(dec_rgb)
        seg_res = self.seghead(query_seg)
        fus = self.decoder(query_fus)
        _ = self.channel_relative_probability(seg_res)
        return recon_ir, recon_rgb, recon_dec_ir, recon_dec_rgb, seg_res, fus
