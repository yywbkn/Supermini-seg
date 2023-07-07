
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import torch
from torch.nn import  Softmax
from einops import rearrange
# from networks_other import init_weights
# from ViTBlock import conv_1x1_bn,conv_nxn_bn,Transformer
from .networks_other import init_weights
from .ViTBlock import conv_1x1_bn,conv_nxn_bn,Transformer



def INF(B,H,W):
    # return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(y)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        # print('self.gamma = ', self.gamma)
        return out



class CC08CC08(nn.Module):
    def __init__(self,in_dim):
        super(CC08CC08, self).__init__()
        self.cc08 = CrissCrossAttention(in_dim=8)
        self.cc0802 = CrissCrossAttention(in_dim=8)

    def forward(self, x, y):
        output00 = self.cc08(x,y)
        # print('output00 shape = ', output00.shape)
        # output01 = self.cam08(output00.contiguous())
        output01 = self.cc0802(output00.contiguous(),output00.contiguous())
        return output01

class CC24CC24(nn.Module):
    def __init__(self,in_dim):
        super(CC24CC24, self).__init__()
        self.cc24 = CrissCrossAttention(in_dim=24)
        self.cc2402 = CrissCrossAttention(in_dim=24)

    def forward(self, x, y):
        output02 = self.cc24(x,y)
        output03 = self.cc2402(output02.contiguous(),output02.contiguous())
        return output03


class CC32CC32(nn.Module):
    def __init__(self,in_dim):
        super(CC32CC32, self).__init__()
        self.cc32 = CrissCrossAttention(in_dim=32)
        self.cc3202 = CrissCrossAttention(in_dim=32)

    def forward(self, x, y):
        output04 = self.cc32(x,y)
        output05 = self.cc3202(output04.contiguous(),output04.contiguous())
        return output05

class CC64CC64(nn.Module):
    def __init__(self,in_dim):
        super(CC64CC64, self).__init__()
        self.cc64 = CrissCrossAttention(in_dim=64)
        self.cc6402 = CrissCrossAttention(in_dim=64)

    def forward(self, x, y):
        output06 = self.cc64(x,y)
        output07 = self.cc6402(output06.contiguous(),output06.contiguous())
        return output07



class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DilatedParallelConvBlockD2, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, dilation=1, groups=out_planes, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=2, dilation=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        output = self.conv0(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        output = d1 + d2
        output = self.bn(output)
        return output


class AHSP(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(AHSP, self).__init__()
        assert out_planes % 4 == 0
        inter_planes = out_planes // 4
        self.conv1x1_down = nn.Conv2d(in_planes, inter_planes, 1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=1, dilation=1, groups=inter_planes, bias=False)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=2, dilation=2, groups=inter_planes, bias=False)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=4, dilation=4, groups=inter_planes, bias=False)
        self.conv4 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=8, dilation=8, groups=inter_planes, bias=False)
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.conv1x1_fuse = nn.Conv2d(out_planes, out_planes, 1, padding=0, groups=4, bias=False)
        self.attention = nn.Conv2d(out_planes, 4, 1, padding=0, groups=4, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)


    def forward(self, input):
        output = self.conv1x1_down(input)
        d1 = self.conv1(output)
        # print('d1 shape = ', d1.shape)
        d2 = self.conv2(output)
        # print('d2 shape = ', d2.shape)
        d3 = self.conv3(output)
        # print('d3 shape = ', d3.shape)
        d4 = self.conv4(output)
        # print('d4 shape = ', d4.shape)
        p = self.pool(output)
        d1 = d1 + p
        d2 = d1 + d2
        d3 = d2 + d3
        d4 = d3 + d4
        att = torch.sigmoid(self.attention(torch.cat([d1, d2, d3, d4], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        d3 = d3 + d3 * att[:, 2].unsqueeze(1)
        d4 = d4 + d4 * att[:, 3].unsqueeze(1)
        output = self.conv1x1_fuse(torch.cat([d1, d2, d3, d4], 1))
        output = self.act(self.bn(output))
        return output


class TransformerParallelConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(TransformerParallelConvBlock, self).__init__()
        assert out_planes % 2 == 0
        inter_planes = out_planes // 2
        # print('inter_planes = ',inter_planes)
        self.conv1x1_down = nn.Conv2d(in_planes, inter_planes, 1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=1, dilation=1, groups=inter_planes, bias=False)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=2, dilation=2, groups=inter_planes, bias=False)
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.conv1x1_fuse = nn.Conv2d(out_planes, out_planes, 1, padding=0, groups=4, bias=False)
        self.attention = nn.Conv2d(out_planes, 2, 1, padding=0, groups=2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

        dims = [16,32,64, 80, 96, 120, 144, 192, 240]
        dims01 = dims[0]
        # transformer 模块的初始化设置
        self.conv1_trans = conv_nxn_bn(in_planes, in_planes )
        self.conv2_trans  = conv_1x1_bn(in_planes, dims01)
        # self.transformer = Transformer(dims01, depth = 6, heads= 8, dim_head = 64, mlp_dim = dims[3])
        self.transformer = Transformer(dims01, depth=4, heads=10, dim_head=1, mlp_dim=dims[0])
        self.conv3_trans = conv_1x1_bn(dims01, in_planes)
        self.conv4_trans = conv_nxn_bn(2 * in_planes, out_planes)

        if out_planes <= 8:
            self.h = 16
        elif out_planes == 24:
            self.h = 4
        else:
            self.h = 2
        # self.new_model = MobileViTBlock(dim = out_planes*2 ,depth =1,channel = inter_planes, kernel_size=3, patch_size=(self.h, self.h),mlp_dim = int(out_planes * 4))

    def forward(self, input):
        #conv 系列操作
        output = self.conv1x1_down(input)
        p = self.pool(output)
        d1 = self.conv1(output)
        # print('d1 shape = ', d1.shape)
        d2 = self.conv2(output)
        # print('d2 shape = ', d2.shape)
        d1 = d1 + p
        d2 = d1 + d2
        att = torch.sigmoid(self.attention(torch.cat([d1, d2], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        output = self.conv1x1_fuse(torch.cat([d1, d2], 1))
        output = self.act(self.bn(output))

        # transformer 系列操作
        y = input.clone()
        # Local representations
        x = self.conv1_trans(input)
        x = self.conv2_trans(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.h, pw=self.h)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.h, w=w // self.h, ph=self.h,pw=self.h)
        # Fusion
        x = self.conv3_trans(x)
        x = torch.cat((x, y), 1)
        x = self.conv4_trans(x)
        output # Fusion= output + x

        # Fusion conv 和 transformer
        output = output + x
        return output


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels):
        super(_GridAttentionBlockND, self).__init__()

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = in_channels
        self.inter_channels = in_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        # Output transform
        self.W = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),bn(self.in_channels),)

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,kernel_size=1, stride=1, padding=0,bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

        self.operation_function = self._concatenation

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''
        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=False)

        f = F.relu(theta_x + phi_g, inplace=True)
        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        # return W_y, sigm_psi_f
        return W_y


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels):
        super(GridAttentionBlock2D, self).__init__(in_channels)

def split(x):
    c = int(x.size()[1])
    c1 = round(c // 2)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2



class Super_MiniSeg(nn.Module):
    def __init__(self, classes=1,P1=2, P2 = 2, P3=2, aux=True):
        super(Super_MiniSeg, self).__init__()
        self.D1 = int(P1 / 2)
        print('P1 = ', P1)
        print('D1 = ', self.D1)
        self.D2 = int(P2 / 2)
        print('P2 = ', P2)
        print('D2 = ', self.D2)
        self.D3 = int(P3 / 2)
        print('P3 = ', P3)
        print('D3 = ', self.D3)

        self.aux = aux

        self.long1 = AHSP(3,8,stride=1)
        self.down1 = AHSP(3,8,stride=1)
        self.long1_02 = AHSP(8, 8, stride=2)
        self.down1_02 = AHSP(8, 8, stride=2)
        self.oneXone = nn.Conv2d(8,1,1,stride=1,padding=0,bias=False)
        # self.level1 = AHSP(8,8)

        self.Gate_attention_08 = GridAttentionBlock2D(in_channels=8)
        # 第一阶段
        self.level1 = nn.ModuleList()
        self.level1_long = nn.ModuleList()
        for i in range(0, P1):
            self.level1.append(AHSP(8,8))
        for i in range(0, self.D1):
            self.level1_long.append(AHSP(8,8))


        self.CC_plus_8 = CrissCrossAttention(in_dim=8)
        # self.CC_CAM_8 = CC08CC08(in_dim=8)
        self.CC_CC_8 = CC08CC08(in_dim=8)
        self.cat1 = nn.Sequential(
                    nn.Conv2d(16, 16, 1, stride=1, padding=0, groups=1, bias=False),
                    nn.BatchNorm2d(16))

        self.long2 = AHSP(8, 24, stride=2)
        self.down2 = AHSP(8, 24, stride=2)

        # 第2阶段
        self.Gate_attention_24 = GridAttentionBlock2D(in_channels=24)
        self.level2 = nn.ModuleList()
        self.level2_long = nn.ModuleList()
        for i in range(0, P2):
            self.level2.append(AHSP(24, 24))
        for i in range(0, self.D2):
            self.level2_long.append(AHSP(24, 24))

        # self.CC_plus_24 = CrissCrossAttention(in_dim=24)
        self.CC_CC_24 = CC24CC24(in_dim=24)


        self.cat2 = nn.Sequential(
                    nn.Conv2d(48, 48, 1, stride=1, padding=0, groups=1, bias=False),
                    nn.BatchNorm2d(48))

        self.long3 = AHSP(24, 32, stride=2)
        self.down3 = AHSP(24, 32, stride=2)

        # 第3阶段
        self.Gate_attention_32 = GridAttentionBlock2D(in_channels=32)
        self.level3 = nn.ModuleList()
        self.level3_long = nn.ModuleList()
        for i in range(0, P3):
            self.level3.append(AHSP(32, 32))
        for i in range(0, self.D3):
            self.level3_long.append(AHSP(32, 32))


        # self.level3= AHSP(32, 32)
        self.CC_plus_32 = CrissCrossAttention(in_dim=32)
        self.CC_CC_32 = CC32CC32(in_dim=32)

        self.cat3 = nn.Sequential(nn.Conv2d(64, 64, 1, stride=1, padding=0, groups=1, bias=False), nn.BatchNorm2d(64))


        # self.up3_conv4 = DilatedParallelConvBlockD2(64, 32)
        self.up3_conv3 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(32)
        self.up3_act = nn.PReLU(32)

        self.up2_conv3 = DilatedParallelConvBlockD2(32, 24)
        self.up2_conv2 = nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(24)
        self.up2_act = nn.PReLU(24)

        self.up1_conv2 = DilatedParallelConvBlockD2(24, 8)
        self.up1_conv1 = nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(8)
        self.up1_act = nn.PReLU(8)

        self.CAM_plus_8 = CAM_Module(in_dim= 8)
        self.CAM_plus_24 = CAM_Module(in_dim=24)
        self.CAM_plus_32 = CAM_Module(in_dim=32)
        self.CAM_plus_64 = CAM_Module(in_dim= 64)

        if self.aux:
            self.pred4 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(64, classes, 1, stride=1, padding=0))
            self.pred3 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(32, classes, 1, stride=1, padding=0))
            self.pred2 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(24, classes, 1, stride=1, padding=0))
        self.pred1 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(8, classes, 1, stride=1, padding=0))

    def forward(self, input):
        long1 = self.long1(input)
        output1 = self.down1(input)

        edge01 = self.oneXone(long1)
        # print('edge01 shape = ', edge01.shape)
        edge02 = self.oneXone(output1)
        # print('edge02 shape = ', edge02.shape)

        long1 = self.long1_02(long1)
        output1 = self.down1_02(output1)

        # print('output1 shape = ', output1.shape)
        # output1_add = self.CC_plus_8(output1, long1)
        # output1_add = self.CC_CC_8(output1, long1)
        output1_add = self.Gate_attention_08(output1, long1)
        # print('output1_add type = ', type(output1_add))
        # print('output1_add shape = ', output1_add.shape)

        for i, layer in enumerate(self.level1):
            if i < self.D1:
                # print('-'*50)
                output1 = self.CC_CC_8(layer(output1_add), output1)
                long1 = self.CC_CC_8(self.level1_long[i](output1_add), long1)
                # output1_add = self.CC_plus_8(output1, long1)
            else:
                # print('-' * 10)
                output1 = self.CC_CC_8(layer(output1_add), output1)
                # output1_add = self.CC_plus_8(output1, long1)

        output1_cat = self.cat1(torch.cat([long1, output1], 1))
        output1_l, output1_r = split(output1_cat)

        # print('output1 type = ', type(output1))
        # print('output1 shape = ', output1.shape)
        # print('long1 shape = ', long1.shape)

        # long2 = self.long2(output1_l + long1)
        # output2 = self.down2(output1_r + output1)
        long2 = self.long2(self.CC_CC_8(output1_l, long1))
        output2 = self.down2(self.CC_CC_8(output1_r, output1))

        # print('output2 shape = ', output2.shape)
        # print('long2 shape = ', long2.shape)
        # output2_add = self.CC_plus_24(output2, long2)
        # output2_add = self.CC_CC_24(output2, long2)
        output2_add = self.Gate_attention_24(output2, long2)

        # print('output2_add type = ', type(output2_add))
        # print('output2_add shape = ', output2_add.shape)

        for i, layer in enumerate(self.level2):
            if i < self.D2:
                output2 = self.CC_CC_24(layer(output2_add), output2)
                long2 = self.CC_CC_24(self.level2_long[i](output2_add), long2)
            else:
                output2 = self.CC_CC_24(layer(output2_add), output2)


        # output2_add = output2 + long2

        output2_cat = self.cat2(torch.cat([long2, output2], 1))
        output2_l, output2_r = split(output2_cat)

        # print('output2 shape = ', output2.shape)
        # print('long2 shape = ', long2.shape)

        # long3 = self.long3(output2_l + long2)
        # output3 = self.down3(output2_r + output2)
        long3 = self.long3(self.CC_CC_24(output2_l, long2))
        output3 = self.down3(self.CC_CC_24(output2_r, output2))

        # output3_add = output3 + long3
        # output3_add = self.CC_CC_32(output3, long3)
        output3_add = self.Gate_attention_32(output3, long3)

        # output3 = self.CC_CC_32(self.level3(output3_add), output3)
        # output3_add = output3 + long3
        # print('output3_add type = ', type(output3_add))
        # print('output3_add shape = ', output3_add.shape)

        for i, layer in enumerate(self.level3):
            if i < self.D3:
                output3 = self.CC_CC_32(layer(output3_add), output3)
                long3 = self.CC_CC_32(self.level3_long[i](output3_add), long3)
            else:
                output3 = self.CC_CC_32(layer(output3_add), output3)

        up3_conv3 = self.up3_bn3(self.up3_conv3(output3))
        up3_conv3 = self.CAM_plus_32(up3_conv3)

        # print('up3_conv3 shape = ', up3_conv3.shape)
        # up3 = self.up3_act(up3_conv4 + up3_conv3)
        up3 = self.up3_act(up3_conv3)
        # up3 = self.up3_act(self.CC_CC_32(up3_conv4, up3_conv3))
        up3 = F.interpolate(up3, output2.size()[2:], mode='bilinear', align_corners=False)
        up2_conv3 = self.up2_conv3(up3)
        up2_conv2 = self.up2_bn2(self.up2_conv2(output2))
        up2_conv2 = self.CAM_plus_24(up2_conv2)
        # print('up2_conv2 shape = ', up2_conv2.shape)
        # up2 = self.up2_act(up2_conv3 + up2_conv2)
        up2 = self.up2_act(self.CC_CC_24(up2_conv3, up2_conv2))

        up2 = F.interpolate(up2, output1.size()[2:], mode='bilinear', align_corners=False)
        up1_conv2 = self.up1_conv2(up2)
        up1_conv1 = self.up1_bn1(self.up1_conv1(output1))
        up1_conv1 = self.CAM_plus_8(up1_conv1)
        # print('up1_conv1 shape = ', up1_conv1.shape)
        # up1 = self.up1_act(up1_conv2 + up1_conv1)
        up1 = self.up1_act(self.CC_CC_8(up1_conv2, up1_conv1))

        # print('up1 shape = ', up1.shape)
        # print('output1.size() = ', output1.size())
        # print('up2 shape = ', up2.shape)
        # print('up3 shape = ', up3.shape)
        # print('up4 shape = ', up4.shape)
        # print('self.pred4(up4) shape = ', self.pred4(up4).shape)

        if self.aux:
            pred3 = F.interpolate(self.pred3(up3), input.size()[2:], mode='bilinear', align_corners=False)
            pred2 = F.interpolate(self.pred2(up2), input.size()[2:], mode='bilinear', align_corners=False)
        pred1 = F.interpolate(self.pred1(up1), input.size()[2:], mode='bilinear', align_corners=False)

        # print('pred2 shape = ', pred2.shape)

        if self.aux:
            return (pred1, pred2, pred3, pred3,edge01,edge02 )
        else:
            return (pred1, )

