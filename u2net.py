import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import math
import time
# class REBNCONVs(nn.Module):
#     def __init__(self,in_ch=3,out_ch=3,dirate=1):
#         super(REBNCONVs,self).__init__()

#         self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
#         self.bn_s1 = nn.BatchNorm2d(out_ch)
#         self.relu_s1 = nn.ReLU(inplace=True)

#     def forward(self,x):

#         hx = x
#         xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

#         return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like1(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    return  src

def _upsample_like(src,tar):

    src_h,src_l = src
    tar_h,tar_l = tar
    src_h = F.upsample(src_h,size=tar_h.shape[2:],mode='bilinear')
    src_l = F.upsample(src_l,size=tar_l.shape[2:],mode='bilinear') if src_l is not None else None
    if src_l is None:
        return src_h
    return  src_h,src_l

def comb(src,tar):
    src_h,src_l = src
    tar_h,tar_l = tar
    return src_h+tar_h,src_l+tar_l

def cat6(src1,src2,src3,src4,src5,src6,dim):
    src_h1,src_l1 = src1
    src_h2,src_l2 = src2
    src_h3,src_l3 = src3
    src_h4,src_l4 = src4
    src_h5,src_l5 = src5
    src_h6,src_l6 = src6
    return torch.cat((src_h1,src_h2,src_h3,src_h4,src_h5,src_h6),dim),torch.cat((src_l1,src_l2,src_l3,src_l4,src_l5,src_l6),dim)

def cat(src,tar,dim):
    src_h,src_l = src
    tar_h,tar_l = tar
    return torch.cat((src_h,tar_h),dim),torch.cat((src_l,tar_l),dim)

class OctConv(nn.Conv2d):
    """
    Octave convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    oct_value : int, default 2
        Octave value.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 oct_alpha=0.5,
                 oct_mode="std",
                 oct_value=2):
        if isinstance(stride, int):
            stride = (stride, stride)
        self.downsample = (stride[0] > 1) or (stride[1] > 1)
        assert (stride[0] in [1, oct_value]) and (stride[1] in [1, oct_value])
        stride = (1, 1)
        if oct_mode == "first":
            in_alpha = 0.0
            out_alpha = oct_alpha
        elif oct_mode == "norm":
            in_alpha = oct_alpha
            out_alpha = oct_alpha
        elif oct_mode == "last":
            in_alpha = oct_alpha
            out_alpha = 0.0
        elif oct_mode == "std":
            in_alpha = 0.0
            out_alpha = 0.0
        else:
            raise ValueError("Unsupported octave convolution mode: {}".format(oct_mode))
        self.h_in_channels = int(in_channels * (1.0 - in_alpha))
        self.h_out_channels = int(out_channels * (1.0 - out_alpha))
        self.l_out_channels = out_channels - self.h_out_channels
        self.oct_alpha = oct_alpha
        self.oct_mode = oct_mode
        self.oct_value = oct_value
        super(OctConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.conv_kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups}

    def forward(self,x):
        hx,lx=x
        if self.oct_mode == "std":
            return F.conv2d(
                input=hx,
                weight=self.weight,
                bias=self.bias,
                **self.conv_kwargs), None

        if self.downsample:
            hx = F.avg_pool2d(
                input=hx,
                kernel_size=(self.oct_value, self.oct_value),
                stride=(self.oct_value, self.oct_value))

        hhy = F.conv2d(
            input=hx,
            weight=self.weight[0:self.h_out_channels, 0:self.h_in_channels, :, :],
            bias=self.bias[0:self.h_out_channels] if self.bias is not None else None,
            **self.conv_kwargs)

        if self.oct_mode != "first":
            hlx = F.conv2d(
                input=lx,
                weight=self.weight[0:self.h_out_channels, self.h_in_channels:, :, :],
                bias=self.bias[0:self.h_out_channels] if self.bias is not None else None,
                **self.conv_kwargs)

        if self.oct_mode == "last":
            hlx = F.interpolate(
                input=hlx,
                scale_factor=self.oct_value,
                mode="nearest")
            hy = hhy + hlx
            ly = None
            return hy

        lhx = F.avg_pool2d(
            input=hx,
            kernel_size=(self.oct_value, self.oct_value),
            stride=(self.oct_value, self.oct_value))
        lhy = F.conv2d(
            input=lhx,
            weight=self.weight[self.h_out_channels:, 0:self.h_in_channels, :, :],
            bias=self.bias[self.h_out_channels:] if self.bias is not None else None,
            **self.conv_kwargs)

        if self.oct_mode == "first":
            hy = hhy
            ly = lhy
            return hy, ly

        if self.downsample:
            hly = hlx
            llx = F.avg_pool2d(
                input=lx,
                kernel_size=(self.oct_value, self.oct_value),
                stride=(self.oct_value, self.oct_value))
        else:
            hly = F.interpolate(
                input=hlx,
                scale_factor=self.oct_value,
                mode="nearest")
            llx = lx
        lly = F.conv2d(
            input=llx,
            weight=self.weight[self.h_out_channels:, self.h_in_channels:, :, :],
            bias=self.bias[self.h_out_channels:] if self.bias is not None else None,
            **self.conv_kwargs)

        hy = hhy + hly
        ly = lhy + lly
        return hy, ly




class OctaveConv(nn.Module):
    """
    Octave convolution block with Batch normalization and ReLU/ReLU6 activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 oct_alpha=0.5,
                 padding=0,
                 oct_mode="last"):
        super(OctaveConv, self).__init__()
        self.last = (oct_mode == "last") or (oct_mode == "std")
        self.conv = OctConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False,
            oct_alpha=oct_alpha,
            oct_mode=oct_mode)

    def forward(self, x):
        ret = self.conv(x)
        return ret

class REBNCONV(nn.Module):
    """
    Octave convolution block with Batch normalization and ReLU/ReLU6 activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dirate=1,
                 oct_alpha=0.5,
                 oct_mode="norm",
                 bn_eps=1e-5):
        super(REBNCONV, self).__init__()
        self.activate = nn.ReLU(inplace=True)
        self.last = (oct_mode == "last") or (oct_mode == "std")
        out_alpha = 0.0 if self.last else oct_alpha
        h_out_channels = int(out_channels * (1.0 - out_alpha))
        l_out_channels = out_channels - h_out_channels
        self.conv = OctConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1*dirate,
            dilation=1*dirate,
            groups=1,
            bias=False,
            oct_alpha=oct_alpha,
            oct_mode=oct_mode)
        self.h_bn = nn.BatchNorm2d(
            num_features=h_out_channels,
            eps=bn_eps)
        if not self.last:
            self.l_bn = nn.BatchNorm2d(
                num_features=l_out_channels,
                eps=bn_eps)

    def forward(self, x):
        hx, lx = self.conv(x)
        hx = self.h_bn(hx)
        if self.activate:
            hx = self.activate(hx)
        if not self.last:
            lx = self.l_bn(lx)
            lx = self.activate(lx)
        return hx, lx



class MaxPool2dx(nn.Module):
    def __init__(self):
        super(MaxPool2dx, self).__init__()
        self.pool1  = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    def forward(self, x):
        x_h, x_l = x
        x_h = self.pool1(x_h)
        x_l = self.pool1(x_l) if x_l is not None else None
        return x_h, x_l



### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,first=False):
        super(RSU7,self).__init__()
        oct_mode = 'norm'
        if first:
            oct_mode = 'first'
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1,oct_mode = oct_mode)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1  = MaxPool2dx()

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2  = MaxPool2dx()

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3  = MaxPool2dx()

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4  = MaxPool2dx()

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5  = MaxPool2dx()

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(cat(hx7,hx6,1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(cat(hx6dup,hx5,1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(cat(hx5dup,hx4,1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(cat(hx4dup,hx3,1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(cat(hx3dup,hx2,1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(cat(hx2dup,hx1,1))

        return comb(hx1d, hxin)

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1  = MaxPool2dx()

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2  = MaxPool2dx()

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3  = MaxPool2dx()

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4  = MaxPool2dx()

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(cat(hx6,hx5,1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(cat(hx5dup,hx4,1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(cat(hx4dup,hx3,1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(cat(hx3dup,hx2,1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(cat(hx2dup,hx1,1))

        return comb(hx1d, hxin)

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1  = MaxPool2dx()

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2  = MaxPool2dx()

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3  = MaxPool2dx()

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(cat(hx5,hx4,1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(cat(hx4dup,hx3,1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(cat(hx3dup,hx2,1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(cat(hx2dup,hx1,1))

        return comb(hx1d, hxin)

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1  = MaxPool2dx()

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2  = MaxPool2dx()

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(cat(hx4,hx3,1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(cat(hx3dup,hx2,1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(cat(hx2dup,hx1,1))

        return comb(hx1d, hxin)

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(cat(hx4,hx3,1))
        hx2d = self.rebnconv2d(cat(hx3d,hx2,1))
        hx1d = self.rebnconv1d(cat(hx2d,hx1,1))

        return comb(hx1d, hxin)


### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,in_ch=2,out_ch=2):
        super(U2NETP,self).__init__()

        self.stage1 = RSU7(in_ch,16,64,first=True)
        self.pool12  = MaxPool2dx()

        self.stage2 = RSU6(64,16,64)
        self.pool23  = MaxPool2dx()

        self.stage3 = RSU5(64,16,64)
        self.pool34  = MaxPool2dx()

        self.stage4 = RSU4(64,16,64)
        self.pool45  = MaxPool2dx()

        self.stage5 = RSU4F(64,16,64)
        self.pool56  = MaxPool2dx()

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = OctaveConv(64,out_ch,3,padding=1)
        self.side2 = OctaveConv(64,out_ch,3,padding=1)
        self.side3 = OctaveConv(64,out_ch,3,padding=1)
        self.side4 = OctaveConv(64,out_ch,3,padding=1)
        self.side5 = OctaveConv(64,out_ch,3,padding=1)
        self.side6 = OctaveConv(64,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):
        #x1=torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        hx = x

        #stage 1
        hx1 = self.stage1((hx,None))
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(cat(hx6up,hx5,1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(cat(hx5dup,hx4,1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(cat(hx4dup,hx3,1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(cat(hx3dup,hx2,1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(cat(hx2dup,hx1,1))


        #side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)

        d2 = _upsample_like1(d2,d1)
        d3 = _upsample_like1(d3,d1)
        d4 = _upsample_like1(d4,d1)
        d5 = _upsample_like1(d5,d1)
        d6 = _upsample_like1(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
        d0 = _upsample_like1(d0,x)
        return x*F.relu(d0)#, F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

net = U2NETP()
d=torch.ones((1,2,512,512))
net.eval()
s=net(d)
t=time.time()
for i in range(10):
    s=net(d)
t=time.time()-t
print(t/10.0)
#dict=net.state_dict()
#torch.save(dict,"x.pt")