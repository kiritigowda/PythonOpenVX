#!/usr/bin/python
from pythonic import *
from vx_utils import *
from PIL import Image as PIL_Image

def EqualizationGraph():
    pil_img = PIL_Image.open('baboon.bmp')
    pil_img = pil_img.convert('L')    
    width = pil_img.width
    height = pil_img.height
    with Graph(verify=True) as g:
	vxImage = pil2vx_rgb(g.vx_context, pil_img)
	input_img = Image(g.context, vx_img=vxImage)
        output = Image(g.context, width, height, Color.VX_DF_IMAGE_RGB)
        yuv_src = Image(g.context, width, height, Color.VX_DF_IMAGE_IYUV)
        ColorConvertNode(g, input_img, yuv_src)
        y_img = ChannelExtractNode(g, yuv_src, Channel.VX_CHANNEL_Y)
        u_img = ChannelExtractNode(g, yuv_src, Channel.VX_CHANNEL_U)
        v_img = ChannelExtractNode(g, yuv_src, Channel.VX_CHANNEL_V)
        out_y = EqualizeHistNode(g, y_img)
        out_yuv = ChannelCombineNode(g, out_y, u_img, v_img, color=Color.VX_DF_IMAGE_IYUV)
        ColorConvertNode(g, out_yuv, output)
    g.vxProcessGraph()
    out_img = vx2np(output.image)
    PIL_Image.fromarray(out_img).show('out_img')
    pil_img.show('original')

EqualizationGraph()
