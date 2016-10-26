#!/usr/bin/python
from pythonic import *
from vx_utils import *
from PIL import Image as PIL_Image


def sobel_graph():
    pil_img = PIL_Image.open('baboon.bmp')
    pil_img = pil_img.convert('L')
    width = pil_img.width
    height = pil_img.height
    with Graph(verify=True) as g:
	vxImage = pil2vx(g.vx_context, pil_img)
	input_img = Image(g.context, vx_img=vxImage)
        yuyv = Image(g.context, width, height, Color.VX_DF_IMAGE_YUYV)
        out = Image(g.context, width, height, Color.VX_DF_IMAGE_RGB)
        y = ChannelExtractNode(g, yuyv, Channel.VX_CHANNEL_Y)
        gx, gy = Sobel3x3Node(g, input_img)
        mag = MagnitudeNode(g, gx, gy)
        shift = Scalar(g.context, Data_type.VX_TYPE_INT32, 0)
        converted = ConvertDepthNode(g, mag, Policy.VX_CONVERT_POLICY_WRAP, color=Color.VX_DF_IMAGE_U8, scalar=shift)
        ChannelCombineNode(g, converted, converted, converted, output=out)
    g.vxProcessGraph()
    out_img = vx2np(out.image)
    PIL_Image.fromarray(out_img).show('out_img')
    pil_img.show('original')

sobel_graph()















