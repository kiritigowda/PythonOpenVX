#from pyvx import vx
import pyvx.vx as vx
import numpy as np
from array import array


def pil2vx(context, pil_image):
    height = pil_image.height
    width = pil_image.width
    addr = vx.imagepatch_addressing_t(width, height, 1, width, vx.SCALE_UNITY, vx.SCALE_UNITY, 1, 3)
    arr = list(pil_image.getdata())
    npArr = np.reshape(arr, (-1,width))
    npArr = np.uint8(npArr)
    vx_image = vx.CreateImageFromHandle(context, vx.DF_IMAGE_U8, addr, npArr[0], vx.IMPORT_TYPE_HOST)
    print vx.QueryImage(vx_image, vx.IMAGE_ATTRIBUTE_WIDTH, 'vx_uint32') == (vx.SUCCESS, width)
    print vx.QueryImage(vx_image, vx.IMAGE_ATTRIBUTE_FORMAT, 'vx_enum') == (vx.SUCCESS, vx.DF_IMAGE_U8)
    return vx_image


def vx2np(vx_image):
    status, width = vx.QueryImage(vx_image, vx.IMAGE_ATTRIBUTE_WIDTH, 'vx_uint32')
    status, height = vx.QueryImage(vx_image, vx.IMAGE_ATTRIBUTE_HEIGHT, 'vx_uint32')
    rect = vx.rectangle_t(0, 0, width-1, height-1)
    addr = vx.imagepatch_addressing_t(width, height, 1, width, vx.SCALE_UNITY, vx.SCALE_UNITY, 1, 1)
    rdata = array('B', [0]) * (width * height)
    status, addr, ptr = vx.AccessImagePatch(vx_image, rect, 0, addr, rdata, vx.READ_ONLY)
    vx.CommitImagePatch(vx_image,rect, 0, addr, rdata)
    npArray = np.reshape(rdata, (-1,width))
    npArray2 = np.uint8(npArray)
    return npArray2


def pil2vx_rgb(context, pil_image):
    height = pil_image.height
    width = pil_image.width
    addr = vx.imagepatch_addressing_t(width, height, 1, width, vx.SCALE_UNITY, vx.SCALE_UNITY, 1, 3)
    arr = list(pil_image.getdata())
    npArr = np.reshape(arr, (-1,width))
    npArr = np.uint8(npArr)
    vx_image = vx.CreateImageFromHandle(context, vx.DF_IMAGE_RGB, addr, npArr[0], vx.IMPORT_TYPE_HOST)
    print vx.QueryImage(vx_image, vx.IMAGE_ATTRIBUTE_WIDTH, 'vx_uint32') == (vx.SUCCESS, width)
    print vx.QueryImage(vx_image, vx.IMAGE_ATTRIBUTE_FORMAT, 'vx_enum') == (vx.SUCCESS, vx.DF_IMAGE_U8)
    return vx_image
