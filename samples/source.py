from pyvx import vx
from PIL import Image
from GoProClass import GoProCamera
from vx_utils import *


def main():
 #   go_pro = GoProCamera()
#    go_pro.shoot_still()
    name='3'
    pil_img = Image.open('3.JPG')#go_pro.get_image()
    pil_img = pil_img.convert('L')
    pil_img = pil_img.resize((600,800))
    # OpenVX
    context = vx.CreateContext()
    graph = vx.CreateGraph(context)
    img_w = pil_img.width
    img_h = pil_img.height
    vxImage = pil2vx(context, pil_img)
    grey = vx.CreateImage(context, img_w, img_h, vx.DF_IMAGE_U8)
    images = [
        vx.CreateImage(context, img_w, img_h, vx.DF_IMAGE_S16),
        vx.CreateImage(context, img_w, img_h, vx.DF_IMAGE_S16),
        vx.CreateImage(context, img_w, img_h, vx.DF_IMAGE_S16),
        vx.CreateImage(context, img_w, img_h, vx.DF_IMAGE_U8),
        vx.CreateImage(context, img_w, img_h, vx.DF_IMAGE_U8),
        vx.CreateImage(context, img_w, img_h, vx.DF_IMAGE_U8)
    ]
    vx.Sobel3x3Node(graph, vxImage, images[1], images[2])
    vx.MagnitudeNode(graph, images[1], images[2], images[0])
    vx.PhaseNode(graph,images[1], images[2], images[3])
    threshold = vx.CreateThreshold(context, vx.THRESHOLD_TYPE_RANGE, vx.TYPE_UINT8)
    #vx.SetThresholdAttribute(threshold,vx.THRESHOLD_ATTRIBUTE_THRESHOLD_up, 10, "vx_uint32")
    vx.SetThresholdAttribute(threshold,vx.THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, 10, "vx_uint32")
    vx.SetThresholdAttribute(threshold,vx.THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, 100, "vx_uint32")
    print vx.QueryThreshold(threshold, vx.THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER,"vx_uint32")
    print vx.QueryThreshold(threshold, vx.THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER,"vx_uint32")
    vx.CannyEdgeDetectorNode(graph, vxImage, threshold, 3, vx.NORM_L1, images[4])

    status = vx.VerifyGraph(graph)
    if status == vx.SUCCESS:
        status = vx.ProcessGraph(graph)
    print status

    mag = vx2np(images[0])
    phase = vx2np(images[3])
    canny = vx2np(images[4])
    pil_img.save(name+'_input.jpg')
    Image.fromarray(mag).save(name+'_magnitude.jpg')
    Image.fromarray(canny).save(name+'_canny.jpg')
    pil_img.show('Source')
    if True:  # Show image using PIL
        pil_img.show()
        Image.fromarray(mag).show('Magnitued')
        Image.fromarray(phase).show('Phase')
        Image.fromarray(canny).show('Canny edge detector')


main()




