from pyvx import vx
from pyvx.backend import lib, ffi
import array
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

COLOR = {vx.DF_IMAGE_IYUV:'DF_IMAGE_IYUV', vx.DF_IMAGE_NV12:'DF_IMAGE_NV12', vx.DF_IMAGE_NV21:'DF_IMAGE_NV21', \
         vx.DF_IMAGE_RGB:'DF_IMAGE_RGB', vx.DF_IMAGE_RGBX:'DF_IMAGE_RGBX', vx.DF_IMAGE_S16:'DF_IMAGE_S16', \
         vx.DF_IMAGE_S32:'DF_IMAGE_S32', vx.DF_IMAGE_U16:'DF_IMAGE_U16', vx.DF_IMAGE_U32:'DF_IMAGE_U32', \
         vx.DF_IMAGE_U8:'DF_IMAGE_U8', vx.DF_IMAGE_UYVY:'DF_IMAGE_UYVY', vx.DF_IMAGE_VIRT:'DF_IMAGE_VIRT', \
         vx.DF_IMAGE_YUV4:'DF_IMAGE_YUV4', vx.DF_IMAGE_YUYV:'DF_IMAGE_YUYV'}


class Channel(object):
    VX_CHANNEL_0 = vx.CHANNEL_0
    VX_CHANNEL_1 = vx.CHANNEL_1
    VX_CHANNEL_2 = vx.CHANNEL_2
    VX_CHANNEL_3 = vx.CHANNEL_3
    VX_CHANNEL_R = vx.CHANNEL_R
    VX_CHANNEL_G = vx.CHANNEL_G
    VX_CHANNEL_B = vx.CHANNEL_B
    VX_CHANNEL_A = vx.CHANNEL_A
    VX_CHANNEL_Y = vx.CHANNEL_Y
    VX_CHANNEL_U = vx.CHANNEL_U
    VX_CHANNEL_V = vx.CHANNEL_V


class Color(object):
    VX_DF_IMAGE_VIRT = vx.DF_IMAGE_VIRT
    VX_DF_IMAGE_RGB  = vx.DF_IMAGE_RGB
    VX_DF_IMAGE_RGBX = vx.DF_IMAGE_RGBX
    VX_DF_IMAGE_NV12 = vx.DF_IMAGE_NV12
    VX_DF_IMAGE_NV21 = vx.DF_IMAGE_NV21
    VX_DF_IMAGE_UYVY = vx.DF_IMAGE_UYVY
    VX_DF_IMAGE_YUYV = vx.DF_IMAGE_YUYV
    VX_DF_IMAGE_IYUV = vx.DF_IMAGE_IYUV
    VX_DF_IMAGE_YUV4 = vx.DF_IMAGE_YUV4
    VX_DF_IMAGE_U8   = vx.DF_IMAGE_U8
    VX_DF_IMAGE_U16  = vx.DF_IMAGE_U16
    VX_DF_IMAGE_S16  = vx.DF_IMAGE_S16
    VX_DF_IMAGE_U32  = vx.DF_IMAGE_U32
    VX_DF_IMAGE_S32  = vx.DF_IMAGE_S32


class Data_type(object):
    VX_TYPE_INVALID = vx.TYPE_INVALID
    VX_TYPE_CHAR = vx.TYPE_CHAR
    VX_TYPE_INT8 = vx.TYPE_INT8
    VX_TYPE_UINT8 = vx.TYPE_UINT8
    VX_TYPE_INT16 = vx.TYPE_INT16
    VX_TYPE_UINT16 = vx.TYPE_UINT16
    VX_TYPE_INT32 = vx.TYPE_INT32
    VX_TYPE_UINT32 = vx.TYPE_UINT32
    VX_TYPE_INT64 = vx.TYPE_INT16
    VX_TYPE_UINT64 = vx.TYPE_UINT64
    VX_TYPE_FLOAT32 = vx.TYPE_FLOAT32
    VX_TYPE_FLOAT64 = vx.TYPE_FLOAT64
    VX_TYPE_ENUM = vx.TYPE_ENUM
    VX_TYPE_SIZE = vx.TYPE_SIZE
    VX_TYPE_DF_IMAGE = vx.TYPE_DF_IMAGE
    VX_TYPE_BOOL = vx.TYPE_BOOL
    VX_TYPE_SCALAR_MAX = vx.TYPE_SCALAR_MAX
    VX_TYPE_RECTANGLE = vx.TYPE_RECTANGLE
    VX_TYPE_KEYPOINT = vx.TYPE_KEYPOINT
    VX_TYPE_COORDINATES2D = vx.TYPE_COORDINATES2D
    VX_TYPE_COORDINATES3D = vx.TYPE_COORDINATES3D
    VX_TYPE_STRUCT_MAX = vx.TYPE_STRUCT_MAX
    VX_TYPE_REFERENCE = vx.TYPE_REFERENCE
    VX_TYPE_CONTEXT = vx.TYPE_CONTEXT
    VX_TYPE_GRAPH = vx.TYPE_GRAPH
    VX_TYPE_NODE = vx.TYPE_NODE
    VX_TYPE_KERNEL = vx.TYPE_KERNEL
    VX_TYPE_PARAMETER = vx.TYPE_PARAMETER
    VX_TYPE_DELAY = vx.TYPE_DELAY
    VX_TYPE_LUT = vx.TYPE_LUT
    VX_TYPE_DISTRIBUTION = vx.TYPE_DISTRIBUTION
    VX_TYPE_PYRAMID = vx.TYPE_PYRAMID
    VX_TYPE_THRESHOLD = vx.TYPE_THRESHOLD
    VX_TYPE_MATRIX = vx.TYPE_MATRIX
    VX_TYPE_CONVOLUTION = vx.TYPE_CONVOLUTION
    VX_TYPE_SCALAR = vx.TYPE_SCALAR
    VX_TYPE_ARRAY = vx.TYPE_ARRAY
    VX_TYPE_IMAGE = vx.TYPE_IMAGE
    VX_TYPE_REMAP = vx.TYPE_REMAP
    VX_TYPE_ERROR = vx.TYPE_ERROR
    VX_TYPE_META_FORMAT = vx.TYPE_META_FORMAT
    VX_TYPE_OBJECT_MAX = vx.TYPE_OBJECT_MAX

class Policy(object):
    VX_CONVERT_POLICY_WRAP = vx.CONVERT_POLICY_WRAP
    VX_CONVERT_POLICY_SATURATE = vx.CONVERT_POLICY_SATURATE
    VX_ROUND_POLICY_TO_ZERO = vx.ROUND_POLICY_TO_ZERO
    VX_ROUND_POLICY_TO_NEAREST_EVEN = vx.ROUND_POLICY_TO_NEAREST_EVEN

class Interpolation(object):
    VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR = vx.INTERPOLATION_TYPE_NEAREST_NEIGHBOR
    VX_INTERPOLATION_TYPE_BILINEAR = vx.INTERPOLATION_TYPE_BILINEAR
    VX_INTERPOLATION_TYPE_AREA = vx.INTERPOLATION_TYPE_AREA

class Scale(object):
    SCALE_PYRAMID_HALF = vx.SCALE_PYRAMID_HALF
    SCALE_PYRAMID_ORB = vx.SCALE_PYRAMID_ORB

class Norm(object):
    NORM_L1 = vx.NORM_L1
    NORM_L2 = vx.NORM_L2

class Reference(object):
    def __init__(self):
        self.vx_ref = None
        self.vx_context = None

    def get_context(self):
        return self.vx_context

    def reference_count(self):
        status, val = vx.QueryReference(self.vx_ref, vx.REF_ATTRIBUTE_COUNT, 'vx_uint32')
        return val

    def reference_type(self):
        status, val = vx.QueryReference(self.vx_ref, vx.REF_ATTRIBUTE_TYPE, 'vx_enum')
        return val


class Policy(object):
    VX_CONVERT_POLICY_WRAP = vx.CONVERT_POLICY_WRAP
    VX_CONVERT_POLICY_SATURATE = vx.CONVERT_POLICY_SATURATE

class Distibution(Reference):
    def __init__(self, context, numBins, offset, range):
        self.distribution = vx.CreateDistribution(context, numBins, offset, range)

    def release_distribution(self):
        vx.ReleaseDistribution(self.distribution)

    def query(self):
        pass

    def access(self):
        pass

    def commit(self):
        pass

class Context(Reference):
    def __init__(self):
        self.vx_context = vx.CreateContext()
        self.vx_ref = vx.reference(self.vx_context)

    def release_context(self):
        try:
            vx.ReleaseContext(self.vx_context)
        except:
            pass

    def get_references(self):
        status, references = vx.QueryContext(self.vx_context, vx.CONTEXT_ATTRIBUTE_REFERENCES, 'vx_uint32')
        if status != 0:
            raise NameError('Failed to query VX_CONTEXT_ATTRIBUTE_REFERENCES')
        return references

    def get_unique_kernels(self):
        status, kernels = vx.QueryContext(self.vx_context, vx.CONTEXT_ATTRIBUTE_UNIQUE_KERNELS, 'vx_uint32')
        if status != 0:
            raise NameError('Failed to query VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNELS')
        return kernels

    def get_modules(self):
        status, modules = vx.QueryContext(self.vx_context, vx.CONTEXT_ATTRIBUTE_MODULES, 'vx_uint32')
        if status != 0:
            raise NameError('Failed to query VX_CONTEXT_ATTRIBUTE_MODULES')
        return modules


class Graph(Reference):
    def __init__(self, ctx = None, verify=False):
        self.verify = verify
        if ctx is None:
            self.graph = None
            self.vx_ref = None
            self.context = None
        else:
            self.graph = vx.CreateGraph(ctx.vx_context)
            self.vx_ref = vx.reference(self.graph)
            self.context = ctx
            self.vx_context = ctx.get_context()

    def __enter__(self):
        if self.context is None:
            self.context = Context()
            self.vx_context = self.context.get_context()
        if self.graph is None:
            self.graph = vx.CreateGraph(self.vx_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verify:
            self.vxVerifyGraph()


    def vxReleaseGraph(self):
        vx.ReleaseGraph(self.graph)

    def vxVerifyGraph(self):
        try:
            status = vx.VerifyGraph(self.graph)
            if status != 0:
                raise NameError("Graph verification failed! with {} error code".format(status))
            else:
                logging.debug('Verify pass')
        except Exception as err:
            raise err


    def vxProcessGraph(self):
        try:
            status = vx.ProcessGraph(self.graph)
            if status != 0:
                raise NameError("Graph processing failed! with {} error code".format(status))
        except Exception as err:
            raise err

    def vxScheduleGraph(self):
        return vx.ScheduleGraph(self.graph)

    def vxWaitGraph(self):
        return vx.WaitGraph(self.graph)

    def vxIsGraphVerify(self):
        return vx.IsGraphVerified(self.graph)

    def get_num_node(self):
        status, nodes = vx.QueryGraph(self.graph, vx.GRAPH_ATTRIBUTE_NUMNODES, 'vx_uint32')
        if status != 0:
            raise NameError('Failed to query VX_GRAPH_ATTRIBUTE_NUMNODES')
        return nodes

    def get_status(self):
        status, graph_status = vx.QueryGraph(self.graph, vx.GRAPH_ATTRIBUTE_STATUS, 'vx_uint32')
        if status != 0:
            raise NameError('Failed to query VX_GRAPH_ATTRIBUTE_NUMNODES')
        return graph_status


class Node(Reference):
    def __init__(self):
        self.node = None
        self.vx_ref = None
        self.context = None

    def vxReleaseNode(self):
        vx.ReleaseNode(self.node)


class Image(Reference):
    def __init__(self, root, width=0, height=0, color=vx.DF_IMAGE_VIRT, value=None, vx_img=None):
        if vx_img is not None:
            self.setup_vx_img(vx_img)
        else:
            self.width = width
            self.height = height
            self.color = color
            try:
                if isinstance(root, Context):   # For real image
                    if value is None:
                        self.image = vx.CreateImage(root.vx_context, width, height, color)
                    else:
                        self.image = vx.CreateUniformImage(root.vx_context, width, height, color, value, "int")
                    self.virtual_image = False
                elif isinstance(root, Graph):   # For virtual image
                    self.image = vx.CreateVirtualImage(root.graph, width, height, color)
                    self.virtual_image = True
            except Exception as exception:
                raise exception

    def setup_vx_img(self, img):
        self.image = img
        self.width = self.get_width()
        self.height = self.get_height()
        self.color = self.get_color()

    def vxReleaseImage(self):
        vx.ReleaseImage(self.image)

    def get_width(self):
        s = vx.GetStatus(vx.reference(self.image))
        status, width = vx.QueryImage(self.image, vx.IMAGE_ATTRIBUTE_WIDTH, 'vx_uint32')
        if status != 0:
            raise NameError('Failed to query VX_IMAGE_ATTRIBUTE_WIDTH')
        return width

    def get_color(self):
        status, color = vx.QueryImage(self.image, vx.IMAGE_ATTRIBUTE_FORMAT, 'vx_df_image')
        if status != 0:
            raise NameError('Failed to query VX_IMAGE_ATTRIBUTE_FORMAT')
        return COLOR[color]

    def get_height(self):
        status, height = vx.QueryImage(self.image, vx.IMAGE_ATTRIBUTE_HEIGHT, 'vx_uint32')
        if status != 0:
            raise NameError('Failed to query VX_IMAGE_ATTRIBUTE_HEIGHT')
        return height


class Threshold(Reference):
    THRESHOLD_TYPE_BINARY = vx.THRESHOLD_TYPE_BINARY
    THRESHOLD_TYPE_RANGE = vx.THRESHOLD_TYPE_RANGE

    def __init__(self, context, thresh_type, data_type=vx.TYPE_UINT8):
        self.threshold = vx.CreateThreshold(context.vx_context, thresh_type, data_type)
        # stat = vx.GetStatus(vx.reference(self.threshold))
        self.thresh_type = thresh_type
        self.lower = None
        self.upper = None
        self.value = None

    def set_lower(self, val):
        vx.SetThresholdAttribute(self.threshold, vx.THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, val, 'vx_int32')
        self.lower = val

    def set_upper(self, val):
        vx.SetThresholdAttribute(self.threshold, vx.THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, val, 'vx_int32')
        self.upper = val

    def set_value(self, val):
        vx.SetThresholdAttribute(self.threshold, vx.THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, val, 'vx_int32')
        self.value = val


class Scalar(Reference):

    def __init__(self, context, data_type, value):
        self._scalar = vx.CreateScalar(context.vx_context, data_type, value)
        self._value = value
        self._type = data_type

    def get_value(self):
        return self._value


class Lut(Reference):
    def __init__(self, context, data_type, count):
        self._lut = vx.CreateLUT(context.vx_context, data_type, count)
        self.data_typr = data_type
        self.count = count
        self.values = []

    def get_lut(self):
        return self.values

    def set_values(self, values):
        self.values = values
        status, ptr = vx.AccessLUT(self._lut, None, vx.READ_AND_WRITE)
        if status:
            raise NameError('Failed to accessLUT')
        for v, i in zip(values, range(len(ptr))):
            ptr[i] = v
        vx.CommitLUT(self._lut, ptr)


class Convolution(Reference):
    def __init__(self, context, columns, rows):
        self._convolution = vx.CreateConvolution(context.vx_context, columns=columns, rows=rows)
        self.array = None
    def get_convolution(self):
        return vx.ReadConvolutionCoefficients(self._convolution, self.array)

    def set_convolution(self, array):
        """
        needs to be fix!
        :param array:
        :return:
        """
        return vx.WriteConvolutionCoefficients(self._convolution, array)


class Matrix(Reference):
    def __init__(self, data_type, context, columns, rows):
        self.columns= columns
        self.rows = rows
        self.data_type = data_type
        self.data = None
        self._matrix = vx.CreateMatrix(context.vx_context, data_type, columns, rows)

    def set_array(self):
        if self.data_type is Data_type.VX_TYPE_FLOAT32:
            self.data = array.array('f')
        else:
            self.data = array.array('l')
        for i in range(0, self.columns*self.rows):
            self.data.append(-1)

    def get_matrix(self):
        vx.ReadMatrix(self._matrix, self.data)
        return self.data

    def set_matrix(self, matrix_values):
        for val, i in zip(matrix_values, range(0, len(matrix_values))):
            self.data[i] = val
        vx.WriteMatrix(self._matrix, self.data)


class Pyramid(Reference):
    def __init__(self, context_graph, levels, scale, width, height, image_format, virtual=False):
        self.levels = levels
        self.scale = scale
        self.width = width
        self.height = height
        self.format = image_format
        if virtual:
            self._pyramid = vx.CreateVirtualPyramid(context_graph.vx_context, levels, scale, width, height, image_format)
        else:
            self._pyramid = vx.CreatePyramid(context_graph.graph, levels, scale, width, height, image_format)


class Array(Reference):
    def __init__(self, context, item_type, capacity):
        self.item_type = item_type
        self.capacity = capacity
        self._array = vx.CreateArray(context.vx_context, item_type, capacity)

    def get_array(self):
        pass

    def add_item(self, count, elements, stride=0):
        items = array.array('l')
        for i in elements:
            items.append(i)
        print vx.AddArrayItems(self._array, count, items, stride)


class Remap(Reference):
    def __init__(self, context, src_width, src_height, dst_width, dst_height):
        self.src_width = src_width
        self.src_height =src_height
        self.dst_width = dst_width
        self.dst_height = dst_height
        self._remap = vx.CreateRemap(context.vx_context, src_width, src_height, dst_width, dst_height)


def Sobel3x3Node(graph, input_img, output_x=None, output_y=None):
        if output_x is None:
            output_x = Image(graph, input_img.get_width(), input_img.get_height())
        if output_y is None:
            output_y = Image(graph, input_img.get_width(), input_img.get_height())
        vx.Sobel3x3Node(graph.graph, input_img.image, output_x.image, output_y.image)
        return output_x, output_y


def MagnitudeNode(graph, grad_x, grad_y, mag=None):
        if mag is None:
            mag = Image(graph, grad_x.get_width(), grad_x.get_height())
        vx.MagnitudeNode(graph.graph, grad_x.image, grad_y.image, mag.image)
        return mag


def PhaseNode(graph, grad_x, grad_y, orientation=None):
        if orientation is None:
            orientation = Image(graph, grad_x.get_width(), grad_x.get_height())
        vx.PhaseNode(graph.graph, grad_x.image, grad_y.image, orientation.image)
        return orientation


def NotNode(graph, src_img, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height())
    vx.NotNode(graph.graph, src_img.image, output.image)
    return output


def ChannelExtractNode(graph, input_image, channel, output_image=None):
    if output_image is None:
            output_image = Image(graph, input_image.get_width(), input_image.get_height(), color=Color.VX_DF_IMAGE_U8)
    vx.ChannelExtractNode(graph.graph, input_image.image, channel, output_image.image)
    return output_image


def Box3x3Node(graph, src_img, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.Box3x3Node(graph.graph, src_img.image, output.image)
    return output


def ColorConvertNode(graph, src_img, output=None, color=None):
    if output is None:
        if color is None:
            raise NameError("Output Image or color has to be specified")
        output = Image(graph, src_img.get_width(), src_img.get_height(), color)
    s = vx.GetStatus(vx.reference(output.image))
    vx.ColorConvertNode(graph.graph, src_img.image, output.image)
    return output


def HistogramNode(graph, src_img, dis):
    vx.HistogramNode(graph.graph, src_img.image, dis)
    return dis


def Gaussian3x3Node(graph, src_img, output=None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.Gaussian3x3Node(graph.graph, src_img.image, output.image)
    return output


def AbsDiffNode(graph, src1, src2, output=None):
    if output is None:
        output = Image(graph, src1.get_width(), src2.get_height(), Color.VX_DF_IMAGE_U8)
    vx.AbsDiffNode(graph.graph, src1.image, src2.image, output.image)
    return output


def AccumulateImageNode(graph, src_img, accum=None):
    if accum is None:
        accum = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_S16)
    vx.AccumulateImageNode(graph.graph, src_img.image, accum.image)
    return accum


def AndNode(graph, src1, src2, output=None):
    if output is None:
        output = Image(graph, src1.get_width(), src1.get_height(), Color.VX_DF_IMAGE_U8)
    vx.AndNode(graph.graph, src1.image, src2.image, output.image)
    return output


def XorNode(graph, src1, src2, output=None):
    if output is None:
        output = Image(graph, src1.get_width(), src2.get_height(), Color.VX_DF_IMAGE_U8)
    vx.XorNode(graph.graph, src1.image, src2.image, output.image)
    return output


def OrNode(graph, src1, src2, output=None):
    if output is None:
        output = Image(graph, src1.get_width(), src2.get_height(), Color.VX_DF_IMAGE_U8)
    vx.OrNode(graph.graph, src1.image, src2.image, output.image)
    return output


def ChannelCombineNode(graph, plane1, plane2, plane3=None, plane4=None, output=None, color=None):
    if output is None:
        if color is None:
            raise NameError("Output Image or color has to be specified")
        output = Image(graph, plane1.get_width(), plane1.get_height(), color)

    if isinstance(plane3, Image):
        i3 = plane3.image
    else:
        i3 = ffi.NULL
    if isinstance(plane4, Image):
        i4 = plane4.image
    else:
        i4 = ffi.NULL
    vx.ChannelCombineNode(graph.graph, plane1.image, plane2.image, i3, i4, output.image)
    return output


def EqualizeHistNode(graph, src_img, output=None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.EqualizeHistNode(graph.graph, src_img.image, output.image)
    return output


def ThresholdNode(graph, src_img, threshold, output):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    n = vx.ThresholdNode(graph.graph, src_img.image, threshold.threshold, output.image)
    s = vx.GetStatus(vx.reference(n))
    return output


def ConvertDepthNode(graph, src_img, policy, output=None, color=None, scalar=None):
    if output is None:
        if color is None:
            raise NameError("Output Image or color has to be specified")
        output = Image(graph, src_img.get_width(), src_img.get_height(), color)
    if scalar is None:
        scalar = Scalar(graph.vx_context, Data_type.VX_TYPE_INT32, 0)
    vx.ConvertDepthNode(graph.graph, src_img.image, output.image, policy, scalar._scalar)
    return output


def AccumulateSquareImageNode(graph, src_img, alpha, accum=None):
    if accum is None:
        accum = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_S16)
    vx.AccumulateSquareImageNode(graph.graph, src_img.image, alpha._scalar, accum.image)
    return accum


def AccumulateWeightedImageNode(graph, src_img, alpha, accum=None):
    if accum is None:
        accum = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.AccumulateWeightedImageNode(graph.graph, src_img.image, alpha._scalar, accum.image)
    return accum


def AddNode(graph, src_img, src_img2, policy, output):
    if output is None:
        if src_img.get_color() is 'DF_IMAGE_U8' and src_img2.get_color() is 'DF_IMAGE_U8':
            output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
        else:
            output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_S16)
    vx.AddNode(graph.graph, src_img.image, src_img2.image, policy, output.image)
    return output


def SubtractNode(graph, src_img, src_img2, policy, output):
    if output is None:
        if src_img.get_color() is 'DF_IMAGE_U8' and src_img2.get_color() is 'DF_IMAGE_U8':
            output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
        else:
            output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_S16)
    vx.SubtractNode(graph.graph, src_img.image, src_img2.image, policy, output.image)
    return output


def Dilate3x3Node(graph, src_img, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.Dilate3x3Node(graph.graph, src_img.image, output.image)
    return output


def Erode3x3Node(graph, src_img, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.Erode3x3Node(graph.graph, src_img.image, output.image)
    return output


def Median3x3Node(graph, src_img, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.Median3x3Node(graph.graph, src_img.image, output.image)
    return output


def MeanStdDevNode(graph, src_img, mean, stddev):
    if mean is None:
        mean = Scalar(graph.vx_context, Data_type.VX_TYPE_FLOAT32, 0)
    if stddev is None:
        stddev = Scalar(graph.vx_context, Data_type.VX_TYPE_FLOAT32, 0)
    vx.MeanStdDevNode(graph.graph, src_img.image, mean._scalar, stddev._scalar)
    return mean, stddev


def TableLookupNode(graph, src_img, lut, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.TableLookupNode(graph.graph, src_img.image, lut._lut, output.image)
    return output


def ConvolveNode(graph, src_img, conv, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_S16)
    vx.ConvolveNode(graph.graph, src_img.image, conv._convolution, output.image)
    return output


def HalfScaleGaussianNode(graph, src_img, output = None, kernel_size=3):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.HalfScaleGaussianNode(graph.graph, src_img.image, output.image, kernel_size)
    return output


def ScaleImageNode(graph, src_img, output = None, interpolation=Interpolation.VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height())
    vx.ScaleImageNode(graph.graph, src_img.image, output.image, interpolation)
    return output


def WarpAffineNode(graph, src_img, matrix, interpolation, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.WarpAffineNode(graph.graph, src_img.image, matrix._matrix, interpolation, output.image)
    return output


def WarpPerspectiveNode(graph, src_img, matrix, interpolation, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.WarpPerspectiveNode(graph.graph, src_img.image, matrix._matrix, interpolation, output.image)
    return output


def MultiplyNode(graph, src_img1, src_img2, scale, overflow_policy, rounding_policy, output = None):
    if output is None:
        if src_img1.get_color() is 'DF_IMAGE_U8' and src_img2.get_color() is 'DF_IMAGE_U8':
            output = Image(graph, src_img1.get_width(), src_img1.get_height(), Color.VX_DF_IMAGE_U8)
        else:
            output = Image(graph, src_img1.get_width(), src_img1.get_height(), Color.VX_DF_IMAGE_S16)
    vx.MultiplyNode(graph.graph, src_img1.image, scale._scalar, overflow_policy, rounding_policy, output.image)
    return output


def GaussianPyramidNode(graph, src_img, gaussian):
    vx.GaussianPyramidNode(graph.graph, src_img.image, gaussian)
    return gaussian


def IntegralImageNode(graph, src, output=None):
    if output is None:
        output = Image(graph, src.get_width(), src.get_height(), Color.VX_DF_IMAGE_U32)
    vx.IntegralImageNode(graph.graph, src.image, output.image)
    return output


def CannyEdgeDetectorNode(graph, src_img, hyst, gradient_size, norm_type, output = None):
    if output is None:
        output = Image(graph, src_img.get_width(), src_img.get_height(), Color.VX_DF_IMAGE_U8)
    vx.CannyEdgeDetectorNode(graph.graph, src_img.image, hyst.threshold, gradient_size, norm_type, output.image)
    return output


def FastCornersNode(graph, src_img, threshold, nonmax_suppression, num_of_corners_to_detect, corners=None, num_corners=None):
    if corners is None:
        corners = Array(graph.vx_context, Data_type.VX_TYPE_KEYPOINT, num_of_corners_to_detect)
    if num_corners is None:
        num_corners = Scalar(graph.vx_context, Data_type.VX_TYPE_SIZE)
    vx.FastCornersNode(graph.graph, src_img.image, threshold._scalar, nonmax_suppression, corners._array, num_corners._scalar)
    return corners, num_corners


def HarrisCornersNode(graph, src_img, threshold, min_distance, sensitivity, gradient_size, block_size, num_of_corners_to_detect, corners=None, num_corners=None):
    if corners is None:
        corners = Array(graph.vx_context, Data_type.VX_TYPE_KEYPOINT, num_of_corners_to_detect)
    if num_corners is None:
        num_corners = Scalar(graph.vx_context, Data_type.VX_TYPE_SIZE)
    vx.HarrisCornersNode(graph.graph, src_img.image, threshold._scalar, min_distance._scalar, sensitivity._scalar, gradient_size, block_size, corners._array, num_corners._scalar)
    return corners, num_corners


def MinMaxLocNode(graph, src_img, min_val=None, max_val=None, min_loc=None, max_loc=None, min_count=None, max_count=None):
    if min_val is None:
        min_val = Scalar(graph.vx_context, Data_type.VX_TYPE_INT16)
    if max_val is None:
        max_val = Scalar(graph.vx_context, Data_type.VX_TYPE_INT16)
    if min_loc is None:
        min_loc = Array(graph.vx_context, Data_type.VX_TYPE_COORDINATES2D, 100)
    if max_loc is None:
        max_loc = Array(graph.vx_context, Data_type.VX_TYPE_COORDINATES2D, 100)
    if min_count is None:
        min_count = Scalar(graph.vx_context, Data_type.VX_TYPE_UINT32)
    if max_count is None:
        max_count = Scalar(graph.vx_context, Data_type.VX_TYPE_UINT32)
    vx.MinMaxLocNode(graph.graph, src_img.image, min_val._scalar, max_val._scalar, min_loc._array, max_loc._array, min_count._scalar, max_count._scalar)


def OpticalFlowPyrLKNode(graph, old_images, new_images, old_points, new_points_estimates, new_points, termination,epsilon, num_iterations, use_initial_estimate, window_dimension):
    vx.OpticalFlowPyrLKNode(graph.graph, old_images._pyramid, new_images._pyramid, old_points._array, new_points_estimates._array, new_points._array, termination,
epsilon._scalar, num_iterations._scalar, use_initial_estimate._scalar, window_dimension)


def RemapNode(graph, src1, src2, table, policy, output=None):
    if output is None:
        output = Image(graph, src1.get_width(), src1.get_height(), Color.VX_DF_IMAGE_U8)
    vx.RemapNode(graph.graph, src1.image, src2.image,table._remap, policy, output.image)
    return output



