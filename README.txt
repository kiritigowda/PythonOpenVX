OpenVX based Python library:

Prerequisites:
	1) Download Python 2.7x https://www.python.org/ftp/python/2.7.12/
	2) Compiled OpenVX implementation- such as OpenVX sample implementation from: https://www.khronos.org/registry/vx/sample/openvx_sample_1.0.1.tar.bz2.
	3) Download and install PyVX package- https://pypi.python.org/pypi/PyVX .
	

Recommended IDE:
	I used PyCharm by JetBrains and it was great, very supportive and convenient IDE.  https://www.jetbrains.com/pycharm/

I suggest before trying to run sophisticate graph to start from a single node graph and extend it node by node.

Running:
Import to your python source code the Pythonic library.
Create your OpenVX graph in one of the two ways:
	1)	Create context and graph as regular instance and add the images and the nodes.
	2)	Create a graph as part of with statement.
	
Example:

	from pythonic import *

	def sobel_graph(input):
		width = 100
		height = 100
		with Graph(verify=True) as g:
			out = Image(g.context, width, height, Color.VX_DF_IMAGE_RGB)
			y = ChannelExtractNode(g, input, Channel.VX_CHANNEL_Y)
			gx, gy = Sobel3x3Node(g, y)
			mag = MagnitudeNode(g, gx, gy)
			shift = Scalar(g.context, Data_type.VX_TYPE_INT32, 0)
			converted = ConvertDepthNode(g, mag, Policy.VX_CONVERT_POLICY_WRAP, color=Color.VX_DF_IMAGE_U8, scalar=shift)
			ChannelCombineNode(g, converted, converted, converted, output=out)
		g.vxProcessGraph()

return out