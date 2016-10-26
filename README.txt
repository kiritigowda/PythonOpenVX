OpenVX based PyOpenVX library:

There are two ways in which PyOpenVX library can be deployed. The first one is by installing a virtual machine, which comprised of compile code and samples. The second one is by manually compile the necessary files.

********************************************************
*               Using a virtual machine                *
********************************************************

We developed a virtual machine with all of the needed packages already installed and compiled. The virtual machine can be download from: 
https://www.dropbox.com/s/0ob5hpmszkkqqda/Ubuntu_PyVX.ova?dl=0

The virtual machine is a VirtualBox virtual machine, in order to run it you need to download VirtualBox from https://www.virtualbox.org/wiki/Downloads, and import our virtual machine into it. If a user name/password is required, you can use: 
User name: ori
Password: 123

Tutorial:
a.	Open VirtualBox
b.	File->Import Application
c.	Choose the virtual machine image ('.ovx' file)
d.	Click 'Next' and then click 'Import'

Run the samples from the VM:
a.	Open shell
b.	Go to /home/ori/Desktop/samples
c.	Run one of the samples:
a.	./sobel_graph.py
b.	./EqualizationGraph.py

The sample code is in the samples folder.

********************************************************
*               Building on a new machine              *
********************************************************

This PyVX extension tested on Ubuntu 16.04.

Prerequisites:
1)	Cmake 2.1 or above (can be download using `sudo apt-get install cmake`)
2)	Download Python 2.7x https://www.python.org/ftp/python/2.7.12/
3)	Download the OpenVX implementation- such as OpenVX sample implementation from: https://www.khronos.org/registry/vx/sample/openvx_sample_1.0.1.tar.bz2 
	a.	Build the OpenVX implantation:
		i.	Unzip the openvx_sample 
		ii.	Run from shell: python Build.py --os=Linux --arch=64
4)	Python libraries: numpy, cffi, PyVX
	a.	Download packages - run from shell: 
		i.	pip install cffi
		ii.	pip install numpy
	b.	Install the PyVX module http://pyvx.readthedocs.io/en/rewrite/ 
		i.	git clone https://github.com/hakanardo/pyvx.git
		ii.	cd pyvx
		iii. git checkout rewrite
		iv.	sudo python setup.py install
		v.	sudo python -mpyvx.build_cbackend --default pyvx /path/to/openvx/install/
5)	Download the Pythonic library from github: https://github.com/NBEL-lab/PythonOpenVX/
6)	Import the Pythonic library and write your App!

Recommended IDE:
We used PyCharm by JetBrains and it was great, very supportive and convenient IDE. 
Link for download: https://www.jetbrains.com/pycharm/

We suggest before trying to run sophisticate graph to start from a single node graph and extend it node by node.

Running:
Import to your python source code the Pythonic library.
Create your OpenVX graph in one of the two ways:
	1)	Create context and graph as regular instance and add the images and the nodes.
	2)	Create a graph as part of with statement.
	
Code Example:

	from pythonic import *

	def sobel_graph(input):
		width = 100
		height = 100
		
		# The OpenVX Context and Graph are created as part of the ‘with’ statement 
		with Graph(verify=True) as g:
			
			# Creating an OpenVX image	
			out = Image(g.context, width, height, Color.VX_DF_IMAGE_RGB)		
			y = ChannelExtractNode(g, input, Channel.VX_CHANNEL_Y)
			
			#Creating and OpenVX Sobel node in the graph
			gx, gy = Sobel3x3Node(g, y)	
			
			#Creating and OpenVX Magnitude node in the graph
			mag = MagnitudeNode(g, gx, gy)	

			#Creating an OpenVX scalar object
			shift = Scalar(g.context, Data_type.VX_TYPE_INT32, 0)

			#Creating an OpenVX conversion node 	
			converted = ConvertDepthNode(g, mag, Policy.VX_CONVERT_POLICY_WRAP, color=Color.VX_DF_IMAGE_U8, scalar=shift)	
			
			# Creating an OpenVX channel combine node
			ChannelCombineNode(g, converted, converted, converted, output=out)	
		
		#Process the OpenVX graph we created
		g.vxProcessGraph()	

	return out
