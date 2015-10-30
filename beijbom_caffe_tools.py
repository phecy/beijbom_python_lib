import glob, os, math, colorsys, scipy, caffe, re, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import beijbom_misc_tools as bmt
import beijbom_confmatrix as confmatrix
from pylab import *
from copy import deepcopy, copy
import cPickle as pickle
from tqdm import tqdm

"""
beijbom_caffe_tools (bct) contains classes and wrappers for caffe.
"""


class Transformer:
	"""
	Transformer is a class for preprocessing and deprocessing images according to the vgg16 pre-processing paradigm (scaling and mean subtraction.)
	"""

	def __init__(self):
		self.mean = []
		self.scale = []

	def set_mean(self, mean):
		"""
		Set the mean to subtract for centering the data.
		"""
		self.mean = mean

	def set_scale(self, scale):
		"""
		Set the data scaling.
		"""
		self.scale = scale

	def preprocess(self, im):
		"""
		preprocess() emulate the pre-processing occuring in the vgg16 caffe prototxt.
		"""
	
		im = np.float32(im)    
		for channel, channel_mean in list(enumerate(self.mean)):
			im[:, :, channel] = im[:, :, channel] - np.ones(im.shape[:2], dtype=np.uint8) * channel_mean
	
		im = im * self.scale
		im = im.transpose(2, 0, 1)
		
		return im

	def deprocess(self, im):
		"""
		inverse of preprocess()
		"""
		im = im.transpose(1, 2, 0)
		im /= self.scale

		for channel, channel_mean in list(enumerate(self.mean)):
			im[:, :, channel] = im[:, :, channel] + np.ones(im.shape[:2], dtype=np.uint8) * channel_mean
		
		return np.uint8(im)



class CaffeSolver:
	"""
	Caffesolver is a class for creating a solver.prototxt file. It sets default values and can export a solver parameter file.
	Note that all parameters are stored as strings. For technical reasons, the strings are stored as strings within strings.
	"""

	def __init__(self, net_prototxt_path = "net.prototxt", debug = False):
		
		self.sp = {}

		# critical:
		self.sp['base_lr'] = '1e-10'
		self.sp['momentum'] = '0.9'
		
		# speed:
		self.sp['test_iter'] = '100'
		self.sp['test_interval'] = '250'
		
		# looks:
		self.sp['display'] = '25'
		self.sp['snapshot'] = '2500'
		self.sp['snapshot_prefix'] = '"snapshot"' # string withing a string!
		
		# learning rate policy
		self.sp['lr_policy'] = '"step"'
		self.sp['stepsize'] = '100000'

		# important, but rare:
		self.sp['gamma'] = '0.1'
		self.sp['weight_decay'] = '0.0005'
		self.sp['net'] = '"' + net_prototxt_path + '"'

		# pretty much never change these.
		self.sp['max_iter'] = '100000'
		self.sp['test_initialization'] = 'false'
		self.sp['average_loss'] = '1' # this has to do with the display.
		self.sp['iter_size'] = '1' #this is for accumulating gradients

		if (debug):
			self.sp['max_iter'] = '12'
			self.sp['test_iter'] = '1'
			self.sp['test_interval'] = '4'
			self.sp['display'] = '1'

	def add_from_file(self, filepath):
		"""
		Reads a caffe solver prototxt file and updates the Caffesolver instance parameters.
		"""
		with open(filepath, 'r') as f:
			for line in f:
				splitLine = line.split(':')
				self.sp[splitLine[0].strip()] = splitLine[1].strip()
		return 1

	def write(self, filepath):
		"""
		Export solver parameters to INPUT "filepath". Sorted alphabetically.
		"""
		f = open(filepath, 'w')
		for key, value in sorted(self.sp.items()):
			if not(type(value) is str):
				raise TypeError('All solver parameters must be strings')
			f.write('%s: %s\n' % (key, value))
		return 1





def run(workdir = None, caffemodel = 'weights.caffemodel', GPU_id = 0,solver = 'solver.prototxt', log = 'train.log', snapshot_prefix = 'snapshot', caffepath = '/home/beijbom/cc/build/tools/caffe', restart = False):
	"""
	run is a simple caffe wrapper for training nets. It basically does two things. (1) ensures that training continues from the most recent model, and (2) makes sure the output is captured in a log file.

	Takes
	workdir: directory where the net prototxt lives.
	caffemodel: name of a stored caffemodel.
	solver: name of solver.prototxt [this refers, in turn, to net.prototxt]
	log: name of log file
	snapshot_prefix: snapshot prefix. 
	caffepath: path the caffe binaries. This is required since we make a system call to caffe.
	restart: determines whether to restart even if there are snapshots in the directory.

	"""

	# finds the latest snapshots
	snapshots = glob.glob("/{}/{}*.solverstate".format(workdir, snapshot_prefix))
	if snapshots:
		_iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in snapshots]
		latest_snapshot = snapshots[np.argmax(_iter)]
	
	# by default, start from the most recent snapshot
	if snapshots and not(restart): 
		print("Running from iter {}.".format(np.max(_iter)))
		runstring  = 'cd {}; {} train -solver {} -snapshot {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solver, latest_snapshot, GPU_id, log)

	# else, start from a pre-trained net defined in caffemodel
	elif(caffemodel): 
		if(os.path.isfile(os.path.join(workdir, caffemodel))):
			print("Fine tuning from {}.".format(os.path.join(workdir, caffemodel)))
			runstring  = 'cd {}; {} train -solver {} -weights {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solver, caffemodel, GPU_id, log)
		else:
			raise IOError("Can't fine intial weight file: " + os.path.join(workdir, caffemodel))

	# Train from scratch. Not recommended for larger nets.
	else: 
		print("No caffemodel specified. Running from scratch!!")
		runstring  = 'cd {}; {} train -solver {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solver, GPU_id, log)
	os.system(runstring)

def classify(workdir, scorelayer, caffemodel = None, GPU_id = 0, snapshot_prefix = 'snapshot', net_prototxt = 'net.prototxt', save = False, ignore_label = np.inf, n_testinstances = None, batch_size = None):
	"""
	classify runs a trained net on a testset defined in a net.prototxt file and returns the ground truth, estimated labels and the score vectors.

	Takes
	workdir: directory where net_prototxt lives. All paths must be given relative to this directory.
	scorelayer: name of layer to extract the scores from
	caffemodel: name of the stored caffemodel. If not given, the most recent snapshot in workdir will be used.
	snapshot_prefix: snapshot prefix. Only used if caffemodel = None.
	net_prorotxt: name of the net prototxt to use. 
	save: wheather to save the output to disk.
	ignore_label: Ignores all labels where the gt = ignore_label. Relevant only for FCN models. 
	n_testinstances: Number of instances in the test list. If not given, this will be extracted automatically from the testlist or LMDB. 

	Gives
	(gt, est, scores): tuple with ground truth (as list), estimated labels (as list), scores as list of np arrays

	"""

	os.chdir(workdir) #move to workdir
	
	# find and load latest model
	if caffemodel is None: #
		caffemodels = glob.glob("{}*.caffemodel".format(snapshot_prefix))
		if caffemodels:
			_iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in caffemodels]
			caffemodel = caffemodels[np.argmax(_iter)]
		else:
			raise IOError("Can't find a trained model in " + workdir + " using prefix: " + snapshot_prefix + ".")

	# find batch size from prototxt
	if batch_size is None:
		with open (net_prototxt, "r") as myfile:
			net_definition_str = myfile.read()
		batch_size = int(re.findall('(?<=batch_size: )[0-9]*', net_definition_str)[-1]) #the batch size for the test set is assumed to be defined last. ======= TODO =======: make this more robust!

	# find the number of instances in test set:
	test_file = os.path.join('./../', re.findall("(?<=source: ../../)[a-z0-9]*.[a-z]*", net_definition_str)[-1])
	if n_testinstances is None:
		if test_file.find('lmdb') > -1:
			in_db = lmdb.open(test_file)
			n_testinstances = int(in_db.stat()['entries'])
		elif test_file.find('txt') > -1: 
			n_testinstances = nbr_lines(test_file)
		else:
			raise NotImplementedError("Only supports image_data_layers defined in XXXtxt files and LMDB inputs defined in XXXlmdb.")

	print("Classifying " + test_file + " from "+ os.path.join(workdir, net_prototxt) + " using " + caffemodel + " with bs:" + str(batch_size) + ", and " + str(n_testinstances) + " total instances.")
	sys.stdout.flush()

	# Load model
	net = load_model(workdir, caffemodel, GPU_id = GPU_id, net_prototxt = net_prototxt)

	# Classify. All the reshaping has to do with being able to handling both FCN and classification nets.
	gtlist = []
	scorelist = []
	for test_itt in tqdm(range(n_testinstances//batch_size + 1)):
		if net.blobs['label'].data.ndim == 1:
			gtlist.extend(list(copy(net.blobs['label'].data).astype(np.uint8)))
			scorelist.extend(list(copy(net.blobs[scorelayer].data).astype(np.float)))
		else:
			gt = copy(net.blobs['label'].data.transpose(0, 2, 3, 1)).astype(np.uint8)
			scores = copy(net.blobs[scorelayer].data.transpose(0, 2, 3, 1)).astype(np.float)
			nclasses = scores.shape[3]
			gt = np.repeat(gt, nclasses, axis = 3)
			keepind = gt != ignore_label
			scores = scores[keepind]
			scorelist.extend(list(np.reshape(scores, [scores.shape[0]/nclasses, nclasses])))
			gt = gt[keepind]
			gtlist.extend(list(np.reshape(gt, [gt.shape[0]/nclasses, nclasses])[:, 0]))
		net.forward()

	# If the net is not a FCN we need to cut of the lists (since the last iteration may be looping around)
	if net.blobs['label'].data.ndim == 1: 
		gtlist = gtlist[:n_testinstances]
		scorelist = scorelist[:n_testinstances]

	# For convenience, include estimated labels
	estlist = [np.argmax(s) for s in scorelist]
	if (save):
		pickle.dump((gtlist, estlist, scorelist), open(os.path.join(workdir, 'predictions_on_' + test_file[5:] + '_using_' + caffemodel +  '.p'), 'wb'))

	return (gtlist, estlist, scorelist)



def cycle_runs(run_params, test_params, cycle_sizes, ncycles):
	"""
	cycle_runs is a wrapper around run and classify methods. It cycles through the various experiments, thus running them in "parrallell". After training net i for cycle_sizes[i] iterations, it will run through the TEST set of all *net.prototxt files in the directory and store these to disk. It will then move on to the next experiment, and cycle though all for ncycles.

	Takes
	run_params: is a list of dictionaries. 
	Each dictionary is passed on directly to "run" method above. Each dictionary must contain values for at least the 
	['workdir'] parameter.

	test_params: is a list of dictionaries. List must be same length as run_params.
	Each directory is passed on to the "classify" method above. Each dictionary must contain values for the 
	['scorelayer'] parameter.

	cycle_sizes: array of ints of the same length as run_params. 
	Cycle_sizes determines the nbr iterations for each experiment in run_params list.
	
	ncycles: integer.
	Total number of cycles to complete.



	"""
	run_defaults = {'solver':'solver.prototxt', 'GPU_id':0, 'log':'train.log','snapshot_prefix':'snapshot','caffepath':'/home/beijbom/cc/build/tools/caffe', 'restart': False}
	test_defaults = {'caffemodel':None, 'snapshot_prefix':'snapshot', 'GPU_id':0, 'save':True, 'ignore_label':255, 'n_testinstances':None}
	for cycle in range(ncycles):
		for (cycle_size, params, tparams) in zip(cycle_sizes, run_params, test_params):
			# add defaults to run_parameter dict
			for key in list(set(run_defaults) - set(params)):
				params[key] = run_defaults[key]
			# find the current iteration
			snapshots = glob.glob(os.path.join(params['workdir'], "{}*.solverstate".format(params['snapshot_prefix'])))
			if snapshots:
				_iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in snapshots]
				max_iter = np.max(_iter) + cycle_size - 1
			else:
				max_iter = cycle_size - 1

			# update solver with new max_iter parameter
			solver = CaffeSolver()
			solver.add_from_file(os.path.join(params['workdir'], params['solver']))
			solver.sp['max_iter'] = str(max_iter)
			solver.sp['snapshot'] = str(1000000) #disable this, don't need it.
			solver.write(os.path.join(params['workdir'], params['solver']))
			print("Running {} from {} to {} itts.".format(params['workdir'], max_iter-cycle_size+1, max_iter+1))
			run(**params)

			# classify all *net.prototxt in workdir
			testnets = glob.glob(os.path.join(params['workdir'], '*net.prototxt'))
			for testnet in testnets:
				# add params to tparams dict
				tparams['workdir'] = params['workdir'] #assuming the same workdir
				tparams['net_prototxt'] = testnet
				for key in list(set(test_defaults) - set(tparams)):
					tparams[key] = test_defaults[key]
				classify(**tparams)



def cycle_runs_debug(run_params, test_params):
	"""
	This is to debug the cycle_runs parameters and setup, before pressing play.
	"""
	run_params = deepcopy(run_params)
	test_params = deepcopy(test_params)
	print('Running tests...')
	for test_param in test_params:
		test_param['n_testinstances'] = 5
	cycle_sizes = np.ones(len(test_params), dtype = np.int) * 4 #4 iterations
	cycle_runs(run_params, test_params, cycle_sizes, 1)
	print('Run test OK. Cleaning up.')
	for run_param in run_params:
		for file_ in glob.glob(os.path.join(run_param['workdir'], 'snapshot*')):
			os.remove(file_)
		for file_ in glob.glob(os.path.join(run_param['workdir'], 'predictions_on*')):
			os.remove(file_)
		os.remove(os.path.join(run_param['workdir'], 'train.log'))

def load_model(workdir, caffemodel, GPU_id = 0, net_prototxt = 'net.prototxt', phase = caffe.TEST):
	"""
	changes current directory to INPUT workdir and loads INPUT net_prototxt.
	"""
	os.chdir(workdir)
	caffe.set_device(GPU_id)
	caffe.set_mode_gpu()
	net = caffe.Net(net_prototxt, caffemodel, phase)
	net.forward() #one forward to initialize the net
	return net


def nbr_lines(fname):
	"""
	Opens INPUT file fname and returns the number of lines in the file.
	"""
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1


def classify_imlist(im_list, net, transformer, batch_size, scorelayer, startlayer = 'conv1_1'):
	"""
	classify_imlist classifies a list of images and returns estimated labels and scores. Only support classification nets (not FCNs).

	Takes
	im_list: list of images to classify (each stored as a numpy array).
	net: caffe net object
	transformer: transformer object as defined above.
	batch_size: batch size for the net.
	scorelayer: name of the score layer.
	startlayer: name of first convolutional layer.
	"""

	nbatches = int(math.ceil(float(len(im_list)) / batch_size))
	scorelist = []
	pos = -1
	for b in range(nbatches):
		for i in range(batch_size):
			pos += 1
			if pos < len(im_list):
				net.blobs['data'].data[i, :, :, :] = transformer.preprocess(im_list[pos])
		net.forward(start = startlayer)
		for i in range(batch_size):
			scorelist.append(net.blobs[scorelayer].data[i,:].flatten())
	scorelist = scorelist[:len(im_list)]
	estlist = [np.argmax(s) for s in scorelist]  
	
	return(estlist, scorelist)


def sac(im, net, transformer, scorelayer, target_size = [1024, 1024], padcolor = [126, 148, 137], startlayer = 'conv1_1'):
	"""
	sac (slice and classify) slices the input image, feed each piece to the
	caffe net object, and then stitch the output back together to an output image

	Takes
	im: input numpy array.
	net: Caffe net object.
	transformer: transformer object as defined above.
	scorelayer: string defining the name of the score layer.
	target_size: size of each slice.
	padcolor: the RGB values used when padding the image.
	startlayer: string defining the name of first convolutional layer.

	Gives
	(est, scores) tuple, where est is an integer image of the same size as the input, and scores is a multi-layer image encoding the score of each class in each layer.

	"""

	input_size = im.shape[:2]
	(imlist, ncells) = bmt.slice_image(im, target_size = target_size, padcolor = padcolor)
	imcounter = -1
	for row in range(ncells[0]):
		for col in range(ncells[1]):
			imcounter += 1
			net.blobs['data'].data[...] = transformer.preprocess(imlist[imcounter])
			net.forward(start = startlayer)
			scores_slice = np.float32(np.squeeze(net.blobs[scorelayer].data.transpose(2, 3, 1, 0)))
			if col == 0:
				scores_row = deepcopy(scores_slice)
			else:
				scores_row = np.concatenate((scores_row, scores_slice), axis = 1) # Build one row (along the columns)
		if row == 0:
			scores = scores_row
		else:
			scores = np.concatenate((scores, scores_row), axis = 0) # Concatenate the rows
	scores = scores[:input_size[0], :input_size[1], :] # Crop away the padding.
	est = np.argmax(scores, axis = 2) # For convenience, get the predictions.
	return (est, scores)
