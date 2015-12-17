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
from settings import CAFFEPATH
from caffe import layers as L, params as P
from beijbom_misc_tools import coral_image_resize, crop_and_rotate

"""
beijbom_caffe_tools (bct) contains classes and wrappers for caffe.
"""

class Transformer:
    """
    Transformer is a class for preprocessing and deprocessing images according to the vgg16 pre-processing paradigm (scaling and mean subtraction.)
    """

    def __init__(self, mean = [0, 0, 0]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

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
        im = im[:, :, ::-1] #change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))
        
        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1] #change to RGB
        
        return np.uint8(im)



class CaffeSolver:
    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default values and can export a solver parameter file.
    Note that all parameters are stored as strings. For technical reasons, the strings are stored as strings within strings.
    """

    def __init__(self, testnet_prototxt_path = "testnet.prototxt", trainnet_prototxt_path = "trainnet.prototxt", debug = False):
        
        self.sp = {}

        # critical:
        self.sp['base_lr'] = '0.001'
        self.sp['momentum'] = '0.9'
        
        # speed:
        self.sp['test_iter'] = '100'
        self.sp['test_interval'] = '250'
        
        # looks:
        self.sp['display'] = '25'
        self.sp['snapshot'] = '2500'
        self.sp['snapshot_prefix'] = '"snapshot"' # string withing a string!
        
        # learning rate policy
        self.sp['lr_policy'] = '"fixed"'

        # important, but rare:
        self.sp['gamma'] = '0.1'
        self.sp['weight_decay'] = '0.0005'
        self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'
        self.sp['test_net'] = '"' + testnet_prototxt_path + '"'

        # pretty much never change these.
        self.sp['max_iter'] = '100000'
        self.sp['test_initialization'] = 'false'
        self.sp['average_loss'] = '25' # this has to do with the display.
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
                if line[0] == '#':
                    continue
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



def run(workdir = None, caffemodel = None, GPU_id = 0, solverfile = 'solver.prototxt', log = 'train.log', snapshot_prefix = 'snapshot', caffepath = CAFFEPATH, restart = False, nbr_iters = None):
    """
    run is a simple caffe wrapper for training nets. It basically does two things. (1) ensures that training continues from the most recent model, and (2) makes sure the output is captured in a log file.

    Takes
    workdir: directory where the net prototxt lives.
    caffemodel: name of a stored caffemodel.
    solverfile: name of solver.prototxt [this refers, in turn, to net.prototxt]
    log: name of log file
    snapshot_prefix: snapshot prefix. 
    caffepath: path the caffe binaries. This is required since we make a system call to caffe.
    restart: determines whether to restart even if there are snapshots in the directory.

    """

    # find initial caffe model
    if not caffemodel:
        caffemodel = glob.glob(os.path.join(workdir, "*initial.caffemodel"))
        if caffemodel:
            caffemodel = os.path.basename(caffemodel[0])

    # finds the latest snapshots
    snapshots = glob.glob(os.path.join(workdir, "{}*.solverstate".format(snapshot_prefix)))
    if snapshots:
        _iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in snapshots]
        max_iter = np.max(_iter)
        latest_snapshot = os.path.basename(snapshots[np.argmax(_iter)])
    else:
        max_iter = 0

    # update solver with new max_iter parameter (if asked for)
    if not nbr_iters is None: 
        solver = CaffeSolver()
        solver.add_from_file(os.path.join(workdir, solverfile))
        solver.sp['max_iter'] = str(max_iter + nbr_iters)
        solver.sp['snapshot'] = str(1000000) #disable this, don't need it.
        solver.write(os.path.join(workdir, solverfile))

    print caffepath
    # by default, start from the most recent snapshot
    if snapshots and not(restart): 
        print "Running {} from iter {}.".format(workdir, np.max(_iter))
        runstring  = 'cd {}; {} train -solver {} -snapshot {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solverfile, latest_snapshot, GPU_id, log)

    # else, start from a pre-trained net defined in caffemodel
    elif(caffemodel): 
        if(os.path.isfile(os.path.join(workdir, caffemodel))):
            print "Fine tuning {} from {}.".format(workdir, caffemodel)
            runstring  = 'cd {}; {} train -solver {} -weights {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solverfile, caffemodel, GPU_id, log)

        else:
            raise IOError("Can't fine intial weight file: " + os.path.join(workdir, caffemodel))

    # Train from scratch. Not recommended for larger nets.
    else: 
        print "No caffemodel specified. Running {} from scratch!!".format(workdir)
        runstring  = 'cd {}; {} train -solver {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solverfile, GPU_id, log)
    os.system(runstring)



def classify(workdir, scorelayer, caffemodel = None, GPU_id = 0, labellayer = 'label', snapshot_prefix = 'snapshot', net_prototxt = 'net.prototxt', save = False, ignore_label = np.inf, n_testinstances = None, batch_size = None):
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
    
    # find latest model
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
        if net.blobs[labellayer].data.ndim == 1:
            gtlist.extend(list(copy(net.blobs[labellayer].data).astype(np.uint8)))
            scorelist.extend(list(copy(net.blobs[scorelayer].data).astype(np.float)))
        else:
            gt = copy(net.blobs[labellayer].data.transpose(0, 2, 3, 1)).astype(np.uint8)
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
    if net.blobs[labellayer].data.ndim == 1: 
        gtlist = gtlist[:n_testinstances]
        scorelist = scorelist[:n_testinstances]

    # For convenience, include estimated labels
    estlist = [np.argmax(s) for s in scorelist]
    if (save):
        pickle.dump((gtlist, estlist, scorelist), open(os.path.join(workdir, 'predictions_on_' + test_file[5:] + '_using_' + caffemodel +  '.p'), 'wb'))

    return (gtlist, estlist, scorelist)



def cycle_runs(run_params, test_params, cycle_sizes, ncycles, classify = True):
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
            params['nbr_iters'] = cycle_size
            run(**params)        

            if classify:
                # classify all *net.prototxt in workdir
                testnets = glob.glob(os.path.join(params['workdir'], '*net.prototxt'))
                for testnet in testnets:
                    # add params to tparams dict
                    tparams['workdir'] = params['workdir'] #assuming the same workdir
                    tparams['net_prototxt'] = testnet
                    for key in list(set(test_defaults) - set(tparams)):
                        tparams[key] = test_defaults[key]
                    classify(**tparams)



def cycle_runs_debug(run_params, test_params, classify=True):
    """
    This is to debug the cycle_runs parameters and setup, before pressing play.
    """
    run_params = deepcopy(run_params)
    test_params = deepcopy(test_params)
    print('Running tests...')
    for test_param in test_params:
        test_param['n_testinstances'] = 5
    cycle_sizes = np.ones(len(test_params), dtype = np.int) * 4 #4 iterations
    cycle_runs(run_params, test_params, cycle_sizes, 1, classify)
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
        scorelist.extend(list(copy(net.blobs[scorelayer].data).astype(np.float)))
        
    scorelist = scorelist[:len(im_list)]
    estlist = [np.argmax(s) for s in scorelist]  
    
    return(estlist, scorelist)


def classify_from_patchlist(imlist, imdict, pyparams, workdir, scorelayer = 'score', startlayer = 'conv1_1', net_prototxt = 'testnet.prototxt', GPU_id = 0, snapshot_prefix = 'snapshot', save = False):

    # Preliminaries    
    caffemodel = find_latest_caffemodel(workdir, snapshot_prefix = snapshot_prefix)
    net = load_model(workdir, caffemodel, GPU_id = GPU_id, net_prototxt = net_prototxt)
    transformer = Transformer(pyparams['im_mean'])
    estlist, scorelist, gtlist = [], [], []
    
    print "classifying {} images in {} using {}".format(len(imlist), workdir, caffemodel)
    for imname in tqdm(imlist):
        
        patchlist = []
        (point_anns, height_cm) = imdict[os.path.basename(imname)]

        # Load image
        im = np.asarray(Image.open(imname))
        (im, scale) = coral_image_resize(im, pyparams['scaling_method'], pyparams['scaling_factor'], height_cm) #resize.

        # Pad the boundaries                        
        im = np.pad(im, ((pyparams['crop_size']*2, pyparams['crop_size']*2),(pyparams['crop_size']*2, pyparams['crop_size']*2), (0, 0)), mode='reflect')        
        
        # Extract patches
        for (row, col, label) in point_anns:
            center_org = np.asarray([row, col])
            center = np.round(pyparams['crop_size']*2 + center_org * scale).astype(np.int)
            patchlist.append(crop_and_rotate(im, center, pyparams['crop_size'], 0, tile = False))
            gtlist.append(label)

        # Classify and append
        [this_estlist, this_scorelist] = classify_imlist(patchlist, net, transformer, pyparams['batch_size'], scorelayer = scorelayer, startlayer = startlayer)
        estlist.extend(this_estlist)
        scorelist.extend(this_scorelist)
        
    if (save):
        pickle.dump((gtlist, estlist, scorelist), open(os.path.join(workdir, 'predictions_using_' + caffemodel +  '.p'), 'wb'))
    return [gtlist, estlist, scorelist]


def find_latest_caffemodel(workdir, snapshot_prefix = 'snapshot'):
    
    caffemodels = glob.glob("{}*.caffemodel".format(os.path.join(workdir, snapshot_prefix)))
    if caffemodels:
        _iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in caffemodels]
        return os.path.basename(caffemodels[np.argmax(_iter)])
    else:
        print "Can't find a trained model in " + workdir + " using prefix: " + snapshot_prefix + "."
        return None


def calculate_image_mean(imlist): 
    """
    Returns mean channel intensity across the images in imlist.
    NOTE: returns mean in in BGR order.
    """
    mean = np.zeros(3).astype(np.float32)
    for imname in imlist:
        im = np.asarray(Image.open(imname))
        if len(im.shape) == 2:
            im = np.mean(im, axis = 0)
            im = np.mean(im, axis = 0)
            mean = mean + [im, im, im]
        else:   
            im = im[:, :, ::-1] #change to BGR
            im = np.mean(im, axis = 0)
            im = np.mean(im, axis = 0)
            mean = mean + im
    mean /= len(imlist)
    print mean
    return mean


def clean_workdirs(workdirs):
    for workdir in workdirs:
        for file_ in glob.glob(os.path.join(workdir, 'snapshot*')):
            if os.path.isfile(file_):
                os.remove(file_)
        for file_ in glob.glob(os.path.join(workdir, 'predictions_*')):
            if os.path.isfile(file_):
                os.remove(file_)
        for file_ in glob.glob(os.path.join(workdir, '*.log')):
            if os.path.isfile(file_):
                os.remove(file_)
    for file_ in glob.glob(os.path.join(workdir, '*.testlog')):
            if os.path.isfile(file_):
                os.remove(file_)



def vgg(pydata_params, data_layer, nclasses, ntop = 2, acclayer = False):
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'beijbom_caffe_data_layers', layer = data_layer,
            ntop=ntop, param_str=str(pydata_params))

    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    n.fc6 = L.InnerProduct(n.pool5, num_output=4096,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    n.relu6 = L.ReLU(n.fc6, in_place=True)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7 = L.InnerProduct(n.fc6, num_output=4096,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    n.relu7 = L.ReLU(n.fc7, in_place=True)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.score = L.InnerProduct(n.fc7, num_output=nclasses,
        param=[dict(lr_mult=5, decay_mult=1), dict(lr_mult=10, decay_mult=0)])

    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    
    if acclayer:
        n.accuracy = L.Accuracy(n.score, n.label)
    return n
    #return n.to_proto()

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, learn=True):
    if learn:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]

    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, param=param)
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def residual_standard_unit(bottom, nout, newdepth = False, learn = True):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    if learn:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    
    stride = 2 if newdepth else 1
    conv1 = L.Convolution(bottom, kernel_size = 3, stride = stride, num_output = nout, weight_filler=dict(type="xavier"), bias_filler=dict(type="constant"), pad = 1, param = param)
    bn1 = L.BatchNorm(conv1)
    relu1 = L.ReLU(bn1, in_place=True)
    conv2 = L.Convolution(relu1, kernel_size = 3, stride = 1,
        num_output=nout, weight_filler=dict(type="xavier"), bias_filler=dict(type="constant"), pad = 1, param = param)
    bn2 = L.BatchNorm(conv2)
   
    if newdepth: 
        conv_compress = L.Convolution(bottom, kernel_size = 1, stride = 2, num_output = nout, pad = 0, param = param)
        bn_compress = L.BatchNorm(conv_compress)
        sumlayer = L.Eltwise(bn2, bn_compress)
    else:
        conv_compress = ''
        bn_compress = ''
        sumlayer = L.Eltwise(bn2, bottom)

    relu2 = L.ReLU(sumlayer, in_place=True)
    return conv1, conv2, relu1, relu2, bn1, bn2, conv_compress, bn_compress, sumlayer 

def residual_bottleneck_unit(bottom, nout, newdepth = False, learn = True):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    if learn:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    
    stride = 2 if newdepth else 1
    conv1 = L.Convolution(bottom, kernel_size = 1, pad = 0, stride = stride, num_output = nout, weight_filler=dict(type="xavier"), bias_filler=dict(type="constant"), param=param)
    bn1 = L.BatchNorm(conv1)
    relu1 = L.ReLU(bn1, in_place=True)
    conv2 = L.Convolution(relu1, kernel_size = 3, stride = 1, pad = 1,
        num_output=nout, weight_filler=dict(type="xavier"), bias_filler=dict(type="constant"), param=param)
    bn2 = L.BatchNorm(conv2)
    relu2 = L.ReLU(bn2, in_place=True)
    conv3 = L.Convolution(relu2, kernel_size = 1, stride = 1, pad = 0,
        num_output = nout * 4, weight_filler = dict(type="xavier"), bias_filler = dict(type="constant"), param=param)
    bn3 = L.BatchNorm(conv3)
   
    if newdepth: 
        conv_compress = L.Convolution(bottom, kernel_size = 1, stride = 2, num_output = nout * 4, pad = 0, param=param)
        bn_compress = L.BatchNorm(conv_compress)
        sumlayer = L.Eltwise(bn3, bn_compress)
    else:
        conv_compress = ''
        bn_compress = ''
        sumlayer = L.Eltwise(bn3, bottom)

    relu3 = L.ReLU(sumlayer, in_place=True)
    return conv1, conv2, conv3, relu1, relu2, relu3, bn1, bn2, bn3, conv_compress, bn_compress, sumlayer 


def residual_net(total_depth, data_layer_params, num_classes = 1000, acclayer = True):
    """
    Generates nets from "Deep Residual Learning for Image Recognition". Nets follow architectures outlined in Table 1. 
    """
    # figure out network structure
    net_defs = {
        18:([2, 2, 2, 2], "standard"),
        34:([3, 4, 6, 3], "standard"),
        50:([3, 4, 6, 3], "bottleneck"),
        101:([3, 4, 23, 3], "bottleneck"),
        152:([3, 8, 36, 3], "bottleneck"),
    }
    assert total_depth in net_defs.keys(), "net of depth:{} not defined".format(total_depth)

    nlayerss, unit_type = net_defs[total_depth] #nlayerss is "double plural" for a list of integers indicating the number of layers in each depth type.
    depths = [64, 128, 256, 512]

    # setup the first couple of layers
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'beijbom_caffe_data_layers', layer = 'ImageNetDataLayer',
                ntop = 2, param_str=str(data_layer_params))
    n.conv1 = L.Convolution(n.data, kernel_size = 7, stride = 2, num_output = 64, weight_filler=dict(type="xavier"), bias_filler=dict(type="constant"), pad = 3, param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.bn1 = L.BatchNorm(n.conv1)
    n.relu1 = L.ReLU(n.bn1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, stride = 2, kernel_size = 3)
    
    if unit_type == "standard": #net uses the two-layer unit
        # for each depth and nlayers
        for depth, nlayers in zip(depths, nlayerss):
            for layeritt in range(1, nlayers + 1):
                s = str(depth) + '_' + str(layeritt) + '_' # layer name convenience variable
                if layeritt is 1 and depth>64: # in these situations we enter a new width & depth.
                    n.__dict__['tops'][s + 'conv1'], n.__dict__['tops'][s + 'conv2'], n.__dict__['tops'][s + 'relu1'], n.__dict__['tops'][s + 'relu2'], n.__dict__['tops'][s + 'bn1'], n.__dict__['tops'][s + 'bn2'], n.__dict__['tops'][s + 'conv_compress'], n.__dict__['tops'][s + 'bn_compress'], n.__dict__['tops'][s + 'sum'] = residual_standard_unit(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]], depth, newdepth = True)
                else: # same width and depth as previously.
                    n.__dict__['tops'][s + 'conv1'], n.__dict__['tops'][s + 'conv2'], n.__dict__['tops'][s + 'relu1'], n.__dict__['tops'][s + 'relu2'], n.__dict__['tops'][s + 'bn1'], n.__dict__['tops'][s + 'bn2'], _, _, n.__dict__['tops'][s + 'sum'] = residual_standard_unit(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]], depth, newdepth = False)
    else: #net uses the three-layer bottleneck unit
        # for each depth and nlayers
        for depth, nlayers in zip(depths, nlayerss):
            for layeritt in range(1, nlayers + 1):
                s = str(depth) + '_' + str(layeritt) + '_' # layer name convenience variable
                if layeritt is 1: 
                    n.__dict__['tops'][s + 'conv1'], n.__dict__['tops'][s + 'conv2'], n.__dict__['tops'][s + 'conv3'], n.__dict__['tops'][s + 'relu1'], n.__dict__['tops'][s + 'relu2'], n.__dict__['tops'][s + 'relu3'], n.__dict__['tops'][s + 'bn1'], n.__dict__['tops'][s + 'bn2'], n.__dict__['tops'][s + 'bn3'], n.__dict__['tops'][s + 'conv_compress'], n.__dict__['tops'][s + 'bn_compress'], n.__dict__['tops'][s + 'sum'] = residual_bottleneck_unit(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]], depth, newdepth = True)
                else: # same width and depth as previously.
                    n.__dict__['tops'][s + 'conv1'], n.__dict__['tops'][s + 'conv2'], n.__dict__['tops'][s + 'conv3'], n.__dict__['tops'][s + 'relu1'], n.__dict__['tops'][s + 'relu2'], n.__dict__['tops'][s + 'relu3'], n.__dict__['tops'][s + 'bn1'], n.__dict__['tops'][s + 'bn2'], n.__dict__['tops'][s + 'bn3'], _, _, n.__dict__['tops'][s + 'sum'] = residual_bottleneck_unit(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]], depth, newdepth = False)
    # add the end layers                    
    n.global_pool = L.Pooling(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]], pooling_param = dict(pool=1, global_pooling=True))
    n.score = L.InnerProduct(n.global_pool, num_output = num_classes,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    
    if acclayer:
        n.accuracy = L.Accuracy(n.score, n.label)

    return n            

