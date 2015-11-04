# standard python imports
import os.path
import json
import time
from random import shuffle
from threading import Thread
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import scipy.misc
import skimage.io

# own class imports
import caffe
from beijbom_misc_tools import crop_and_rotate
from beijbom_misc_tools import tile_image
from beijbom_caffe_tools import Transformer


# ==============================================================================
# ==============================================================================
# =========================== RANDOM POINT PATCH LAYER =========================
# ==============================================================================
# ==============================================================================

class RandomPointDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'labels']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'imgs_per_batch' in params.keys(), 'Params must include imgs_per_batch.'
        assert 'crop_size' in params.keys(), 'Params must include crop_size.'
        assert 'im_scale' in params.keys(), 'Params must include im_scale.'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'
        assert 'cache_images' in params.keys(), 'Params must include cache_images.'

        self.t0 = 0
        self.t1 = 0
        self.batch_size = params['batch_size']
        crop_size = params['crop_size']
        cache_images = params['cache_images']
        imgs_per_batch = params['imgs_per_batch']
        imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            imdict = json.load(f)

        transformer = TransformerWrapper()
        transformer.set_mean(params['im_mean'])
        transformer.set_scale(params['im_scale'])

        # === Check some of the input variables
        assert len(imlist) >= imgs_per_batch, 'Image list must be longer than the number of images you ask for per batch.'

        print "Setting up RandomPointDataLayer with batch size:{}".format(self.batch_size)

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = PatchBatchAdvancer(self.thread_result, self.batch_size, imlist, imdict, imgs_per_batch, crop_size, transformer, cache_images)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, crop_size, crop_size)
        top[1].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        """ happens during setup """
        pass

    def forward(self, bottom, top):
        #print time.clock() - self.t0, "seconds since last call to forward."
        if self.thread is not None:
            self.t1 = timer()
            self.join_worker()
            #print "Waited ", timer() - self.t1, "seconds for join."

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.t0 = time.clock()
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class PatchBatchAdvancer():
    """
    The PatchBatchAdvancer is a helper class to RandomPointDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, batch_size, imlist, imdict, imgs_per_batch, crop_size, transformer, cache_images):
        self.result = result
        self.batch_size = batch_size
        self.imlist = imlist
        self.imdict = imdict
        self.imgs_per_batch = imgs_per_batch
        self.crop_size = crop_size
        self.transformer = transformer
        self._cur = 0
        self.cache_images = cache_images
        self.imcache = {}
        shuffle(self.imlist)

        if self.cache_images is True:
            # Load all images to RAM
            print "TMPBA is caching images"
            t0 = timer()
            for imname in self.imlist:
                im = np.asarray(Image.open(imname))
                im = np.pad(im, ((self.crop_size*2, self.crop_size*2),(self.crop_size*2, self.crop_size*2), (0, 0)), mode='reflect') 
                self.imcache[imname] = im
            print "TMPBA cached images in {} seconds".format(timer() - t0)


        print "PatchBatchAdvancer is initialized with {} images, {} imgs per batch, and {}x{} pixel patches".format(len(imlist), imgs_per_batch, crop_size, crop_size)

    def __call__(self):
        t1 = timer()
        self.result['data'] = []
        self.result['labels'] = []

        if self._cur + self.imgs_per_batch >= len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        # Grab images names from imlist
        imnames = self.imlist[self._cur : self._cur + self.imgs_per_batch]

        # Figure out how many patches to grab from each image
        patches_per_image = self.chunkify(self.batch_size, self.imgs_per_batch)

        # Make nice output string
        # output_str = [str(npatches) + ' from ' + os.path.basename(imname) + '(id ' + str(itt) + ')' for imname, npatches, itt in zip(imnames, patches_per_image, range(self._cur, self._cur + self.imgs_per_batch))]
        
        # Loop over each image
        for imname, npatches in zip(imnames, patches_per_image):
            self._cur += 1

            # randomly select the rotation angle for each patch             
            angles = np.random.choice(360, size = npatches, replace = True)

            # randomly select whether to flip this particular patch.
            flips = np.round(np.random.rand(npatches))*2-1

            # Randomly permute the patch list for this image. Sampling is done with replacement 
            # so that if we ask for more patches than is available, it still computes.
            point_anns = self.imdict[os.path.basename(imname)][0]
            point_anns = [point_anns[pp] for pp in np.random.choice(len(point_anns), size = npatches, replace = True)]

            if self.cache_images:
                im = self.imcache[imname]
            else:
                # Load image
                im = np.asarray(Image.open(imname))
                # Pad the boundaries                        
                im = np.pad(im, ((self.crop_size*2, self.crop_size*2),(self.crop_size*2, self.crop_size*2), (0, 0)), mode='reflect')        
            
            for ((row, col, label), angle, flip) in zip(point_anns, angles, flips):
                center_org = np.asarray([row, col])
                center = np.round(self.crop_size*2 + center_org).astype(np.int)
                patch = self.transformer(crop_and_rotate(im, center, self.crop_size, angle, tile = False))
                self.result['data'].append(patch[::flip, :, :])
                self.result['labels'].append(label)
        #print 'TMPBA finished in {} seconds.'.format(timer() - t1)

    def chunkify(self, k, n):
        """ 
        Returns a list of n integers, so that the sum of the n integers is k.
        The list is generated so that the n integers are as even as possible
        """
        lst = range(k)
        return [ len(lst[i::n]) for i in xrange(n) ]
        

class TransformerWrapper(Transformer):
    def __init__(self):
        Transformer.__init__(self)
    def __call__(self, im):
        return self.preprocess(im)




# ==============================================================================
# ==============================================================================
# ============================== REGRESSION LAYER ==============================
# ==============================================================================
# ==============================================================================

class RandomPointRegressionDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.top_names = ['data', 'labels']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'im_scale' in params.keys(), 'Params must include im_scale.'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'

        self.t0 = 0
        self.t1 = 0
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']
        self.nclasses = params['nclasses']
        imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            imdict = json.load(f)

        transformer = TransformerWrapper()
        transformer.set_mean(params['im_mean'])
        transformer.set_scale(params['im_scale'])

        print "Setting up RandomPointRegressionDataLayer with batch size:{}".format(self.batch_size)

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = RegressionBatchAdvancer(self.thread_result, self.batch_size, imlist, imdict, transformer, self.nclasses, self.im_shape)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, self.nclasses)

    def reshape(self, bottom, top):
        """ happens during setup """
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, self.nclasses)
        #pass

    def forward(self, bottom, top):
        # print time.clock() - self.t0, "seconds since last call to forward."
        if self.thread is not None:
            self.t1 = timer()
            self.join_worker()
            # print "Waited ", timer() - self.t1, "seconds for join."

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.t0 = time.clock()
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class RegressionBatchAdvancer():
    """
    The RegressionBatchAdvancer is a helper class to RandomPointRegressionDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, batch_size, imlist, imdict, transformer, nclasses, im_shape):
        self.result = result
        self.batch_size = batch_size
        self.imlist = imlist
        self.imdict = imdict
        self.transformer = transformer
        self._cur = 0
        self.nclasses = nclasses
        self.im_shape = im_shape
        shuffle(self.imlist)

        print "RegressionBatchAdvancer is initialized with {} images".format(len(imlist))

    def __call__(self):
        
        t0 = timer()
        self.result['data'] = []
        self.result['labels'] = []

        if self._cur == len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        imname = self.imlist[self._cur]

        # Load image
        im = np.asarray(Image.open(imname))
        im = scipy.misc.imresize(im, self.im_shape)
        point_anns = self.imdict[os.path.basename(imname)][0]

        class_hist = np.zeros(self.nclasses).astype(np.float32)
        for (row, col, label) in point_anns:
            class_hist[label] += 1
        class_hist /= len(point_anns)

                
        self.result['data'].append(self.transformer.preprocess(im))
        self.result['labels'].append(class_hist)
        self._cur += 1
        # print "loaded image {} in {} secs.".format(self._cur, timer() - t0)


# ==============================================================================
# ==============================================================================
# ============================== MULTILABEL LAYER ==============================
# ==============================================================================
# ==============================================================================

class RandomPointMultiLabelDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'labels']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'im_scale' in params.keys(), 'Params must include im_scale.'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'

        self.t0 = 0
        self.t1 = 0
        self.batch_size = params['batch_size']
        self.nclasses = params['nclasses']
        self.im_shape = params['im_shape']
        imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            imdict = json.load(f)

        transformer = TransformerWrapper()
        transformer.set_mean(params['im_mean'])
        transformer.set_scale(params['im_scale'])

        print "Setting up RandomPointRegressionDataLayer with batch size:{}".format(self.batch_size)

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = MultiLabelBatchAdvancer(self.thread_result, self.batch_size, imlist, imdict, transformer, self.nclasses, self.im_shape)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, self.nclasses)

    def reshape(self, bottom, top):
        """ happens during setup """
        pass

    def forward(self, bottom, top):
        # print time.clock() - self.t0, "seconds since last call to forward."
        if self.thread is not None:
            self.t1 = timer()
            self.join_worker()
            # print "Waited ", timer() - self.t1, "seconds for join."

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.t0 = time.clock()
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class MultiLabelBatchAdvancer():
    """
    The MultiLabelBatchAdvancer is a helper class to RandomPointRegressionDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, batch_size, imlist, imdict, transformer, nclasses, im_shape):
        self.result = result
        self.batch_size = batch_size
        self.imlist = imlist
        self.imdict = imdict
        self.transformer = transformer
        self._cur = 0
        self.nclasses = nclasses
        self.im_shape = im_shape
        shuffle(self.imlist)

        print "MultiLabelBatchAdvancer is initialized with {} images".format(len(imlist))

    def __call__(self):
        
        t0 = timer()
        self.result['data'] = []
        self.result['labels'] = []

        if self._cur == len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        imname = self.imlist[self._cur]

        # Load image
        im = np.asarray(Image.open(imname))
        im = scipy.misc.imresize(im, self.im_shape)
        point_anns = self.imdict[os.path.basename(imname)][0]

        class_in_image = np.zeros(self.nclasses).astype(np.float32)
        for (row, col, label) in point_anns:
            class_in_image[label] = 1

                
        self.result['data'].append(self.transformer.preprocess(im))
        self.result['labels'].append(class_in_image)
        self._cur += 1
        # print "loaded image {} in {} secs.".format(self._cur, timer() - t0)