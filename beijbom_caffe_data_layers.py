# standard python imports
import os.path
import json
from random import shuffle
from threading import Thread
import numpy as np
from PIL import Image

# own class imports
import caffe
from beijbom_misc_tools import crop_and_rotate
from beijbom_misc_tools import tile_image
from beijbom_caffe_tools import Transformer

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

        self.batch_size = params['batch_size']
        crop_size = params['crop_size']
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
        self.batch_advancer = PatchBatchAdvancer(self.thread_result, self.batch_size, imlist, imdict, imgs_per_batch, crop_size, transformer)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, crop_size, crop_size)
        top[1].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        """ happens during setup """
        pass

    def forward(self, bottom, top):
        if self.thread is not None:
            self.join_worker() 

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 

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
    def __init__(self, result, batch_size, imlist, imdict, imgs_per_batch, crop_size, transformer):
        self.result = result
        self.batch_size = batch_size
        self.imlist = imlist
        self.imdict = imdict
        self.imgs_per_batch = imgs_per_batch
        self.crop_size = crop_size
        self.transformer = transformer
        self._cur = 0
        print "The mighty PatchBatchAdvancer is initialized with {} images, {} imgs per batch, and {}x{} pixel patches".format(len(imlist), imgs_per_batch, crop_size, crop_size)

    def __call__(self):

        self.result['data'] = []
        self.result['labels'] = []

        if self._cur + self.imgs_per_batch > len(self.imlist):
            print "The mighty PatchBatchAdvancer finished an epoch. Shuffling image list"
            self._cur = 0
            shuffle(self.imlist)
        
        # Grab images names from imlist
        imnames = self.imlist[self._cur : self._cur + self.imgs_per_batch]

        # Figure out how many patches to grab from each image
        patches_per_image = self.chunkify(self.batch_size, self.imgs_per_batch)

        # Make nice output string
        output_str = [str(npatches) + ' from ' + os.path.basename(imname) + '(id ' + str(itt) + ')' for imname, npatches, itt in zip(imnames, patches_per_image, range(self._cur, self._cur + self.imgs_per_batch))]
        
        print "The mighty PatchBatchAdvancer is producing patches: [{}]".format(", ".join(output_str))
        # Loop over each image
        for imname, npatches in zip(imnames, patches_per_image):
            self._cur += 1

            # randomly select the rotation angle for each patch             
            angles = np.random.choice(360, size = npatches, replace = True)

            # randomly permute the patch list for this image. Sampling is done with replacement 
            # so that if we ask for more patches than the batch size it still computes.
            point_anns = self.imdict[os.path.basename(imname)][0]
            point_anns = [point_anns[pp] for pp in np.random.choice(len(point_anns), size = npatches, replace = True)]

            # Load image
            im = np.asarray(Image.open(imname))
            scale = 1 # just building for the future
            if not scale == 1:
                im = scipy.misc.imresize(im, scale) 
            offset = np.asarray(im.shape[:2])
            im = tile_image(im)        
            
            # crop patches
            for ((row, col, label), angle) in zip(point_anns, angles):
                # print "processing row:{}, col:{}, label:{}, angle:{}, from image:{}".format(row, col, label, angle, imname)
                center_org = np.asarray([row, col])
                center = np.round(offset + center_org * scale).astype(np.int)
                patch = self.transformer(crop_and_rotate(im, center, self.crop_size, angle, tile = False))
                self.result['data'].append(patch)
                self.result['labels'].append(label)

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