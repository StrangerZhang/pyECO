import mxnet as mx
import numpy as np
import pickle
import os
import cv2
from mxnet.gluon.model_zoo import vision
import mxnet as mx

from ..config import config
from  .  import _gradient

import ipdb as pdb

def mround(x):
    x_ = x.copy()
    idx = (x - np.floor(x)) >= 0.5
    x_[idx] = np.floor(x[idx]) + 1
    idx = ~idx
    x_[idx] = np.floor(x[idx])
    return x_

class Feature:
    def _sample_patch(self, im, pos, sample_sz, output_sz):
        pos = np.floor(pos)

        # downsample factor
        resize_factor = np.min(sample_sz / output_sz)
        df = max(np.floor(resize_factor - 0.1), 1)
        if df > 1:
            # compute offset and new center position
            os = (pos - 1) % df
            pos = (pos - 1 - os) / df + 1

            # new sample size
            sample_sz = sample_sz / df

            # downsample image
            im = im[int(os[0])::int(df), int(os[1])::int(df), :]

        sample_sz = np.maximum(mround(sample_sz), 1)
        xs = np.floor(pos[1]) + np.arange(0, sample_sz[1]) - np.floor((sample_sz[1]+1)/2)
        ys = np.floor(pos[0]) + np.arange(0, sample_sz[0]) - np.floor((sample_sz[0]+1)/2)
        xmin = max(0, int(xs.min()))
        xmax = min(im.shape[1], int(xs.max()+1))
        ymin = max(0, int(ys.min()))
        ymax = min(im.shape[0], int(ys.max()+1))
        # extract image
        im_patch = im[ymin:ymax, xmin:xmax, :]

        left = right = top = down = 0
        if xs.min() < 0:
            left = int(abs(xs.min()))
        if xs.max() > im.shape[1]:
            right = int(xs.max() - im.shape[1])
        if ys.min() < 0:
            top = int(abs(ys.min()))
        if ys.max() > im.shape[0]:
            down = int(ys.max() - im.shape[0])
        if left != 0 or right != 0 or top != 0 or down != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, down, left, right, cv2.BORDER_REPLICATE)
        im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])))
        if len(im_patch.shape) == 2:
            im_patch = im_patch[:, :, np.newaxis]
        return im_patch

    def _feature_normalization(self, x):
        if hasattr(config, 'normalize_power') and config.normalize_power > 0:
            if config.normalize_power == 2:
                x = x * np.sqrt((x.shape[0]*x.shape[1]) ** config.normalize_size * (x.shape[2]**config.normalize_dim) / (x**2).sum(axis=(0, 1, 2)))
            else:
                x = x * ((x.shape[0]*x.shape[1]) ** config.normalize_size) * (x.shape[2]**config.normalize_dim) / ((np.abs(x) ** (1. / config.normalize_power)).sum(axis=(0, 1, 2)))

        if config.square_root_normalization:
            x = np.sign(x) * np.sqrt(np.abs(x))
        return x

class ResNet50Feature(Feature):
    def __init__(self, fname, compressed_dim):
        self._ctx = mx.gpu(config.gpu_id) if config.use_gpu else mx.cpu(0)
        self._resnet50 = vision.resnet50_v2(pretrained=True, ctx = self._ctx)
        self._compressed_dim = compressed_dim
        self._stride = [4, 16]
        self._cell_size = [4, 16]
        self.min_cell_size = np.min(self._cell_size)
        self.num_dim = None
        self.penalty = [0., 0.]
        self.sample_sz = None
        self.input_sz = None
        self.data_sz = None

    def init_size(self, img_sample_sz):
        img_sample_sz = img_sample_sz.astype(np.int32)
        feat1, feat2 = self._forward(mx.ndarray.ones(tuple([1, 3, img_sample_sz[0], img_sample_sz[1]]), ctx=self._ctx))
        self.num_dim = [feat1.shape[2], feat2.shape[2]]
        self.sample_sz = img_sample_sz
        self.input_sz = img_sample_sz
        self.data_sz = [np.array(feat1.shape[:2]),
                        np.array(feat2.shape[:2])]
        return img_sample_sz


    def _forward(self, x):
        # stage1
        bn0 = self._resnet50.features[0].forward(x)
        conv1 = self._resnet50.features[1].forward(bn0)     # x2
        bn1 = self._resnet50.features[2].forward(conv1)
        relu1 = self._resnet50.features[3].forward(bn1)
        pool1 = self._resnet50.features[4].forward(relu1)   # x4
        # stage2
        stage2 = self._resnet50.features[5].forward(pool1)  # x4
        stage3 = self._resnet50.features[6].forward(stage2) # x8
        stage4 = self._resnet50.features[7].forward(stage3) # x16
        return [pool1.asnumpy().transpose(2, 3, 1, 0),
                stage4.asnumpy().transpose(2, 3, 1, 0)]

    def get_features(self, img, pos, sample_sz, scales):
        feat1 = []
        feat2 = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        patches = []
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            patch = mx.nd.array(patch, ctx=self._ctx)
            normalized = mx.image.color_normalize(patch,
                                                  mean=mx.nd.array([0.485, 0.456, 0.406], ctx=self._ctx),
                                                  std=mx.nd.array([0.229, 0.224, 0.225], ctx=self._ctx))
            normalized = normalized.transpose((2, 0, 1)).expand_dims(axis=0)
            patches.append(normalized)
        patches = mx.nd.concat(*patches, dim=0)
        f1, f2 = self._forward(patches)
        return f1, f2

def fhog(I, bin_size=8, num_orients=9, clip=0.2, crop=False):
    soft_bin = -1
    M, O = _gradient.gradMag(I.astype(np.float32), 0, True)
    H = _gradient.fhog(M, O, bin_size, num_orients, soft_bin, clip)
    return H

class FHogFeature(Feature):
    def __init__(self, fname, cell_size=6, compressed_dim=10, num_orients=9, clip=.2):
        self.fname = fname
        self._cell_size = cell_size
        self._compressed_dim = [compressed_dim]
        self._soft_bin = -1
        self._bin_size = cell_size 
        self._num_orients = num_orients
        self._clip = clip

        self.min_cell_size = self._cell_size
        self.num_dim = [3 * num_orients + 5 - 1]
        self.penalty = [0.]
        self.sample_sz = None
        self.input_sz = None
        self.data_sz = None

    def init_size(self, img_sample_sz, max_cell_size=None):
        if max_cell_size is not None:
            new_img_sample_sz = (1 + 2 * mround(img_sample_sz / ( 2 * max_cell_size))) * max_cell_size
            feature_sz_choices = (new_img_sample_sz.reshape(-1, 1) + np.arange(0, max_cell_size).reshape(1, -1)) // self.min_cell_size
            num_odd_dimensions = np.sum((feature_sz_choices % 2) == 1, 0)
            best_choice = np.argmax(num_odd_dimensions.flatten())
            pixels_added = best_choice - 1
            img_sample_sz = mround(new_img_sample_sz + pixels_added)

        self.sample_sz = img_sample_sz
        self.input_sz = img_sample_sz
        self.data_sz = [img_sample_sz // self._cell_size]
        return img_sample_sz

    def get_features(self, img, pos, sample_sz, scales):
        feat = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            h, w, c = patch.shape
            M, O = _gradient.gradMag(patch.astype(np.float32), 0, True)
            H = _gradient.fhog(M, O, self._bin_size, self._num_orients, self._soft_bin, self._clip)
            # drop the last dimension
            H = H[:, :, :-1]
            H = self._feature_normalization(H)
            feat.append(H)
        return [np.stack(feat, axis=3)]

class TableFeature(Feature):
    def __init__(self, fname, compressed_dim, table_name, use_for_color, cell_size=1):
        self.fname = fname
        self._table_name = table_name
        self._color = use_for_color
        self._cell_size = cell_size
        self._compressed_dim = [compressed_dim]
        self._factor = 32
        self._den = 8
        # load table
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._table = pickle.load(open(os.path.join(dir_path, "lookup_tables", self._table_name+".pkl"), "rb")) # need to change TODO

        self.num_dim = [self._table.shape[1]]
        self.min_cell_size = self._cell_size
        self.penalty = [0.]
        self.sample_sz = None
        self.input_sz = None
        self.data_sz = None

    def init_size(self, img_sample_sz, max_cell_size=None):
        if max_cell_size is not None:
            new_img_sample_sz = (1 + 2 * mround(img_sample_sz / ( 2 * max_cell_size))) * max_cell_size
            feature_sz_choices = new_img_sample_sz.reshape(-1, 1) + np.arange(0, max_cell_size).reshape(1, -1) // self.min_cell_size
            num_odd_dimensions = np.sum(np.sum((feature_sz_choices % 2) == 1, 0), 1)
            best_choice = np.argmax(num_odd_dimensions.flatten())
            pixels_added = best_choice - 1
            img_sample_sz = mround(new_img_sample_sz + pixels_added)

        self.sample_sz = img_sample_sz
        self.input_sz = img_sample_sz
        self.data_sz = [img_sample_sz // self._cell_size]
        return img_sample_sz

    def integralVecImage(self, img):
        w, h, c = img.shape
        intImage = np.zeros((w+1, h+1, c), dtype=img.dtype)
        intImage[1:, 1:, :] = np.cumsum(np.cumsum(img, 0), 1)
        return intImage

    def average_feature_region(self, features, region_size):
        region_area = region_size ** 2
        if features.dtype == np.float32:
            maxval = 1.
        else:
            maxval = 255
        intImage = self.integralVecImage(features)
        i1 = np.arange(region_size, features.shape[0]+1, region_size).reshape(-1, 1)
        i2 = np.arange(region_size, features.shape[1]+1, region_size).reshape(1, -1)
        region_image = (intImage[i1, i2, :] - intImage[i1, i2-region_size,:] - intImage[i1-region_size, i2, :] + intImage[i1-region_size, i2-region_size, :])  / (region_area * maxval)
        return region_image

    def get_features(self, img, pos, sample_sz, scales):
        feat = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            h, w, c = patch.shape
            if c == 3:
                RR = patch[:, :, 0].astype(np.int32)
                GG = patch[:, :, 1].astype(np.int32)
                BB = patch[:, :, 2].astype(np.int32)
                index = RR // self._den + (GG // self._den) * self._factor + (BB // self._den) * self._factor * self._factor
                features = self._table[index.flatten()].reshape((h, w, self._table.shape[1]))
            else:
                features = self._table[patch.flatten()].reshape((h, w, self._table.shape[1]))
            if self._cell_size > 1:
                features = self.average_feature_region(features, self._cell_size)
            feat.append(self._feature_normalization(features))
        return [np.stack(feat, axis=3)]

