import mxnet as mx
import numpy as np
import pickle
import os
import cv2
from mxnet.gluon.model_zoo import vision

from ..config import config

import _gradient

import ipdb as pdb

def mround(x):
    x_ = x.copy()
    idx = (x - np.floor(x)) >= 0.5
    x_[idx] = np.floor(x[idx]) + 1
    idx = ~idx
    x_[idx] = np.floor(x[idx])
    return x_

class Feature:
    # def __init__(self):
    def _sample_patch(self, im, pos, sample_sz, output_sz):
        pos = np.floor(pos)

        # downsample factor
        resize_factor = np.min(sample_sz / output_sz)
        df = max(np.floor(resize_factor - 0.1), 1)
        if df > 1:
            # compute offset and new center position
            os = mod(pos - 1, df)
            pos = (pos - 1 - os) / df + 1

            # new sample size
            sample_sz = sample_sz / df

            # downsample image
            im = im[os[0]::df, os[1]::df, :]

        sample_sz = np.maximum(mround(sample_sz), 1)
        xmin = int(max(0, pos[1] - np.floor((sample_sz[1]+1)/2)))
        xmax = int(min(im.shape[1], pos[1] + np.floor((sample_sz[1]+1)/2)-1))
        ymin = int(max(0, pos[0] - np.floor((sample_sz[0]+1)/2)))
        ymax = int(min(im.shape[0], pos[0] + np.floor((sample_sz[0]+1)/2)-1))

        # extract image
        im_patch = im[ymin:ymax, xmin:xmax, :]

        im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])))
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
    def __init__(self, fname, img_sample_sz, stage, output_layer=[100, 100], downsample_factor=[2, 1],
            compressed_dim=[16, 64], input_size_mode='adaptive', input_size_scale=1):
        self._resnet50 = vision.resnet50()
        self._output_layer = output_layer
        self._downsample_factor = downsample_factor
        self._compressed_dim = compressed_dim
        self._input_size_mode = input_size_mode
        self._input_size_scale = input_size_scale
        self._stride = [2**x for x in stage]
        self._cell_size = [s * df for s, df in zip(self._stride, self._downsample_factor)]

        self.min_cell_size = np.min(self._cell_size)
        self.num_dim = [self._resnet50[s].shape[3] for s in stage]
        self.penalty = [0.]
        self.sample_sz = None
        self.input_sz = None
        self.data_sz = None

    def init_size(self, img_sample_sz):
        pass

    def get_features(self, frame):
        return None

def fhog(I, bin_size=8, num_orients=9, clip=0.2, crop=False):
    soft_bin = -1
    M, O = _gradient.gradMag(I.astype(np.float32), 0, True)
    H = _gradient.fhog(M, O, bin_size, num_orients, soft_bin, clip)
    return H

class FHogFeature(Feature):
    def __init__(self, fname, cell_size=6, compressed_dim=10, num_orients=9, clip=.2):
        self.fname = fname
        self._cell_size = cell_size
        self._compressed_dim = compressed_dim
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
        self.data_sz = img_sample_sz // self._cell_size
        return img_sample_sz

    def get_features(self, img, pos, sample_sz, scales):
        patches = self._sample_patch(img, pos, sample_sz*scales, sample_sz)
        h, w, c = patches.shape
        # features = []
        # for img in imgs:
        M, O = _gradient.gradMag(patches.astype(np.float32), 0, True)
        H = _gradient.fhog(M, O, self._bin_size, self._num_orients, self._soft_bin, self._clip)
        # drop the last dimension
        H = H[:, :, :-1]
        H = self._feature_normalization(H)
        return H
        # features.append(H)
        # features = np.stack(features, axis=3)
        # return features

class TableFeature(Feature):
    def __init__(self, fname, compressed_dim, table_name, use_for_color, cell_size=1):
        self.fname = fname
        self._table_name = table_name
        self._color = use_for_color
        self._cell_size = cell_size
        self._compressed_dim = compressed_dim
        self._factor = 32
        self._den = 8
        # load table
        self._table = pickle.load(open(os.path.join("./lookup_tables", self._table_name+".pkl"), "rb"))

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
        self.data_sz = img_sample_sz // self._cell_size
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
        patches = self._sample_patch(img, pos, sample_sz*scales, sample_sz)
        h, w, c = patches.shape
        if c == 3:
            RR = patches[:, :, 0].astype(np.int32)
            GG = patches[:, :, 1].astype(np.int32)
            BB = patches[:, :, 2].astype(np.int32)
            index = RR // self._den + (GG // self._den) * self._factor + (BB // self._den) * self._factor * self._factor
            features = self._table[index.flatten()].reshape((h, w, self._table.shape[1]))
        else:
            features = self._table[img.flatten()].reshape((h, w, self._table.shape[1]))
        if self._cell_size > 1:
            features = self.average_feature_region(features, self._cell_size)
        features = self._feature_normalization(features)
        return features

