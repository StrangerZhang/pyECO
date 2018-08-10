import numpy as np
import cv2

from scipy import signal
from numpy.fft import fft2, fftshift, ifft2

from .config import config
from .features import FHogFeature, TableFeature, mround
from .fourier_tools import cfft2, interpolate_dft, shift_sample, full_fourier_coeff, cubic_spline_fourier, compact_fourier_coeff
from .optimize_score import optimize_score
from .sample_space_model import update_sample_space_model
from .train import train_joint, train_filter
from .scale_filter import ScaleFilter

import ipdb as pdb

class ECOTracker:
    def __init__(self, width, height, is_color):
        self._width = width
        self._height = height
        self._is_color = is_color
        self._frame_num = 0
        self._frames_since_last_train = np.inf

    def _cosine_window(self, size):
        cos_window = signal.hann(int(size[0]+2))[:, np.newaxis].dot(signal.hann(int(size[1]+2))[np.newaxis, :])
        cos_window = cos_window[1:-1, 1:-1]
        return cos_window

    def _get_interp_fourier(self, sz):
        if config.interp_method == 'none':
            interp1_fs = np.ones((sz[0], 1), dtype=np.float64)
            interp2_fs = np.ones((1, sz[1]), dtype=np.float64)
        elif config.interp_method == 'ideal':
            interp1_fs = np.ones((sz[0], 1)) / sz[0]
            interp2_fs = np.ones((1, sz[0])) / sz[1]
        elif config.interp_method == 'bicubic':
            f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float64)[:, np.newaxis] / sz[0]
            interp1_fs = np.real(cubic_spline_fourier(f1, config.interp_bicubic_a) / sz[0])
            f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float64)[np.newaxis, :] / sz[1]
            interp2_fs = np.real(cubic_spline_fourier(f2, config.interp_bicubic_a) / sz[1])
        else:
            raise("Unknow dft interpolation method")
        if config.interp_centering:
            f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float32)[:, np.newaxis]
            interp1_fs = interp1_fs * np.exp(-1j*np.pi / sz[0] * f1)
            f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float32)[np.newaxis, :]
            interp2_fs = interp2_fs * np.exp(-1j*np.pi / sz[1] * f2)

        if config.interp_windowing:
            win1 = signal.hann(sz[0]+2)[:, np.newaxis]
            win2 = signal.hann(sz[1]+2)[np.newaxis, :]
            interp1_fs = interp1_fs * win1[1:-2]
            interp2_fs = interp2_fs * win2[1:-2]
        return interp1_fs, interp2_fs

    def _get_reg_filter(self, sz, target_sz, reg_window_edge):
        if config.use_reg_window:

            # normalization factor
            reg_scale = 0.5 * target_sz

            # construct grid
            wrg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wcg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wrs, wcs = np.meshgrid(wrg, wcg)

            # construct the regularization window
            reg_window = (reg_window_edge - config.reg_window_min) * (np.abs(wrs/reg_scale[0])**config.reg_window_power + \
                            np.abs(wcs/reg_scale[1])**config.reg_window_power) + config.reg_window_min

            # compute the DFT and enforce sparsity
            reg_window_dft = fft2(reg_window) / np.prod(sz)
            reg_window_dft[np.abs(reg_window_dft) < config.reg_sparsity_threshold* np.max(np.abs(reg_window_dft.flatten()))] = 0

            # do the inverse transform, correct window minimum
            reg_window_sparse = np.real(np.fft.ifft2(reg_window_dft))
            reg_window_dft[0, 0] = reg_window_dft[0, 0] - np.prod(sz) * np.min(reg_window_sparse.flatten()) + config.reg_window_min
            reg_window_dft = fftshift(reg_window_dft)

            # find the regularization filter by removing the zeros
            row_idx = np.logical_not(np.all(reg_window_dft==0, axis=1))
            col_idx = np.logical_not(np.all(reg_window_dft==0, axis=0))
            mask = np.outer(row_idx, col_idx)
            reg_filter = np.real(reg_window_dft[mask]).reshape(np.sum(row_idx), -1)
        else:
            # else use a scaled identity matrix
            reg_filter = config.reg_window_min
        return reg_filter.T

    def init(self, frame, bbox, total_frame=np.Inf):
        """
            frame -- need rgb image
            bbox -- need xmin, ymin, height, width
        """
        self._pos = np.array([bbox[1]+(bbox[3]-1)/2., bbox[0]+(bbox[2]-1)/2.], dtype=np.float32)
        self._target_sz = np.array([bbox[3], bbox[2]]) # (width, height)
        # self._features = config.features
        self._num_samples = min(config.num_samples, total_frame)

        # calculate search area and initial scale factor
        search_area = np.prod(self._target_sz * config.search_area_scale)
        if search_area > config.max_image_sample_size:
            self._current_scale_factor = np.sqrt(search_area / config.max_image_sample_size)
        elif search_area < config.min_image_sample_size:
            self._current_scale_factor = np.sqrt(search_area / config.min_image_sample_size)
        else:
            self._current_scale_factor = 1.

        # target size at the initial scale
        self._base_target_sz = self._target_sz / self._current_scale_factor

        # target size, taking padding into account
        if config.search_area_shape == 'proportional':
            self._img_sample_sz = np.floor(self._base_target_sz * config.search_area_scale)
        elif config.search_area_shape == 'square':
            self._img_sample_sz = np.ones((2), dtype=np.float32) * np.sqrt(np.prod(self._base_target_sz * config.search_area_scale))
        else:
            raise("unimplemented")

        features = [feature for feature in config.features
                if ("use_for_color" in feature and feature["use_for_color"] == self._is_color) or
                    "use_for_color" not in feature]

        self._features = []
        cnn_feature_idx = -1
        for idx, feature in enumerate(features):
            if feature['fname'] == 'cn' or feature['fname'] == 'ic':
                self._features.append(TableFeature(**feature))
            elif feature['fname'] == 'fhog':
                self._features.append(FHogFeature(**feature))
            elif feature['fname'] == 'cnn':
                has_cnn_feature = idx
                pass
            else:
                raise("unimplemented features")
        self._features = sorted(self._features, key=lambda x:x.min_cell_size)
        # print(self._features)

        if config.use_projection_matrix:
            self._sample_dim = [ feature._compressed_dim for feature in self._features ]
        else:
            self._sample_dim = [ feature.num_dim for feature in self._features ]

        # calculate image sample size
        # max_cell_size = max([feature.min_cell_size for feature in self._features])
        if cnn_feature_idx >= 0:
            self._img_sample_sz = self._features[cnn_feature_idx].init_size(self._img_sample_sz)
        else:
            self._img_sample_sz = self._features[0].init_size(self._img_sample_sz)

        for feature in self._features:
            feature.init_size(self._img_sample_sz)

        self._feature_dim = [ feature.num_dim for feature in self._features]

        self._feature_sz = np.array([feature.data_sz for feature in self._features], dtype=np.int32)

        # number of fourier coefficients to save for each filter layer, this will be an odd number
        filter_sz = self._feature_sz + (self._feature_sz + 1) % 2

        # the size of the label function DFT. equal to the maximum filter size
        self._k1 = np.argmax(filter_sz, axis=0)[0]
        self._output_sz = filter_sz[self._k1]

        # get the remaining block indices
        self._block_inds = list(range(len(self._features)))
        self._block_inds.remove(self._k1)

        # how much each feature block has to be padded to the obtain output_sz
        self._pad_sz = [((self._output_sz - filter_sz_) / 2).astype(np.int32) for filter_sz_ in filter_sz]

        # compute the fourier series indices and their transposes
        self._ky = [np.arange(-np.ceil(sz[0]-1)/2, np.floor((sz[0]-1)/2)+1) for sz in filter_sz]
        self._kx = [np.arange(-np.ceil(sz[1]-1)/2, 1) for sz in filter_sz]

        # construct the gaussian label function using poisson formula
        sig_y = np.sqrt(np.prod(np.floor(self._base_target_sz))) * config.output_sigma_factor * (self._output_sz / self._img_sample_sz)
        yf_y = [np.sqrt(2 * np.pi) * sig_y[0] / self._output_sz[0] * np.exp(-2 * (np.pi * sig_y[0] * ky_ / self._output_sz[0])**2)
                    for ky_ in self._ky]
        yf_x = [np.sqrt(2 * np.pi) * sig_y[1] / self._output_sz[1] * np.exp(-2 * (np.pi * sig_y[1] * kx_ / self._output_sz[1])**2)
                    for kx_ in self._kx]
        self._yf = [yf_y_.reshape(-1, 1) * yf_x_ for yf_y_, yf_x_ in zip(yf_y, yf_x)]

        # construct cosine window
        self._cos_window = [self._cosine_window(feature_sz_)[:, :, np.newaxis]
                for feature_sz_ in self._feature_sz]

        # compute fourier series of interpolation function
        self._interp1_fs = []
        self._interp2_fs = []
        for sz in filter_sz:
            interp1_fs, interp2_fs = self._get_interp_fourier(sz)
            self._interp1_fs.append(interp1_fs)
            self._interp2_fs.append(interp2_fs)

        # get the reg_window_edge parameter
        reg_window_edge = []
        for feature in self._features:
            if hasattr(feature, 'reg_window_edge'):
                reg_window_edge.append(feature.reg_window_edge)
            else:
                reg_window_edge.append(config.reg_window_edge * np.ones((len(feature.num_dim)), dtype=np.float32))

        # construct spatial regularization filter
        self._reg_filter = [self._get_reg_filter(self._img_sample_sz, self._base_target_sz, reg_window_edge_)
                                for reg_window_edge_ in reg_window_edge]

        # compute the energy of the filter (used for preconditioner)
        self._reg_energy = [np.real(reg_filter.flatten().dot(reg_filter.flatten()))
                        for reg_filter in self._reg_filter]

        if config.use_scale_filter:
            self._scale_filter = ScaleFilter(self._target_sz)
            self._num_scales = self._scale_filter.num_scales
            self._scale_step = self._scale_filter.scale_step
            self._scale_factor = self._scale_filter.scale_factors
        else:
            # use the translation filter to estimate the scale
            self._num_scales = config.number_of_scales
            self._scale_step = config.scale_step
            self._scale_exp = np.arange(-np.floor(self._num_scales-1)/2, np.ceil(self._num_scales-1)/2)
            self._scale_factor = self._scale_step**self._scale_exp

        if self._num_scales > 0:
            # force reasonable scale changes
            self._min_scale_factor = self._scale_step ** np.ceil(np.log(np.max(5 / self._img_sample_sz)) / np.log(self._scale_step))
            self._max_scale_factor = self._scale_step ** np.floor(np.log(np.min(frame.shape[:2] / self._base_target_sz)) / np.log(self._scale_step))

        # set conjugate gradient options
        self._init_CG_opts = {'CG_use_FR': True,
                         'tol': 1e-6,
                         'CG_standard_alpha': True
                        }
        self._CG_opts = {'CG_use_FR': self._init_CG_opts['CG_use_FR'],
                         'tol': 1e-6,
                         'CG_standard_alpha': self._init_CG_opts['CG_standard_alpha']
                        }
        if config.CG_forgetting_rate == np.Inf or config.learning_rate >= 1:
            self._CG_opts['init_forget_factor'] = 0.
        else:
            self._CG_opts['init_forget_factor'] = (1 - config.learning_rate) ** config.CG_forgetting_rate

        # init ana allocate
        self._prior_weights = np.zeros((config.num_samples, 1), dtype=np.float32)
        self._sample_weights = np.zeros_like(self._prior_weights)
        self._samplesf = [[]] * len(self._features)

        for i in range(len(self._features)):
            self._samplesf[i] = np.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2), self._sample_dim[i], config.num_samples), dtype=np.complex128)

        # allocate
        self._scores_fs_feat = [[]] * len(self._features)

        # distance matrix stores the square of the euclidean distance between each pair of samples
        self._distance_matrix = np.ones((self._num_samples, self._num_samples)) * np.Inf

        # kernale matrix, used to udpate distance matrix
        self._gram_matrix = np.ones((self._num_samples, self._num_samples)) * np.Inf

        self._latest_ind = []
        self._frames_since_last_train = np.inf
        self._num_training_samples = 0

        # find the minimum allowed sample weight. samples are discarded if their weights become lower
        config.minimum_sample_weight = config.learning_rate * (1 - config.learning_rate)**(2*config.num_samples)

        # extract sample and init projection matrix
        sample_pos = mround(self._pos)
        sample_scale = self._current_scale_factor * self._scale_factor
        xl = [feature.get_features(frame, sample_pos, self._img_sample_sz, self._current_scale_factor) for feature in self._features]  # get features
        xlw = [x * y for x, y in zip(xl, self._cos_window)]                                                                            # do windowing
        xlf = [cfft2(x) for x in xlw]                                                                                                  # fourier series
        xlf = interpolate_dft(xlf, self._interp1_fs, self._interp2_fs)                                                                 # interpolate features
        xlf = compact_fourier_coeff(xlf)                                                                                               # new sample to be added
        shift_sample_ = 2 * np.pi * (self._pos - sample_pos) / (sample_scale * self._img_sample_sz)
        xlf = shift_sample(xlf, shift_sample_, self._kx, self._ky)
        self._proj_matrix = self._init_proj_matrix(xl, self._sample_dim, config.proj_init_method)
        # import scipy.io as sio
        # data = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/data.mat")
        # self._proj_matrix = [data['projection_matrix'][0][0][0],
        #                      data['projection_matrix'][0][0][1]]
        xlf_proj = self._proj_sample(xlf, self._proj_matrix)

        merged_sample, new_sample, merged_sample_id, new_sample_id, self._distance_matrix, self._gram_matrix, \
                self._prior_weights = update_sample_space_model(self._samplesf,
                                                                xlf_proj,
                                                                self._distance_matrix,
                                                                self._gram_matrix,
                                                                self._prior_weights,
                                                                self._num_training_samples)
        if self._num_training_samples < config.num_samples:
            self._num_training_samples += 1

        if config.update_projection_matrix:
            for i in range(len(self._features)):
                if merged_sample_id > 0:
                    self._samplesf[i][:, :, :, merged_sample_id] = merged_sample[i]
                if new_sample_id > 0:
                    self._samplesf[i][:, :, :, new_sample_id] = new_sample[i]

        # train_tracker
        new_sample_energy = [np.abs(xlf * np.conj(xlf)) for xlf in xlf_proj]
        self._sample_energy = new_sample_energy

        # init conjugate gradient param
        sample_energy = new_sample_energy
        self._CG_state = None
        if config.update_projection_matrix:
            self._init_CG_opts['maxit'] = np.ceil(config.init_CG_iter / config.init_GN_iter)
            self._hf = [[[]] * len(self._features) for _ in range(2)]
            self._proj_energy = [2*np.sum(np.abs(yf_.flatten()**2) / np.sum(self._feature_dim)) * np.ones_like(P) for P, yf_ in zip(self._proj_matrix, self._yf)]
        else:
            self._CG_opts['maxit'] = config.init_CG_iter
            self._hf = [[[]] * len(self._features)]

        # init the filter with zeros
        for i in range(len(self._features)):
            self._hf[0][i] = np.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2), int(self._sample_dim[i])), dtype=np.complex128)

        if config.update_projection_matrix:
            # init gauss-newton optimization of the filter and projection matrix
            self._hf, self._proj_matrix, self._res_norms = train_joint(self._hf,
                                                                       self._proj_matrix,
                                                                       xlf,
                                                                       self._yf,
                                                                       self._reg_filter,
                                                                       self._sample_energy,
                                                                       self._reg_energy,
                                                                       self._proj_energy,
                                                                       self._init_CG_opts)

            # re-project and insert training sample
            xlf_proj = self._proj_sample(xlf, self._proj_matrix)
            for i in range(len(self._features)):
                self._samplesf[i][:, :, :, 0] = xlf_proj[i]

            # udpate the gram matrix since the sample has changed
            if config.distance_matrix_update_type == 'exact':
                # find the norm of the reprojected sample
                new_train_sample_norm = 0.
                for i in range(len(self._features)):
                    new_train_sample_norm += np.real(2 * xlf_proj[i].flatten().dot(xlf_proj[i].flatten()))
                self._gram_matrix[0, 0] = new_train_sample_norm
        self._hf_full = full_fourier_coeff(self._hf)

        if config.use_scale_filter:
            self._scale_filter.update(frame, self._pos, self._base_target_sz, self._current_scale_factor)
        self._frame_num += 1

    def _init_proj_matrix(self, init_sample, compressed_dim, proj_method):
        x = [np.reshape(x, (-1, x.shape[2]), order='F') for x in init_sample]
        x = [z - z.mean(0) for z in x]
        proj_matrix_ = []
        for x_, compressed_dim_  in zip(x, compressed_dim):
            if proj_method == 'pca':
                proj_matrix, _, _ = np.linalg.svd(x_.T.dot(x_))
                proj_matrix = proj_matrix[:, :compressed_dim_]
            elif proj_method == 'rand_uni':
                proj_matrix = np.random.randn(x[1], compressed_dim_)
                proj_matrix = proj_matrix / np.sqrt(np.sum(proj_matrix**2, 0))
            elif proj_method == 'none':
                proj_matrix = []
            else:
                raise("Unknow initialization method for the projection matrix")
            proj_matrix_.append(proj_matrix)
        return proj_matrix_

    def _proj_sample(self, x, P):
        if len(x[0].shape) == 3:
            x = [x_[:, :, :, np.newaxis].transpose(3, 2, 0, 1) for x_ in x]
        elif len(x[1].shape) == 4:
            x = [x_.transpose(3, 2, 0, 1) for x_ in x]
        x = [np.matmul(P_.T, x_.reshape(x_.shape[0], x_.shape[1], -1)).reshape(x_.shape[0], -1, x_.shape[2], x_.shape[3])
                for x_, P_ in zip(x, P)]
        x = [x_.transpose(2, 3, 1, 0).squeeze() for x_ in x]
        return x

    # target localization
    # def localization

    def update(self, frame, train=True):
        # target localization step
        pos = self._pos
        old_pos = np.zeros((2))
        # import scipy.io as sio
        # data = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/update.mat")
        # self._hf_full = data['hf_full'][0][0]
        # self._proj_matrix = data['projection_matrix'][0][0]
        for _ in range(config.refinement_iterations):
            if np.any(old_pos != pos):
                old = pos
                # extract fatures at multiple resolutions
                sample_pos = mround(pos)
                det_sample_pos = sample_pos
                sample_scale = self._current_scale_factor * self._scale_factor
                xt = [feature.get_features(frame, sample_pos, self._img_sample_sz, sample_scale)
                        for feature in self._features]                                          # get features
                xt_proj = self._proj_sample(xt, self._proj_matrix)                              # project sample
                xt_proj = [feat_map_ * cos_window_
                        for feat_map_, cos_window_ in zip(xt_proj, self._cos_window)]           # do windowing
                xtf_proj = [cfft2(x) for x in xt_proj]                                          # compute the fourier series
                xtf_proj = interpolate_dft(xtf_proj, self._interp1_fs, self._interp2_fs)        # interpolate features to cont

                # compute convolution for each feature block in the fourier domain, then sum over blocks
                self._scores_fs_feat[self._k1] = np.sum(self._hf_full[self._k1] * xtf_proj[self._k1], 2)
                scores_fs = self._scores_fs_feat[self._k1]
                # scores_fs_sum shape: height x width x num_scale
                for i in self._block_inds:
                    self._scores_fs_feat[i] = np.sum(self._hf_full[i]*xtf_proj[i], 2)
                    scores_fs[self._pad_sz[i][0]:-self._pad_sz[i][0],
                              self._pad_sz[i][1]:-self._pad_sz[i][1]] += self._scores_fs_feat[i]

                # optimize the continuous score function with newton's method.
                # import scipy.io as sio
                # data = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/opt.mat")
                trans_row, trans_col, scale_idx = optimize_score(scores_fs, config.newton_iterations)
                # trans_row, trans_col, scale_idx = optimize_score(data['scores_fs'], config.newton_iterations)

                # compute the translation vector in pixel-coordinates and round to the cloest integer pixel
                translation_vec = np.array([trans_row, trans_col]) * (self._img_sample_sz / self._output_sz) * \
                                    self._current_scale_factor * self._scale_factor[scale_idx]
                scale_change_factor = self._scale_factor[scale_idx]

                # udpate position
                old_pos = pos
                pos = sample_pos + translation_vec

                if config.clamp_position:
                    pos = np.maximum(np.array(0, 0), np.minimum(np.array(frame.shape[:2]), pos))

                # do scale tracking with scale filter
                if self._num_scales > 0 and config.use_scale_filter:
                    scale_change_factor = self._scale_filter.track(frame, pos, self._base_target_sz,
                           self._current_scale_factor)
                # udpate the scale
                self._current_scale_factor = self._current_scale_factor * scale_change_factor

                # adjust to make sure we are not to large or to small
                if self._current_scale_factor < self._min_scale_factor:
                    self._current_scale_factor = self._min_scale_factor
                elif self._current_scale_factor > self._max_scale_factor:
                    self._current_scale_factor = self._max_scale_factor

        # model udpate step
        if config.learning_rate > 0:
            # use the sample that was used for detection
            sample_scale = sample_scale[scale_idx]
            # xlf_proj = [ xf[:, :(xf.shape[1]+1)//2, :, scale_idx] for xf in xtf_proj ]
            xlf_proj = [ xf[:, :(xf.shape[1]+1)//2, :] for xf in xtf_proj ]

            # shift the sample so that the target is centered
            shift_sample_ = 2 * np.pi * (pos - sample_pos) / (sample_scale * self._img_sample_sz)
            xlf_proj = shift_sample(xlf_proj, shift_sample_, self._kx, self._ky)

        # update the samplesf to include the new sample. The distance matrix, kernel matrix and prior 
        # weight are also updated
        merged_sample, new_sample, merged_sample_id, new_sample_id, self._distance_matrix, self._gram_matrix, \
                self._prior_weights = update_sample_space_model(self._samplesf,
                                                                xlf_proj,
                                                                self._distance_matrix,
                                                                self._gram_matrix,
                                                                self._prior_weights,
                                                                self._num_training_samples)
        if self._num_training_samples < self._num_samples:
            self._num_training_samples += 1
        if config.learning_rate > 0:
            for i in range(len(self._features)):
                if merged_sample_id > 0:
                    self._samplesf[i][:, :, :, merged_sample_id] = merged_sample[i]
                if new_sample_id > 0:
                    self._samplesf[i][:, :, :, merged_sample_id] = new_sample[i]
        self._sample_weights = self._prior_weights

        # training filter
        if self._frame_num < config.skip_after_frame or \
                self._frames_since_last_train >= config.train_gap:
            new_sample_energy = [np.real(xlf * np.conj(xlf)) for xlf in xlf_proj]
            self._CG_opts['maxit'] = config.CG_iter
            self._sample_energy = [(1 - config.learning_rate)*se + config.learning_rate*nse
                                for se, nse in zip(self._sample_energy, new_sample_energy)]

            # do conjugate gradient optimization of the filter
            # import scipy.io as sio
            # data = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/train_filter.mat")
            self._hf, self._res_norms, self._CG_state = train_filter(
                                                         self._hf,
                                                         self._samplesf,
                                                         self._yf,
                                                         self._reg_filter,
                                                         self._sample_weights,
                                                         self._sample_energy,
                                                         self._reg_energy,
                                                         self._CG_opts,
                                                         self._CG_state)
            self._hf_full = full_fourier_coeff(self._hf)
            self._frames_since_last_train = 0
        else:
            self._frames_since_last_train += 1

        if config.use_scale_filter:
            self._scale_filter.update(frame, self._pos, self._base_target_sz, self._current_scale_factor)

        # udpate the target size
        self._target_sz = self._base_target_sz * self._current_scale_factor

        # save position and calculate fps
        bbox = np.array([pos[0], pos[1], self._target_sz[0], self._target_sz[1]])
        self._pos = pos
        pdb.set_trace()
        self._frame_num += 1
        # TODO visualization tracking results and intermediate response
        return bbox
