from eco import GMM
import scipy.io as sio
import numpy as np

import ipdb as pdb

if __name__ == '__main__':
    data = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/sample_space_model.mat")['sample_space_model_data']
    gmm = GMM(30)
    samplesf = data[0][0]
    xlf_proj = data[1][0]
    num_training_samples = data[2][0].squeeze()
    merged_sample = data[3][0]
    new_sample = data[4][0]
    merged_sample_id = data[5][0].squeeze()
    new_sample_id = data[6][0].squeeze()
    merged_sample_idx = 0
    new_sample_idx = 0
    seq_len = samplesf.shape[2]
    for i in range(seq_len):
        samplesf_ = [x.transpose(2, 3, 1, 0) for x in samplesf[i, 0, :]]
        xlf_proj_ = [x.transpose(2, 3, 1, 0) for x in xlf_proj[i, 0, :]]
        merged_sample_ret, new_sample_ret, merged_sample_id_ret, new_sample_id_ret = gmm.update_sample_space_model(samplesf_, xlf_proj_, num_training_samples[i])
        if merged_sample_id_ret == -1:
            assert merged_sample_id_ret == merged_sample_id[i]
        else:
            assert merged_sample_id_ret == merged_sample_id[i] - 1
            merged_sample_ = [x.transpose(2, 3, 1, 0) for x in merged_sample[merged_sample_idx, 0, :]]
            merged_sample_idx += 1
            assert np.allclose(merged_sample_ret[0], merged_sample_[0])
            assert np.allclose(merged_sample_ret[1], merged_sample_[1])
        if new_sample_id_ret == -1:
            assert new_sample_id_ret == new_sample_id[i]
        else:
            assert new_sample_id_ret == new_sample_id[i] - 1
            new_sample_ = [x.transpose(2, 3, 1, 0) for x in new_sample[new_sample_idx, 0, :]]
            new_sample_idx += 1
            assert np.allclose(new_sample_ret[0], new_sample_[0])
            assert np.allclose(new_sample_ret[1], new_sample_[1])

