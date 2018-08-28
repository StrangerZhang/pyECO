import scipy.io as sio
import numpy as np
from eco import optimize_score

import ipdb as pdb

if __name__ == '__main__':
    data = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/score_fs.mat")['scores_fs_data']
    for i in range(data[0][0].shape[3]):
        trans_row, trans_col, scale_idx = optimize_score(data[0][0][:,:,:,i], 5)
        assert np.allclose(trans_row, data[1][0][i, 0], 1e-5, 1e-5) and \
                np.allclose(trans_col, data[2][0][i, 0], 1e-5, 1e-5) and \
                np.allclose(scale_idx, data[3][0][i, 0]-1)

