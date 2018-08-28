import scipy.io as sio
from eco import ScaleFilter
import ipdb as pdb
import numpy as np

if __name__ == '__main__':
    data = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/scale_filter.mat")['scale_filter_data'][0]
    scale_filter = ScaleFilter(data[0].squeeze())
    pdb.set_trace()
    for i in range(0, data[1].shape[3]-1, 2):
        scale_filter.update(data[1][:, :, :, i], data[2][i], data[3][i], data[4][i])
        scale_change_factor = scale_filter.track(data[1][: ,:, :, i+1], data[2][i+1], data[3][i+1], data[4][i+1])
        print(scale_change_factor, data[5][i//2])
        # assert np.allclose(scale_change_factor, data[5][i//2]), "iter: {}".format(i)
