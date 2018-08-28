import numpy as np
import scipy.io as sio

from eco import train_filter, lhs_operation, lhs_operation_joint, train_joint, train_filter, preconditioned_conjugate_gradient

import ipdb as pdb

if __name__ == '__main__':
    data_input = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/joint.mat")
    data_output = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/joint_ret.mat")
    hf = [[data_input['hf'][0,0,0][:,:,:,np.newaxis],data_input['hf'][0,0,1][:,:,:,np.newaxis]],
          [[], []]]
    proj_matrix = data_input['projection_matrix'][0][0]
    xlf = [x[:,:,:,np.newaxis] for x in data_input['xlf'][0][0]]
    yf = data_input['yf'][0][0]
    reg_filter = data_input['reg_filter'][0][0]
    sample_energy = [x[:,:,:,np.newaxis] for x in data_input['sample_energy'][0][0]]
    reg_energy = data_input['reg_energy'][0][0]
    proj_energy = data_input['proj_energy'][0][0]

    init_CG_opts = {'CG_use_FR': True,
                    'tol': 1e-6,
                    'CG_standard_alpha': True,
                    'maxit': 10}

    hf_ret, proj_matrix_ret, _  = train_joint(hf, proj_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, init_CG_opts)

    hf_ = [x[:,:,:,np.newaxis] for x in data_output['hf'][0][0]]
    proj_matrix = [x for x in data_output['projection_matrix'][0][0]]
    pdb.set_trace()

    assert np.allclose(hf_ret[0], hf_[0])
    assert np.allclose(hf_ret[1], hf_[1])
    assert np.allclose(proj_matrix_ret[0], proj_matrix[0])
    assert np.allclose(proj_matrix_ret[1], proj_matrix[1])
