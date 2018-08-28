import numpy as np
import scipy.io as sio

from eco import train_filter, lhs_operation, lhs_operation_joint, train_joint, train_filter, preconditioned_conjugate_gradient, inner_product_joint, diag_precond
from eco import config, symmetrize_filter

import ipdb as pdb


if __name__ == '__main__':
    data_input = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/pcg_ccot.mat")

    init_samplef_proj = [x.transpose(2, 3, 1, 0) for x in data_input['init_samplef_proj'][0][0]]
    init_hf = [x.transpose(2, 3, 0, 1) for x in data_input['init_hf'][0][0]]
    rhs_samplef = [[x[:,:,:,np.newaxis] for x in data_input['rhs_samplef'][0][0]],
                   [x for x in data_input['rhs_samplef'][1][0]]]
    fyf = data_input['fyf'][0][0]
    hf = data_input['hf']
    hf = [[hf[0][0][0][:,:,:,np.newaxis], hf[0][0][1][:,:,:,np.newaxis]],
          [hf[1][0][0], hf[1][0][1]]]
    init_samplef = [x.transpose(2, 3, 1, 0) for x in data_input['init_samplef'][0][0]]
    init_samplef_H = data_input['init_samplef_H'][0][0]
    reg_filter = data_input['reg_filter'][0][0]
    projection_matrix = data_input['projection_matrix'][0][0]
    diag_M = data_input['diag_M']
    diag_M = [[diag_M[0][0][0][:,:,:,np.newaxis], diag_M[0][0][1][:,:,:,np.newaxis]],
              [diag_M[1][0][0], diag_M[1][0][1]]]

    init_CG_opts = {'CG_use_FR': True,
                    'tol': 1e-6,
                    'CG_standard_alpha': True}
    init_CG_opts['maxit'] = np.ceil(config.init_CG_iter / config.init_GN_iter)
    hf_, res_norms_temp, _ = preconditioned_conjugate_gradient(
            lambda x: lhs_operation_joint(x, init_samplef_proj, reg_filter, init_samplef, init_samplef_H, init_hf, config.projection_reg), # A
            rhs_samplef,                                                                                                                   # b
            init_CG_opts,                                                                                                                  # opts
            lambda x: diag_precond(x, diag_M),                                                                                             # M1
            None,                                                                                                                          # M2
            inner_product_joint,
            hf)
    proj_matrix = [x + y for x, y in zip(projection_matrix, hf_[1])]
    hf_ = symmetrize_filter(hf_[0])
    pdb.set_trace()


    data_output = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/pcg_ccot_ret.mat")
    hf_gt = [x[:,:,:,np.newaxis] for x in data_output['hf'][0][0]]
    proj_matrix_gt = data_output['projection_matrix'][0][0]
    assert np.allclose(hf_[0], hf_gt[0])
    assert np.allclose(hf_[1], hf_gt[1])
    assert np.allclose(proj_matrix[0], proj_matrix_gt[0])
    assert np.allclose(proj_matrix[1], proj_matrix_gt[1])

    # data_output = sio.loadmat("/Users/fyzhang/Desktop/codes/vot/ECO/lhs_joint.mat")['data']
    # x = data_output[0, 0]
    # Ax = data_output[0, 1]
    # seq_len = x.shape[3]
    # for i in range(seq_len):
    #     x_ = x[:,:,:,i]
    #     x_ = [[x_[0][0][0][:,:,:,np.newaxis], x_[0][0][1][:,:,:,np.newaxis]],
    #           [x_[1][0][0], x_[1][0][1]]]
    #     Ax_ret = lhs_operation_joint(x_, init_samplef_proj, reg_filter, init_samplef, init_samplef_H, init_hf, config.projection_reg)
    #     Ax_ = Ax[:,:,:,i]
    #     Ax_ = [[Ax_[0][0][0][:,:,:,np.newaxis], Ax_[0][0][1][:,:,:,np.newaxis]],
    #            [Ax_[1][0][0], Ax_[1][0][1]]]
    #     assert np.allclose(Ax_ret[0][0], Ax_[0][0], 1e-5, 1e-5)
    #     assert np.allclose(Ax_ret[0][1], Ax_[0][1], 1e-5, 1e-5)
    #     assert np.allclose(Ax_ret[1][0], Ax_[1][0], 1e-5, 1e-5)
    #     assert np.allclose(Ax_ret[1][1], Ax_[1][1], 1e-5, 1e-5)
