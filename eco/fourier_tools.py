import numpy as np
from numpy.fft import fftshift, fft, ifft, ifftshift
import ipdb as pdb

np.seterr(divide='ignore', invalid='ignore')

def fft2(x):
    return fft(fft(x, axis=1), axis=0)

def ifft2(x):
    return ifft(ifft(x, axis=1), axis=0)

def cfft2(x):
    in_shape = x.shape
    # if both dimensions are odd
    if in_shape[0] % 2 == 1 and in_shape[1] % 2 == 1:
        xf = fftshift(fftshift(fft2(x), 0), 1)
    else:
        out_shape = (in_shape[0] + (in_shape[0] + 1) % 2, \
                     in_shape[1] + (in_shape[1] + 1) % 2)
        # xf =  np.zeros(out_shape, dtype=np.dtype)
        xf = np.complex(np.zeros(out_shape, dtype=complex))
        xf[0:out_shape[0], 0:out_shape[1], :, :] = fftshift(fftshift(fft2(x), 0), 1)
        if out_shape[0] != in_shape[0]:
            xf[-1,:,:,:] = xf[0,::-1,:,:].conjugate()
        if out_shape[1] != in_shape[1]:
            xf[:,-1,:,:] = xf[::-1,0,:,:].conjugate()
    return xf

def cifft2(xf):
    x = np.real(ifft2(ifftshift(ifftshift(xf, 0),1)))
    return x

def compact_fourier_coeff(xf):
    if isinstance(xf, list):
        return [x[:, :(x.shape[1]+1)//2, :] for x in xf]
    else:
        mid = (xf.shape[1] + 1) // 2
        return xf[:, :mid, :]
    # return xf

def cubic_spline_fourier(f, a):
    # The continuous fourier transform of a cubic spline kernel
    bf = - ( - 12 * a + 12 * np.exp( - np.pi * f * 2j) + 12 * np.exp(np.pi * f * 2j) + 6 * a * np.exp(-np.pi * f * 4j) + \
        6 * a * np.exp(np.pi * f * 4j) + f * (np.pi * np.exp(-np.pi*f*2j)*12j) - f * (np.pi * np.exp(np.pi * f * 2j) * 12j) + \
        a*f*(np.pi*np.exp(-np.pi*f*2j)*16j) - a * f * (np.pi*np.exp(np.pi*f*2j)*16j) + \
        a*f*(np.pi*np.exp(-np.pi*f*4j)*4j) - a * f * (np.pi*np.exp(np.pi*f*4j)*4j)-24)
    # bf[f == 0] = 1
    # bf = np.real(bf)
    bf /= (16 * f**4 * np.pi**4)
    bf[f == 0] = 1
    return bf

def full_fourier_coeff(xf):
    # Reconstructs the full Fourier series coefficients
    xf = [np.concatenate([xf_, np.rot90(xf_[:, :-1,:], 2)], axis=1) for xf_ in xf]
    return xf

def interpolate_dft(xf, interp1_fs, interp2_fs):
    return [xf_ * interp1_fs_[:, :, np.newaxis] * interp2_fs_[:, :, np.newaxis] for xf_, interp1_fs_, interp2_fs_ in zip(xf, interp1_fs, interp2_fs)]


def resize_dft(inputdft, desired_len):
    # resize a one-dimensional DFT to the desired length.
    input_len = len(inputdft)
    minsz = min(input_len, desired_len)

    scaling = desired_len / input_len

    # if inputdft.shape[0] > 1:
    #     new_size = (desired_len, 1)
    # else:
    #     new_size = (1, desired_len)

    resize_dft = np.zeros(desired_len, dtype=inputdft.dtype)

    mids = int(np.ceil(minsz / 2))
    mide = int(np.floor((minsz - 1) / 2) - 1)

    resize_dft[:mids] = scaling * inputdft[:mids]
    resize_dft[-mide:] = scaling * inputdft[-mide:]
    return resize_dft

def sample_fs(xf, grid_sz=None):
    sz = xf.shape[:2]
    if grid_sz is None or sz == grid_sz:
        x = np.prod(sz) * cifft2(xf)
    else:
        sz = np.array(sz)
        grid_sz = np.array(grid_sz)
        if np.any(grid_sz < sz):
            raise("The grid size must be larger than or equal to the siganl size")
        tot_pad = grid_sz - sz
        pad_sz = np.ceil(tot_pad / 2)
        xf_pad = np.pad(xf, pad_sz)
        if np.any(tot_pad % 2 == 1):
            xf_pad = xf_pad[:-tot_pad[0] % 2, :-tot_pad[1] % 2]
        x = np.prod(grid_sz) * cifft2(xf_pad)
    return x

def shift_sample(xf, shift, kx, ky):
    shift_exp_y = [np.exp(1j * shift[0] * ky_) for ky_ in ky]
    shift_exp_x = [np.exp(1j * shift[1] * kx_) for kx_ in kx]
    xf = [xf_ * sy_.reshape(-1, 1, 1) * sx_.reshape((1, -1, 1)) for xf_, sy_, sx_ in zip(xf, shift_exp_y, shift_exp_x)]
    return xf

def symmetrize_filter(hf):
    # ensure hermetian symmetry
    for i in range(len(hf)):
        dc_ind = int((hf[i].shape[0]+1) / 2)
        hf[i][dc_ind:, -1, :] = np.conj(np.flipud(hf[i][:dc_ind-1, -1, :]))
    return hf
