import tqdm
import numpy as np
from scipy.signal import convolve2d
from functools import reduce
from timeit import default_timer as timer


def uiqi(im1, im2, image_mask, block_size=64, return_map=False):
    if len(im1.shape)==3:
        return np.array([uiqi(im1[:,:,i], im2[:,:,i], block_size, return_map=return_map) for i in range(im1.shape[2])])

    total_q = []
    '''for i in tqdm.tqdm(range(im1.shape[0] - block_size)):
        for j in range(im1.shape[1] - block_size):
            cur_x = im1[i: i + block_size, j: j + block_size]
            cur_y = im2[i: i + block_size, j: j + block_size]
            delta_x = np.std(cur_x, ddof=1)
            delta_y = np.std(cur_y, ddof=1)
            delta_xy = np.sum((cur_x - np.mean(cur_x)) * (cur_y - np.mean(cur_y))) / (block_size ** 2 - 1)
            mu_x = np.mean(cur_x)
            mu_y = np.mean(cur_y)
            q1 = min(delta_xy / (delta_x * delta_y + 1e-100), 1)
            q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2 + 1e-100)
            q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2 + 1e-100)
            cur_q = q1 * q2 * q3
            if cur_q > 1 or cur_q < 0:
                print(cur_q)
                print(q1, q2, q3)
                print(delta_xy, delta_x, delta_y)
                exit()
            total_q.append(cur_q)'''

    '''t_s = timer()
    total_x = []
    total_y = []
    total_mask = []
    for i in range(im1.shape[0] - block_size):
        for j in range(im1.shape[1] - block_size):
            cur_x = im1[i: i + block_size, j: j + block_size]
            cur_y = im2[i: i + block_size, j: j + block_size]
            cur_mask = image_mask[i: i + block_size, j: j + block_size]
            total_x.append(cur_x)
            total_y.append(cur_y)
            total_mask.append(cur_mask)

    total_x = np.array(total_x)
    total_y = np.array(total_y)
    total_mask = np.array(total_mask)
    delta_x = np.std(total_x, axis=(1, 2), ddof=1)
    delta_y = np.std(total_y, axis=(1, 2), ddof=1)
    delta_xy = np.sum((total_x - np.mean(total_x, axis=(1, 2), keepdims=True)) *
                      (total_y - np.mean(total_y, axis=(1, 2), keepdims=True)), axis=(1, 2)) / (block_size ** 2 - 1)
    mu_x = np.mean(total_x, axis=(1, 2))
    mu_y = np.mean(total_y, axis=(1, 2))

    q1 = delta_xy / (delta_x * delta_y + 1e-100)
    q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2 + 1e-100)
    q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2 + 1e-100)
    q = q1 * q2 * q3
    q[np.where(q > 1)] = 1
    q[np.where(q < -1)] = -1
    q_mask = np.ones(q.shape)

    for i in tqdm.tqdm(range(q.shape[0])):
        # if np.sum(total_mask[i]) <= 0.9 * (block_size ** 2):
        if np.where(total_mask[i] == 0)[0].shape[0] >= 0.1 * (block_size ** 2):
            q_mask[i] = 0
    q_value = np.sum(q * q_mask) / np.sum(q_mask)
    t_e = timer()'''
    delta_x = np.std(im1, ddof=1)
    delta_y = np.std(im2, ddof=1)
    delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / (im1.shape[0] * im1.shape[1] - 1)
    mu_x = np.mean(im1)
    mu_y = np.mean(im2)
    q1 = delta_xy / (delta_x * delta_y)
    q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2)
    q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2)
    q = q1 * q2 * q3
    return q


def partial_sums(x, kernel_size=8):
    """Calculate partial sums of array in boxes (kernel_size x kernel_size).
    This corresponds to:
    scipy.signal.convolve2d(x, np.ones((kernel_size, kernel_size)), mode='valid')
    >>> partial_sums(np.arange(12).reshape(3, 4), 2)
    array([[10, 14, 18],
           [26, 30, 34]])
    """
    assert len(x.shape) >= 2 and x.shape[0] >= kernel_size and x.shape[1] >= kernel_size
    sums = x.cumsum(axis=0).cumsum(axis=1)
    sums = np.pad(sums, 1)[:-1, :-1]
    return (
        sums[kernel_size:, kernel_size:]
        + sums[:-kernel_size, :-kernel_size]
        - sums[:-kernel_size, kernel_size:]
        - sums[kernel_size:, :-kernel_size]
    )


def universal_image_quality_index(x, y, kernel_size=8):
    """Compute the Universal Image Quality Index (UIQI) of x and y."""

    N = kernel_size ** 2

    x = x.astype(np.float)
    y = y.astype(np.float)
    e = np.finfo(np.float).eps

    # sums and auxiliary expressions based on sums
    S_x = partial_sums(x, kernel_size)
    S_y = partial_sums(y, kernel_size)
    PS_xy = S_x * S_y
    SSS_xy = S_x*S_x + S_y*S_y

    # sums of squares and product
    S_xx = partial_sums(x*x, kernel_size)
    S_yy = partial_sums(y*y, kernel_size)
    S_xy = partial_sums(x*y, kernel_size)

    num = 4 * PS_xy * (N * S_xy - PS_xy)
    den = (N*(S_xx + S_yy) - SSS_xy) / (SSS_xy + e)

    Q_s = (num) / (den + e)

    return np.mean(Q_s)

    return np.mean(Q_s)


def Measure_Quality(res_prev, res, QualMeasOpts):  # noqa: N803
    """
    :param res_prev: (np.ndarray)
        object being compared to
    :param res: (np.ndarray)
        true image
    :param QualMeasOpts: (str or listof(str))
        any of: 'RMSE', 'nRMSE' , 'CC', 'MSSIM', 'UQI', 'SSD'
    :return:
    """
    values = []
    # make it a list:
    for qual in QualMeasOpts:
        if "RMSE" == qual:
            N = np.prod(res_prev.shape)
            diff = res_prev - res
            values.append(np.sqrt(np.sum(diff ** 2) / N))
        if "nRMSE" == qual:
            N = reduce(lambda x, y: x * y, res_prev.shape)
            diff = res_prev - res
            values.append(
                (np.sqrt(np.sum(diff ** 2) / N) / ((0.00001 + np.sqrt(np.sum(res ** 2)) / N)))
            )
        if "CC" == qual:
            cc = np.corrcoef(res_prev.ravel(), res.ravel())
            values.append(cc[0, 1])

        if "MSSIM" == qual:
            N = np.prod(res_prev.shape)

            # Compute the mean pixel values of the two images

            mean_res_p = res_prev.mean()
            mean_res = res.mean()
            if mean_res == 0 and mean_res_p == 0:
                raise ValueError("Initialising with 0 matrix not valid")
            # Luminance Comparison

            K1 = 0.01  # K1 is a small constant <<1
            d = np.max(res_prev) - np.min(res_prev)  # dynamic range of the pixel values
            l = ((2 * mean_res * mean_res_p) + (K1 * d) ** 2) / (
                (mean_res_p ** 2) + (mean_res ** 2) + K1 * d ** 2
            )

            # Contrast comparison

            K2 = 0.02
            sres_p = res_prev.std()
            sres = res.std()

            c = ((2 * sres_p * sres) + (K2 * d) ** 2) / ((sres_p ** 2) + (sres ** 2) + K2 * d ** 2)

            # Structure comparison
            diffres_p = res_prev - mean_res_p
            diffres = res - mean_res
            delta = (1 / (N - 1)) * np.sum(diffres_p * diffres)
            s = (delta + (((K2 * d) ** 2)) / 2) / ((sres_p * sres) + ((K2 * d ** 2) / 2))

            values.append((1 / N) * l * c * s)
        if "UQI" == qual:
            N = np.prod(res_prev.shape)
            # Mean
            mean_res_p = np.mean(res_prev)
            mean_res = np.mean(res)

            # Variance
            varres_p = np.var(res_prev)
            varres = np.var(res)
            if mean_res == 0 and mean_res_p == 0:
                raise ValueError("Initialising with 0 matrix not valid")
            # Covariance
            cova = np.sum((res - mean_res) * (res_prev - mean_res_p)) / (N - 1)
            front = (2 * cova) / (varres + varres_p)
            back = (2 * mean_res * mean_res_p) / ((mean_res ** 2) + (mean_res_p ** 2))

            values.append(np.sum(front * back))
        if "SSD" == qual:
            values.append(np.sum((res_prev - res) ** 2))
    if len(values) == 1:
        return np.array(values[0])
    else:
        return np.array(values)
