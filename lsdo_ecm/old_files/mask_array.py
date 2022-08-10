import numpy as np
h_m = np.random.random((3,2))
h_lower=0.2
h_upper=0.6
tropos_mask = h_m <= h_lower
strato_mask = h_m >  h_upper
smooth_mask = np.logical_and(~tropos_mask, ~strato_mask)

temp_coeffs = (1,2,3,4)

T0 = 100
L = 8
T1=25

def compute_temps(h_m, tropos_mask, strato_mask, smooth_mask):
    a, b, c, d = temp_coeffs
    temp_K = np.zeros(h_m.shape, dtype=h_m.dtype)
    temp_K += tropos_mask * (T0 - L * h_m)
    temp_K += strato_mask * T1
    temp_K += smooth_mask * (a * h_m ** 3 + b * h_m ** 2 + c * h_m + d)
    return temp_K

def compute_temp_derivs(h_m, tropos_mask, strato_mask, smooth_mask):
    a, b, c, _ = temp_coeffs

    derivs = np.zeros(h_m.shape, dtype=h_m.dtype)
    derivs += tropos_mask * -L
    derivs += smooth_mask * (3 * a * h_m ** 2 + 2 * b * h_m + c)

    return derivs