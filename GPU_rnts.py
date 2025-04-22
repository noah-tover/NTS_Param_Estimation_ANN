# This script contains python-translated code for random normal tempered stable numbers from the TemStaR v.90 library. 
# It is designed to work on the GPU in order to speed up simulation. 
import math
import torch
import os
import pandas as pd
import numpy as np

def change_ntsparam2stdntsparam(ntsparam):
    """
    Convert raw NTS params to standardized params.
    ntsparam: list or 1D tensor of length 3, 4, 5, or 6
    Returns dict with keys 'stdparam' (tensor [alpha, theta_std, beta_std]),
    'mu' (scalar), 'sig' (scalar).
    """
    p = torch.as_tensor(ntsparam, dtype=torch.float64)
    n = p.numel()

    if n == 3:
        a, th_std, bet_std = p
        sig_std = torch.tensor(1.0, dtype=p.dtype)
        mu_std  = torch.tensor(0.0, dtype=p.dtype)

    elif n == 4:
        a = p[0]
        th = p[1] * p[3]
        bet = p[2] * p[3]
        gam = torch.sqrt(p[3] - bet**2 * (2 - a) / (2 * th))
        m  = torch.tensor(0.0, dtype=p.dtype)
        dt = torch.tensor(1.0, dtype=p.dtype)

        z = (2 - a) / (2 * th)
        th_std  = th * dt
        bet_std = bet * torch.sqrt(dt / (gam**2 + bet**2 * z))
        sig_std = torch.sqrt(gam**2 * dt + bet**2 * dt * z)
        mu_std  = m * dt

    else:
        a   = p[0]
        th  = p[1]
        bet = p[2]
        gam = p[3]
        m   = p[4]
        dt  = p[5] if n >= 6 else torch.tensor(1.0, dtype=p.dtype)

        z = (2 - a) / (2 * th)
        th_std  = th * dt
        bet_std = bet * torch.sqrt(dt / (gam**2 + bet**2 * z))
        sig_std = torch.sqrt(gam**2 * dt + bet**2 * dt * z)
        mu_std  = m * dt

    stdparam = torch.stack([a, th_std, bet_std])
    return {'stdparam': stdparam, 'mu': mu_std, 'sig': sig_std}


def chf_stdNTS(u, param):

    a, th, b = param
    g2 = torch.abs(1 - b**2 * (2 - a) / (2 * th))

   
    inner = th + g2 * u**2 / 2 - 1j * (b * u)

    expo = 1j * (-b) * u \
           - (2 * th**(1 - a/2) / a) * (inner**(a/2) - th**(a/2))

    return torch.exp(expo)


def ipnts(u, ntsparam, maxmin=(-10, 10), du=0.01, N=2**17):
    """
    u: tensor of uniforms in (0,1)
    ntsparam: raw param list for NTS
    du: grid spacing
    N: number of FFT points (power of two)
    """
    device = u.device
    newp = change_ntsparam2stdntsparam(ntsparam)
    stdp = newp['stdparam']
    mu, sig = newp['mu'], newp['sig']

    h = du
    s = 1.0 / (h * N)
    t1 = torch.arange(1, N+1, device=device, dtype=torch.float64)
    t2 = 2 * math.pi * (t1 - 1 - N/2) * s

    cfv = chf_stdNTS(t2, stdp)

    alt = ((-1.0)**(t1 - 1)).to(device)
    x1  = alt * cfv

    fftv = torch.fft.fft(x1)
    factor = s * ((-1.0)**(t1 - 1 - N/2)).to(device)
    pdf  = (fftv * factor).real.clamp(min=0)

    x = (t1 - 1 - N/2) * h
    cdf = torch.cumsum(pdf * h, dim=0)
    cdf = cdf / cdf[-1]

    # lin interpolation for FFT
    u_flat = u.flatten()
    idx = torch.searchsorted(cdf, u_flat, right=False)
    idx = idx.clamp(1, N-1)

    x_low  = x[idx - 1]
    x_high = x[idx]
    c_low  = cdf[idx - 1]
    c_high = cdf[idx]

    vals = x_low + (u_flat - c_low) * (x_high - x_low) / (c_high - c_low)
    vals = vals.view_as(u)

    return vals * sig + mu


def rnts(n, ntsparam, u=None, device=None):
    """
    Random num generator. Just an ipnts wrapper.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if u is None:
        u = torch.rand(n, device=device, dtype=torch.float64)
    else:
        u = torch.as_tensor(u, device=device, dtype=torch.float64)

    return ipnts(u, ntsparam)

####################################################################




def generate_rnts_matrices_from_halton(paths, csv_path, output_dir, nstart, nend):
    torch.manual_seed(1234)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(csv_path, index_col=0)

    for idx in range(nstart, nend + 1):
        row = df.iloc[idx - 1]  # CSV index is 1-based from R

        alpha = row['alpha']
        theta = row['theta']
        beta  = row['betas']
        gamma = row['gammas']
        tao   = row['tao']
        mu    = 0.0

        ntsparam = [alpha, theta, beta, gamma, mu]

        ntimestep = math.ceil(tao * 250)

        u = torch.rand(paths, ntimestep, dtype=torch.float64, device=device)

        X = ipnts(u, ntsparam)  # stays on GPU

        output_path = os.path.join(output_dir, f"rnts_{idx}.pt")
        torch.save(X, output_path)

        print(f"Saved {output_path}")


generate_rnts_matrices_from_halton(20000, halton_csv, output_folder, nstart, nend)

for filename in os.listdir(output_folder):
    if filename.startswith('rnts') and filename.endswith('.pt'):
        file_path = os.path.join(output_folder, filename)

        tensor = torch.load(file_path)
        array = tensor.cpu().numpy()

        npy_filename = filename.replace('.pt', '.npy')
        npy_path = os.path.join(output_folder, npy_filename)

        np.save(npy_path, array)

        print(f"Converted {filename} to {npy_filename}")
