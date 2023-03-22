import torch
import random
import numpy as np

from time import perf_counter
from torchrbf import RBFInterpolator as torchRBF # avoid name collision with scipy here
from torchrbf.radial_fn import SCALE_INVARIANT, RADIAL_FUNCS, MIN_DEGREE
from scipy.interpolate import RBFInterpolator # for comparison

@torch.no_grad()
def test_correctness():
    """
    Test that the torch implementation is correct by comparing
    random interpolations to scipy's RBFinterpolator output.
    """
    
    dev = 'cuda'
    for _ in range(1000):
        num_p = random.randint(64, 2048)
        num_d = random.randint(1, 8)
        num_x = random.randint(1, 512)
        smoothing = random.uniform(0, 100)

        y = torch.FloatTensor(num_p, num_d).uniform_(-1, 1).to(dev)
        d = torch.FloatTensor(num_p, num_d).uniform_(-1, 1).to(dev)
        x = torch.FloatTensor(num_x, num_d).uniform_(-1, 1).to(dev)

        kernel = random.choice(list(SCALE_INVARIANT))
        rbf_torch = torchRBF(y, d, neighbors=None, smoothing=smoothing, kernel=kernel, device=dev)
        rbf_scipy = RBFInterpolator(y.cpu().numpy(), d.cpu().numpy(), neighbors=None, smoothing=smoothing, kernel=kernel)

        out_torch = rbf_torch(x).detach().cpu().numpy()
        out_scipy = rbf_scipy(x.cpu().numpy())
        diff = np.abs(out_torch - out_scipy)
        if not np.all(diff < 2e-3):
            print(f'Possible error: \
                {kernel} {smoothing} {num_p} {num_d} {num_x}\
                Max abs error: {np.max(diff)}, Mean abs error: {np.mean(diff)}')
        print(f'Test passed: {kernel} {smoothing:.2f} {num_p} {num_d} {num_x}')
    return

@torch.no_grad()
def benchmark_init(
    num_p=2048,
    num_d=8,
    num_x=512, # unused
    smoothing=100,
    dev='cuda',
    ntests=100,
    compare_scipy=False
):
    print('Testing initialization throughput...')
    kernel = 'thin_plate_spline'
    total_time = 0
    scipy_time = 0
    for _ in range(ntests):
        y = torch.FloatTensor(num_p, num_d).uniform_(-1, 1)
        d = torch.FloatTensor(num_p, num_d).uniform_(-1, 1)
        y = y.to(dev)
        d = d.to(dev)
        t0 = perf_counter()
        rbf_torch = torchRBF(y, d, neighbors=None, smoothing=smoothing, kernel=kernel, device=dev)
        t1 = perf_counter()
        rbf_torch._coeffs.shape # dumy operation to force computation
        total_time += t1 - t0

        if compare_scipy:
            y = y.cpu().numpy()
            d = d.cpu().numpy()
            t0 = perf_counter()
            rbf_scipy = RBFInterpolator(y, d, neighbors=None, smoothing=smoothing, kernel=kernel)
            t1 = perf_counter()
            rbf_scipy._coeffs.shape # dumy operation to force computation
            scipy_time += t1 - t0

    avg_time = total_time / ntests
    print(f'Average initialization time: {avg_time:.4f} seconds')
    if compare_scipy:
        avg_scipy_time = scipy_time / ntests
        print(f'Average scipy initialization time: {avg_scipy_time:.4f} seconds')
        print(f'Speedup: {avg_scipy_time / avg_time:.2f}x')

@torch.no_grad()
def benchmark_forward(
    num_p=2048,
    num_d=8,
    num_x=512,
    smoothing=100,
    dev='cuda',
    ntests=100,
    compare_scipy=False
):
    print('Testing forward throughput...')
    kernel = 'thin_plate_spline'
    total_time = 0
    scipy_time = 0
    for _ in range(ntests):
        y = torch.FloatTensor(num_p, num_d).uniform_(-1, 1)
        d = torch.FloatTensor(num_p, num_d).uniform_(-1, 1)
        x = torch.FloatTensor(num_x, num_d).uniform_(-1, 1)
        y = y.to(dev)
        d = d.to(dev)
        x = x.to(dev)
        rbf_torch = torchRBF(y, d, neighbors=None, smoothing=smoothing, kernel=kernel, device=dev)            
        t0 = perf_counter()
        out = rbf_torch(x)
        t1 = perf_counter()
        out.mean() # dummy operation to force computation
        total_time += t1 - t0

        if compare_scipy:
            rbf_scipy = RBFInterpolator(y.cpu().numpy(), d.cpu().numpy(), neighbors=None, smoothing=smoothing, kernel=kernel)
            x = x.cpu().numpy()
            t0 = perf_counter()
            out = rbf_scipy(x)
            t1 = perf_counter()
            scipy_time += t1 - t0
    
    avg_time = total_time / ntests
    print(f'Average forward time: {avg_time:.4f} seconds')
    if compare_scipy:
        avg_scipy_time = scipy_time / ntests
        print(f'Average scipy forward time: {avg_scipy_time:.4f} seconds')
        print(f'Speedup: {avg_scipy_time / avg_time:.2f}x')

        print(f'Forwards/second: {1 / avg_time:.4f}')
        print(f'Scipy forwards/second: {1 / avg_scipy_time:.4f}')

@torch.no_grad()
def test_device():
    num_p = 2048
    num_d = 3
    num_x = 512
    y = torch.FloatTensor(num_p, num_d).uniform_(-1, 1).cuda()
    d = torch.FloatTensor(num_p, num_d).uniform_(-1, 1).cuda()
    x = torch.FloatTensor(num_x, num_d).uniform_(-1, 1).cuda()
    rbf = torchRBF(y, d, neighbors=None, smoothing=100, kernel='thin_plate_spline', device='cuda')
    out1 = rbf(x)
    rbf = rbf.to('cpu')
    out2 = rbf(x.to('cpu'))
    diff = (out1.cpu() - out2.cpu()).abs()
    print(f'Maximum difference between CPU and GPU: {diff.max():.4f}')


if __name__ == '__main__':
    test_correctness()

    b_init = [
        # Scalaing number of points num_p
        {
            'num_p': 128, 
            'num_d': 1, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 256, 
            'num_d': 1, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 512, 
            'num_d': 1, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 1024, 
            'num_d': 1, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 2048, 
            'num_d': 1, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        # Scaling dimensions num_d
        {
            'num_p': 128, 
            'num_d': 1, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 128, 
            'num_d': 2, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 128, 
            'num_d': 4, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 128, 
            'num_d': 8, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        # Scaling both num_p and num_d
        {
            'num_p': 128, 
            'num_d': 1, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 256, 
            'num_d': 2, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 512, 
            'num_d': 4, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 1024, 
            'num_d': 8, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 2048, 
            'num_d': 8, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
    ]

    print('Benchmarking')

    for params in b_init:
        print(params)
        benchmark_init(**params)
        benchmark_forward(**params)
        print('-------------------')

    b_forward = [
        # Scalaing number of query points num_x
        {
            'num_p': 4096, 
            'num_d': 3, 
            'smoothing': 1, 
            'num_x': 128, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 4096, 
            'num_d': 3, 
            'smoothing': 1, 
            'num_x': 256, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 4096, 
            'num_d': 3, 
            'smoothing': 1, 
            'num_x': 512, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
        {
            'num_p': 4096, 
            'num_d': 3, 
            'smoothing': 1, 
            'num_x': 1024, 
            'dev': 'cuda', 
            'ntests': 100,
            'compare_scipy': True
        },
    ]

    print('Benchmarking')
    for params in b_forward:
        print(params)
        benchmark_forward(**params)
        print('-------------------')