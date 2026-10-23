import json
from pathlib import Path
import sys
import time

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('DCNv3 comparison requires a CUDA GPU and compiled extension.')
    operator_dir = Path(__file__).resolve().parents[1] / 'models' / 'ops_dcnv3'
    sys.path.insert(0, str(operator_dir))
    from functions.dcnv3_func import DCNv3Function, dcnv3_core_pytorch

    torch.manual_seed(20261021)
    start = time.perf_counter()
    results = []
    groups, channels, kernel = 2, 16, 3
    for batch, height, width, scale in ((1, 5, 7, 1.0), (2, 6, 4, 2.0)):
        image = torch.randn(batch, height, width, groups * channels, device='cuda') * 0.1
        offset = torch.rand(batch, height, width, groups * kernel * kernel * 2,
                            device='cuda') * 0.8 - 0.4
        mask = torch.rand(batch, height, width, groups, kernel * kernel, device='cuda')
        mask = mask.softmax(-1).flatten(-2).contiguous()
        reference_inputs = [value.detach().clone().requires_grad_()
                            for value in (image, offset, mask)]
        cuda_inputs = [value.detach().clone().requires_grad_()
                       for value in (image, offset, mask)]
        parameters = (kernel, kernel, 1, 1, 1, 1, 1, 1, groups, channels, scale)
        expected = dcnv3_core_pytorch(*reference_inputs, *parameters)
        actual = DCNv3Function.apply(*cuda_inputs, *parameters, 2)
        upstream = torch.randn_like(expected)
        reference_grads = torch.autograd.grad(expected, reference_inputs, upstream)
        cuda_grads = torch.autograd.grad(actual, cuda_inputs, upstream)
        torch.cuda.synchronize()
        comparisons = [('forward', actual, expected)] + [
            (name, measured, reference)
            for name, measured, reference in zip(
                ('grad_input', 'grad_offset', 'grad_mask'), cuda_grads, reference_grads
            )
        ]
        errors = {}
        for name, measured, reference in comparisons:
            torch.testing.assert_close(measured, reference, rtol=2e-4, atol=2e-5)
            errors[name] = float((measured - reference).detach().abs().max())
        results.append({'shape': list(image.shape), 'offset_scale': scale,
                        'max_absolute_errors': errors})
    elapsed = time.perf_counter() - start
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(),
                      'capability': list(torch.cuda.get_device_capability()),
                      'dtype': 'float32', 'rtol': 2e-4, 'atol': 2e-5,
                      'elapsed_seconds': elapsed, 'cases': results, 'passed': True}, indent=2))

if __name__ == '__main__':
    main()
