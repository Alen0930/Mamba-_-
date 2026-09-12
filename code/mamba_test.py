import torch
from mamba_ssm import Mamba, Mamba2
import mamba_ssm.modules.mamba_simple as ms
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
ms.selective_scan_fn = lambda u, dt, A, B, C, D=None, z=None, db=None, ds=False, **kw: selective_scan_ref(u, dt, A, B, C, D, z, db, ds)
print(torch.cuda.is_available())
d = 'cuda'
m1 = Mamba(d_model=16, d_state=16, d_conv=4, expand=2, use_fast_path=False).to(d)
m2 = Mamba2(d_model=128, d_state=128, d_conv=4, expand=2, use_mem_eff_path=False).to(d)
x1 = torch.randn(4, 32, 16).to(d)
x2 = torch.randn(4, 32, 128).to(d)
y1 = m1(x1)
y2 = m2(x2)
print(y1.shape, y2.shape, bool(y1.isfinite().all() and y2.isfinite().all()))
