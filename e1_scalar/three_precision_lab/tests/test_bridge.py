import unittest
import numpy as np
try:
 import torch
 TORCH=True
except ImportError:TORCH=False

@unittest.skipUnless(TORCH,'PyTorch optional')
class BridgeTests(unittest.TestCase):
 def test_cpu_mixed_outer_roles(self):
    from tpir.bridge import solve_bridge
    from tpir.problems import make_problem,reference_solution,errors
    a,b,_,_=make_problem(dict(n=12,kappa=10,seed=0))
    r,h=solve_bridge(a,b,'mixed16inputs_w32_r64',device='cpu',block_size=4)
    ref,_=reference_solution(a,b)
    self.assertEqual(h.dtype,np.float32)
    self.assertEqual(r['factor_metadata']['num_half_gemm_updates'],2)
    self.assertLess(errors(a,b,h[-1],ref,32)['forward_error_over_u'],10)
 def test_cpu_native_roles(self):
    from tpir.bridge import solve_bridge
    a=np.array([[4,1],[1,3]],np.float32);b=np.array([1,2],np.float32)
    r,h=solve_bridge(a,b,'native32_w32_r64',device='cpu')
    self.assertEqual(h.dtype,np.float32);self.assertEqual(r['residual_dtype'],'torch.float64')
 @unittest.skipUnless(TORCH and torch.cuda.is_available(),'No CUDA device')
 def test_cuda_mixed_smoke(self):
    from tpir.bridge import solve_bridge
    a=np.eye(32,dtype=np.float32)+np.ones((32,32),dtype=np.float32)*.01;b=np.ones(32,np.float32)
    r,h=solve_bridge(a,b,'mixed16inputs_w32_r64',device='cuda',block_size=8)
    self.assertNotEqual(r['status'],'numerical_failure');self.assertEqual(r['factor_metadata']['num_half_gemm_updates'],3)
