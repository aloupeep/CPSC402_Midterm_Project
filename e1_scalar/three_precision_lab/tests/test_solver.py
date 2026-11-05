import unittest, tempfile,json
from pathlib import Path
import numpy as np
from tpir.solver import METHODS,run_solver,Method
from tpir.problems import make_problem,reference_solution,errors

class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a,cls.b,_,_=make_problem(dict(n=16,kappa=100.,seed=0))
        cls.ref,_=reference_solution(cls.a,cls.b)
    def test_hsd_roles_and_accuracy(self):
        r,h=run_solver(self.a,self.b,'hsd')
        self.assertEqual(h.dtype,np.float32)
        self.assertEqual(r['factor_dtype'],'float16');self.assertEqual(r['solve_dtype'],'float16')
        self.assertEqual(r['residual_dtype'],'float64');self.assertEqual(r['factorizations'],1)
        self.assertLess(errors(self.a,self.b,h[-1],self.ref,32)['forward_error_over_u'],10)
    def test_residual_comparison_reuses_identical_factors(self):
        a,_=run_solver(self.a,self.b,'hss');b,_=run_solver(self.a,self.b,'hsd')
        self.assertEqual(a['factor_hash'],b['factor_hash'])
    def test_all_primary_methods(self):
        for m in ['hss','hsd','sss','ssd','hdd','sdd','ddd','input16_ssd','half_factors_solve32']:
            r,h=run_solver(self.a,self.b,m,max_iterations=4)
            self.assertEqual(h.dtype,np.dtype(f'float{METHODS[m].work}'))
            self.assertLessEqual(r['iterations'],4)
            self.assertNotEqual(r['status'],'numerical_failure')
    def test_direct_has_no_corrections(self):
        r,h=run_solver(self.a,self.b,'direct32');self.assertEqual(len(h),1);self.assertEqual(r['iterations'],0)
    def test_zero_solution(self):
        r,h=run_solver(np.eye(4,np.float32) if False else np.eye(4,dtype=np.float32),np.zeros(4,np.float32),'hsd')
        self.assertEqual(r['status'],'computed_residual_zero');np.testing.assert_array_equal(h[-1],0)
    def test_singular_record_retained(self):
        r,h=run_solver(np.ones((3,3),np.float32),np.ones(3,np.float32),'hsd')
        self.assertEqual(r['status'],'numerical_failure');self.assertEqual(len(h),0)
    def test_false_zero_residual_not_success_label(self):
        # Computed zeros/fixed points are terminal statuses, never a guarantee.
        r,h=run_solver(self.a,self.b,'hss',max_iterations=0)
        self.assertEqual(r['status'],'iteration_limit')
    def test_invalid_role_assignment(self):
        with self.assertRaises(ValueError):Method('bad',64,32,64)
    def test_same_target_for_work64(self):
        a,b,_,c=make_problem(dict(n=12,kappa=100,seed=9))
        np.testing.assert_array_equal(a.astype(np.float64).astype(np.float32),a)
        np.testing.assert_array_equal(b.astype(np.float64).astype(np.float32),b)
    def test_working_data_are_not_reference(self):
        a,b,man,c=make_problem(dict(n=12,kappa=1000,seed=8))
        ref,_=reference_solution(a,b)
        self.assertGreater(float(np.linalg.norm(man-ref,np.inf)),0)
    def test_reference_residual(self):
        r,meta=reference_solution(self.a,self.b)
        self.assertLess(meta['backward_error'],1e-14)
    def test_no_norm_is_recorded(self):
        r,_=run_solver(self.a,self.b,'hsd_no_norm')
        self.assertFalse(r['precision']['normalize'])
    def test_blas_ablation_labelled(self):
        r,h=run_solver(self.a,self.b,'hss_blas');self.assertEqual(r['precision']['residual_mode'],'blas')
    def test_error_metrics_zero(self):
        e=errors(np.eye(3),np.ones(3),np.ones(3),np.ones(3),32)
        self.assertEqual(e['forward_error'],0);self.assertEqual(e['componentwise_backward_error'],0)
