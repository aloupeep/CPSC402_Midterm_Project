import unittest
from fractions import Fraction
import numpy as np
from tpir.arithmetic import Arithmetic,lu_factor,residual,scaled_solve,NumericalFailure


def exact_round16(value):
    """Independent rational nearest-neighbor oracle, including tie-to-even."""
    sign=-1 if value<0 else 1
    if abs(value)>=65520:return np.float16(sign*np.inf)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        center=np.float16(float(value))
        candidates=[center,np.nextafter(center,np.float16(np.inf)),np.nextafter(center,np.float16(-np.inf))]
    candidates=[v for v in candidates if np.isfinite(v)]
    def key(v):
        distance=abs(Fraction(float(v))-value)
        bits=int(np.asarray(v).view(np.uint16))
        return distance,bits&1
    return min(candidates,key=key)

class ArithmeticTests(unittest.TestCase):
    def test_tie_to_even(self):
        h=Arithmetic(16)
        self.assertEqual(float(h.add(1,2**-11)),1)
        self.assertEqual(float(h.add(1+2**-10,2**-11)),1+2**-9)
    def test_subnormal(self):
        h=Arithmetic(16)
        self.assertEqual(float(h.cast(2**-24)),2**-24)
        self.assertEqual(float(h.cast(2**-25)),0)
    def test_overflow(self):
        self.assertTrue(np.isinf(Arithmetic(16).mul(65504,2)))
    def test_separate_not_fused(self):
        h=Arithmetic(16);a=h.cast(1+2**-10);b=h.cast(1-2**-10)
        separated=h.sub(1,h.mul(a,b))
        fused=h.cast(1-float(a)*float(b))
        self.assertEqual(float(separated),0)
        self.assertNotEqual(float(fused),0)
    def test_dtype_operations(self):
        for bits in (16,32,64):
            f=Arithmetic(bits)
            for op in (f.add,f.sub,f.mul,f.div):
                self.assertEqual(op([1,2],[2,3]).dtype,f.dtype)
    def test_random_rational_oracle(self):
        rng=np.random.default_rng(975)
        pairs=rng.integers(0,65536,size=(500,2),dtype=np.uint16).view(np.float16)
        f=Arithmetic(16);count=0
        for a,b in pairs:
            if not np.isfinite(a) or not np.isfinite(b):continue
            af,bf=Fraction(float(a)),Fraction(float(b))
            for op,exact in [(f.add,af+bf),(f.sub,af-bf),(f.mul,af*bf)]+([(f.div,af/bf)] if bf else []):
                v=float(op(a,b));oracle=float(exact_round16(exact))
                self.assertEqual(v,oracle,(a,b,op.__name__));count+=1
        self.assertGreater(count,1800)
    def test_residual_precision_matters(self):
        a=np.array([[1,2**-24]],dtype=np.float32);b=np.array([1],dtype=np.float32);x=np.ones(2,dtype=np.float32)
        self.assertEqual(float(residual(a,b,x,32)[0]),0)
        self.assertEqual(float(residual(a,b,x,64)[0]),-2**-24)
    def test_no_promoted_rounded_input_residual(self):
        a=np.array([[1+2**-12]],np.float32);x=np.ones(1,np.float32);b=np.ones(1,np.float32)
        self.assertNotEqual(float(residual(a,b,x,64)[0]),float(residual(a.astype(np.float16),b,x,64)[0]))
    def test_gepp_reconstruction(self):
        rng=np.random.default_rng(12);a=rng.normal(size=(9,9));a[0,0]=0
        for bits,tol in [(16,.02),(32,1e-5),(64,1e-13)]:
            f=lu_factor(a,bits);l=np.tril(f.lu.astype(float),-1)+np.eye(len(a));u=np.triu(f.lu.astype(float))
            err=np.linalg.norm(a[f.permutation,:]-l@u,np.inf)/np.linalg.norm(a,np.inf)
            self.assertLess(err,tol);self.assertEqual(f.lu.dtype,np.dtype(f'float{bits}'))
    def test_exact_pivoted_solve(self):
        a=np.array([[0,2,1],[1,0,0],[0,1,1]],float);x=np.array([1,2,3],float)
        for bits in [16,32,64]:
            f=lu_factor(a,bits);np.testing.assert_array_equal(f.solve(a@x),x)
    def test_half_is_not_rounded_single_factor(self):
        a=np.random.default_rng(5).normal(size=(8,8)).astype(np.float32)
        f16=lu_factor(a,16);f32=lu_factor(a.astype(np.float16).astype(np.float32),32)
        self.assertFalse(np.array_equal(f16.lu,f32.lu.astype(np.float16)))
    def test_same_factors_different_solve_arithmetic(self):
        a=np.random.default_rng(4).normal(size=(8,8));b=np.random.default_rng(6).normal(size=8)
        f=lu_factor(a,16);z16=f.solve(b);z32=f.solve(b,32)
        self.assertEqual(z16.dtype,np.float16);self.assertEqual(z32.dtype,np.float32)
        self.assertFalse(np.array_equal(z16.astype(np.float32),z32))
    def test_singular_is_not_fallback(self):
        with self.assertRaises(NumericalFailure):lu_factor([[1,1],[1,1]],16)
    def test_factor_overflow(self):
        with self.assertRaises(NumericalFailure):lu_factor([[1e8,0],[0,1]],16)
    def test_tiny_rhs_normalization(self):
        f=lu_factor(np.eye(3),16);b=np.array([1,2,-3],np.float64)*2**-40
        z,ex=scaled_solve(f,b,32,64,True)
        raw,_=scaled_solve(f,b,32,64,False)
        np.testing.assert_array_equal(z,b.astype(np.float32));np.testing.assert_array_equal(raw,np.zeros(3))
        self.assertLess(ex,0)
    def test_zero_rhs_normalization(self):
        f=lu_factor(np.eye(2),16);z,ex=scaled_solve(f,np.zeros(2),32,64)
        np.testing.assert_array_equal(z,np.zeros(2));self.assertEqual(ex,0)
    def test_fp32_update_rounding(self):
        w=Arithmetic(32);self.assertEqual(float(w.add(w.cast(1),w.cast(2**-25))),1.)
        self.assertNotEqual(float(Arithmetic(64).add(1,2**-25)),1.)
