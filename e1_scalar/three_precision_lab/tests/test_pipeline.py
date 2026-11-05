import unittest,tempfile,json
from pathlib import Path
from tpir.experiments import run,collect,highprec
from tpir.report import report

class PipelineTests(unittest.TestCase):
    def test_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cfg=p/'config.json';out=p/'run'
            cfg.write_text(json.dumps({'cases':[{'n':6,'kappa':10,'seed':0}], 'methods':['hss','hsd','ssd'],'max_iterations':4}))
            run(cfg,out);report(out)
            self.assertEqual(len(list((out/'records').glob('*.json'))),3)
            snapshot={f.name:f.read_bytes() for f in (out/'records').glob('*.json')}
            run(cfg,out,True)
            for f in (out/'records').glob('*.json'):self.assertEqual(f.read_bytes(),snapshot[f.name])
            collect([out],p/'archive.zip');self.assertTrue((p/'archive.zip').exists())
            with self.assertRaises(ValueError):run(cfg,out)
            cfg.write_text(cfg.read_text().replace('"max_iterations": 4','"max_iterations": 5'))
            with self.assertRaises(ValueError):run(cfg,out,True)
    def test_highprecision_audit(self):
        try:import mpmath
        except ImportError:self.skipTest('mpmath optional dependency not installed')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cfg=p/'config.json';out=p/'run'
            cfg.write_text(json.dumps({'cases':[{'n':5,'kappa':100,'seed':0}], 'methods':['hss','hsd'],'max_iterations':10}))
            run(cfg,out);v=highprec(out,dps=60,max_cases=1)
            self.assertEqual(len(v),2)
            for r in v:
                self.assertAlmostEqual(r['forward_error'],float(r['recorded_forward_error']),delta=1e-12)
    def test_resume_refuses_changed_environment(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cfg=p/'c.json';out=p/'run'
            cfg.write_text(json.dumps({'cases':[{'n':4,'kappa':2,'seed':0}],'methods':['hsd'],'max_iterations':2}))
            run(cfg,out)
            m=json.loads((out/'manifest.json').read_text());m['numpy']='different';(out/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Environment changed'):run(cfg,out,True)
    def test_resume_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cfg=p/'c.json';out=p/'run'
            cfg.write_text(json.dumps({'cases':[{'n':4,'kappa':2,'seed':0}],'methods':['hsd'],'max_iterations':2}))
            run(cfg,out)
            c=next((out/'cases').glob('*.json'));d=json.loads(c.read_text());d['A_sha256']='corrupted';c.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'Regenerated inputs differ'):run(cfg,out,True)
    def test_completed_case_bytes_preserved_on_resume(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cfg=p/'c.json';out=p/'run'
            cfg.write_text(json.dumps({'cases':[{'n':4,'kappa':2,'seed':0}],'methods':['hsd'],'max_iterations':2}))
            run(cfg,out)
            snapshot={f:f.read_bytes() for f in (out/'cases').glob('*')}
            run(cfg,out,True)
            for f,original in snapshot.items():self.assertEqual(f.read_bytes(),original)
