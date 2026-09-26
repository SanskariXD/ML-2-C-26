import importlib.util
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location('fix', Path(__file__).with_name('kaggle_storage_fix.py'))
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)
helper = types.ModuleType('compact_storage')
exec(fix.HELPER, helper.__dict__)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def sample(self):
        rng = np.random.default_rng(42)
        base = rng.normal(size=(71, 384)).astype(np.float16)
        inv = np.r_[np.arange(71), rng.integers(0, 71, size=300)].astype(np.int32)
        p = self.root/'emb_idx.npy'
        u = self.root/'emb_idx.npy.uniq.tmp.npy'
        np.save(u, base)
        return p, u, base, inv
    def test_bit_exact_reads_cosines_and_topk(self):
        p,u,base,inv = self.sample()
        before = u.read_bytes()
        helper.save_indexed(p,u,inv)
        a = helper.load_embedding(p)
        expected = base[inv]
        for key in [slice(None), slice(3,29,2), 23, -1, np.array([9,2,9,71,95]), np.array([],np.int64)]:
            np.testing.assert_array_equal(a[key], expected[key])
        np.testing.assert_array_equal(a[[9,2], :12], expected[[9,2], :12])
        queries = expected[:7].astype(np.float32)
        reference = queries @ expected.astype(np.float32).T
        actual = queries @ a[:].astype(np.float32).T
        np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(np.argsort(actual,axis=1),np.argsort(reference,axis=1))
        self.assertEqual(before, u.read_bytes())
        self.assertFalse(p.exists())
        a.release_pages()
    def test_legacy_arrays(self):
        p,u,base,inv = self.sample()
        np.save(p, base[inv])
        np.testing.assert_array_equal(helper.load_embedding(p), base[inv])
    def test_publish_failure_preserves_source_and_can_retry(self):
        p,u,base,inv = self.sample()
        before = u.read_bytes()
        replace = helper.os.replace
        def fail_manifest(a,b):
            if str(b).endswith('.indexed.json'):
                raise OSError('simulated interrupted publication')
            return replace(a,b)
        with patch.object(helper.os,'replace',side_effect=fail_manifest):
            with self.assertRaises(OSError):
                helper.save_indexed(p,u,inv)
        self.assertEqual(before,u.read_bytes())
        self.assertFalse(helper.embedding_exists(p))
        helper.save_indexed(p,u,inv)
        np.testing.assert_array_equal(helper.load_embedding(p)[:],base[inv])
    def test_invalid_inverse_preserves_checkpoint(self):
        p,u,base,inv = self.sample()
        before=u.read_bytes()
        inv[0]=999
        with self.assertRaises(ValueError):
            helper.save_indexed(p,u,inv)
        self.assertEqual(before,u.read_bytes())
    def test_disk_guard_does_not_truncate_file(self):
        p=self.root/'important.npy'
        p.write_bytes(b'preserve')
        with patch.object(helper.shutil,'disk_usage',return_value=types.SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError,'DISK SPACE'):
                helper.safe_open_memmap(p,mode='w+',dtype=np.float16,shape=(999999,384))
        self.assertEqual(p.read_bytes(),b'preserve')
    def test_allocation_guard_and_read(self):
        p=self.root/'new.npy'
        a=helper.safe_open_memmap(p,mode='w+',dtype=np.float16,shape=(12,3))
        a[:]=3
        a.flush()
        np.testing.assert_array_equal(np.load(p),np.full((12,3),3,np.float16))
    def test_completed_encode_resumes_without_model_inference(self):
        import hashlib
        import sys
        from contextlib import nullcontext
        p,u,base,inv=self.sample()
        texts=[f'text {j}' for j in inv]
        fingerprint=hashlib.sha1(('\n'.join(f'text {j}' for j in range(len(base))) + f'\n#n={len(base)}\n#dim=384').encode()).hexdigest()[:16]
        Path(str(p)+'.uniq.meta').write_text(fingerprint)
        Path(str(u)+'.progress').write_text(str(len(base)))
        src=(Path(__file__).parents[1]/'code/business_entity_resolution/src/dense.py').read_text()
        module=types.ModuleType('test_dense')
        utils=types.SimpleNamespace(LOG=types.SimpleNamespace(info=lambda *a:None),timed=lambda *a:nullcontext())
        torch=types.SimpleNamespace(set_num_threads=lambda *a:None)
        cfg=types.SimpleNamespace(low_memory=False,force=False,dense_batch=512,dense_batch_fixed=True)
        calls=[]
        def encoder(models,uniq,mm,resume,nu,*args,**kw):
            calls.append((resume,nu))
            self.assertEqual(resume,nu)
        with patch.dict(sys.modules,utils=utils,compact_storage=helper,torch=torch):
            exec(fix.transform_dense(src),module.__dict__)
            module._pick_device=lambda:('cpu',False)
            module._load_st_model=lambda *a:types.SimpleNamespace(get_sentence_embedding_dimension=lambda:384)
            module._encode_unique_parallel=encoder
            module._encode_to(str(p),texts,cfg)
        self.assertEqual(calls,[(71,71)])
        np.testing.assert_array_equal(helper.load_embedding(p)[:],base[inv])
        self.assertFalse(Path(str(p)+'.tmp.npy').exists())
    def test_move_verifies_and_preserves_work_path(self):
        work=self.root/'work'
        work.mkdir()
        (work/'checkpoint').write_bytes(b'precious checkpoint')
        destination=self.root/'scratch'
        with patch.object(fix,'different_filesystem',return_value=True):
            self.assertTrue(fix.relocate_work(work,destination,reserve_gib=0))
        self.assertTrue(work.is_symlink())
        self.assertEqual((work/'checkpoint').read_bytes(),b'precious checkpoint')
        self.assertFalse((self.root/'work.before_scratch_move').exists())
    def test_move_refuses_same_filesystem(self):
        work=self.root/'work'
        work.mkdir()
        self.assertFalse(fix.relocate_work(work,self.root/'scratch',reserve_gib=0))
        self.assertFalse(work.is_symlink())
    def test_installer_idempotent(self):
        repo=self.root/'repo'
        shutil.copytree(Path(__file__).parents[1]/'code/business_entity_resolution/src',repo/'src')
        fix.patch_repo(repo)
        first={p.name:p.read_bytes() for p in (repo/'src').glob('*.py')}
        fix.patch_repo(repo)
        second={p.name:p.read_bytes() for p in (repo/'src').glob('*.py')}
        self.assertEqual(first,second)
        self.assertIn('scatter skipped',first['dense.py'].decode())
        self.assertNotIn('# Scatter unique',first['dense.py'].decode())

if __name__=='__main__':
    unittest.main()
