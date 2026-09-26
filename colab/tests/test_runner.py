import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def zipped(self, missing=False):
        src = self.root / 'student_resource.zip'
        with zipfile.ZipFile(src, 'w') as z:
            for f in runner.FILES[:-1] if missing else runner.FILES:
                z.writestr('student_resource/dataset/' + f, f + '\n')
            z.writestr('../../unwanted.txt', 'ignore arbitrary ZIP members')
        return src

    def test_nested_zip_exact_files_and_reuse(self):
        source = self.zipped()
        target = self.root / 'dataset'
        hashes = runner.stage_dataset(source, target)
        self.assertEqual(set(hashes), set(runner.FILES))
        self.assertFalse((self.root / 'unwanted.txt').exists())
        self.assertEqual(hashes, runner.stage_dataset(source, target))
        # Corrupt an extracted file: re-stage rather than trusting old completion.
        (target / runner.FILES[0]).write_text('partial')
        self.assertEqual(hashes, runner.stage_dataset(source, target))

    def test_missing_training_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, 'seven TSV'):
            runner.stage_dataset(self.zipped(missing=True), self.root / 'dataset')

    def test_directory_copy_and_changed_input(self):
        source = self.root / 'source'
        for f in runner.FILES:
            p = source / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f)
        target = self.root / 'dataset'
        first = runner.stage_dataset(source, target)
        (source / runner.FILES[0]).write_text('changed')
        second = runner.stage_dataset(source, target)
        self.assertNotEqual(first, second)
        self.assertEqual((source / runner.FILES[0]).read_text(), 'changed')

    def make_checkpoint(self):
        local, remote = self.root / 'local', self.root / 'drive'
        cp = runner.Checkpoints(local, remote, {'code': 'test', 'mode': 'full'})
        p = local / 'work/models/model.txt'
        p.parent.mkdir(parents=True)
        p.write_text('complete model')
        cp.save(['train'])
        return cp, p

    def test_interrupted_stage_discards_partial_and_restores_completed(self):
        cp, p = self.make_checkpoint()
        p.write_text('partial overwritten model')
        broken = cp.local / 'work/incomplete.npy'
        broken.write_text('incomplete memmap')
        runner.atomic_json(cp.local / 'stage_running.json', {'stage': 'predict'})
        self.assertEqual(cp.restore(), ['train'])
        self.assertEqual(p.read_text(), 'complete model')
        self.assertFalse(broken.exists())

    def test_failed_backup_keeps_previous_manifest(self):
        cp, p = self.make_checkpoint()
        original = cp.manifest.read_bytes()
        p.write_text('new model data')
        with patch.object(runner, 'atomic_copy', side_effect=OSError('Drive quota')):
            with self.assertRaises(OSError):
                cp.save(['train', 'predict'])
        self.assertEqual(cp.manifest.read_bytes(), original)
        runner.atomic_json(cp.local / 'stage_running.json', {'stage': 'predict'})
        self.assertEqual(cp.restore(), ['train'])
        self.assertEqual(p.read_text(), 'complete model')

    def test_changed_signature_rejected(self):
        cp, _ = self.make_checkpoint()
        other = runner.Checkpoints(cp.local, cp.remote, {'mode': 'dev'})
        with self.assertRaisesRegex(RuntimeError, 'NEW RUN_NAME'):
            other.restore()

    def test_corrupted_backup_rejected(self):
        cp, p = self.make_checkpoint()
        data = json.loads(cp.manifest.read_text())
        blob = cp.blobs / next(iter(data['files'].values()))['sha256']
        blob.write_text('x' * len('complete model'))
        p.unlink()
        with self.assertRaisesRegex(RuntimeError, 'corrupted'):
            cp.restore()

    def test_full_and_dev_arguments_preserve_model_cap(self):
        for mode, frac in [('full', '1.0'), ('dev', '0.03')]:
            args = runner.command('python', Path('/data'), Path('/local'), 'train', 'train', mode, 512, 2)
            self.assertEqual(args[args.index('--dev-frac') + 1], frac)
            self.assertEqual(args[args.index('--max-train-rows') + 1], '30000000')
            self.assertIn('--validator', args)
            self.assertIn('--dense', args)
            self.assertNotIn('--no-dense', args)


if __name__ == '__main__':
    unittest.main()
