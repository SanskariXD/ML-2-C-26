"""Colab orchestration around the unchanged BER pipeline (stdlib only).

Completed stages are checkpointed to Drive. An interrupted stage is recomputed
from the last complete checkpoint; partial memmaps are never treated as complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
BER = REPO / 'code/business_entity_resolution'
GIB = 1024 ** 3
FILES = [f'{split}/{split}_source{i}.tsv' for split in ('train', 'test') for i in (1, 2, 3)] + ['train/train_ground_truth.tsv']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def atomic_copy(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.partial')
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def dataset_root(source):
    source = Path(source)
    found = [p.parent.parent for p in source.rglob('train_source1.tsv')
             if all((p.parent.parent / f).is_file() for f in FILES)]
    if len(found) != 1:
        raise ValueError(f'Expected exactly one dataset with all seven TSVs under {source}; found {len(found)}')
    return found[0]


def stage_dataset(source, dest):
    """Copy only the seven required TSVs, with full hashes and a completion marker."""
    source, dest = Path(source), Path(dest)
    if not source.exists():
        raise FileNotFoundError(f'Set DATASET_PATH to your existing Drive folder or ZIP: {source}')
    marker = dest / 'dataset_manifest.json'
    source_stat = {'path': str(source.resolve())}
    if source.is_file():
        source_stat['zip_sha256'] = digest(source)
    else:
        root = dataset_root(source)
        source_stat['files'] = {f: digest(root / f) for f in FILES}
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get('source') == source_stat and all(
                (dest / f).is_file() and digest(dest / f) == old['files'][f] for f in FILES):
            print('Dataset already staged and verified.', flush=True)
            return old['files']
    dest.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        marker.unlink()
    if source.is_file():
        with zipfile.ZipFile(source) as z:
            names = set(z.namelist())
            prefixes = [n[:-len(FILES[0])] for n in names if n.endswith(FILES[0])]
            prefixes = [p for p in prefixes if all(p + f in names for f in FILES)]
            if len(prefixes) != 1:
                raise ValueError('ZIP must contain exactly one train/ and test/ dataset with all seven TSV files.')
            needed = sum(z.getinfo(prefixes[0] + f).file_size for f in FILES)
            if shutil.disk_usage(dest).free < needed + 2 * GIB:
                raise RuntimeError('Not enough local disk to extract the dataset.')
            for f in FILES:
                target = dest / f
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + '.partial')
                print('Extracting', f, flush=True)
                # Extract fixed known paths only; ignore arbitrary ZIP member paths.
                with z.open(prefixes[0] + f) as src, tmp.open('wb') as out:
                    shutil.copyfileobj(src, out, 8 * 1024 * 1024)
                os.replace(tmp, target)
    else:
        needed = sum((root / f).stat().st_size for f in FILES)
        if shutil.disk_usage(dest).free < needed + 2 * GIB:
            raise RuntimeError('Not enough local disk to copy the dataset.')
        for f in FILES:
            print('Copying', f, flush=True)
            atomic_copy(root / f, dest / f)
    hashes = {f: digest(dest / f) for f in FILES}
    if source.is_dir() and hashes != source_stat['files']:
        raise RuntimeError('Dataset changed while copying. Rerun the dataset cell.')
    atomic_json(marker, {'source': source_stat, 'files': hashes})
    return hashes


class Checkpoints:
    """Content-addressed files; publish a manifest only after every blob is copied."""
    def __init__(self, local, remote, signature):
        self.local, self.remote = Path(local), Path(remote)
        self.signature = signature
        self.remote.mkdir(parents=True, exist_ok=True)
        self.local.mkdir(parents=True, exist_ok=True)
        self.manifest = self.remote / 'checkpoint.json'
        self.blobs = self.remote / 'checkpoint_blobs'
        self.blobs.mkdir(exist_ok=True)

    def load(self):
        if not self.manifest.exists():
            return None
        data = json.loads(self.manifest.read_text())
        if data['signature'] != self.signature:
            raise RuntimeError('Run settings, dataset, or code changed. Use a NEW RUN_NAME to avoid incompatible caches.')
        return data

    def restore(self):
        state = self.load()
        # All work/out files belong to this dedicated run directory. If a stage
        # failed, discard its possibly incomplete caches before restoring.
        interrupted = (self.local / 'stage_running.json').exists()
        if interrupted:
            for sub in ('work', 'output'):
                shutil.rmtree(self.local / sub, ignore_errors=True)
        if state is None:
            if not interrupted and any((self.local / s).exists() for s in ('work', 'output')):
                raise RuntimeError('Local files exist without a complete Drive checkpoint. Use a new RUN_NAME.')
            return []
        for relative, info in state['files'].items():
            target = self.local / relative
            if target.is_file() and target.stat().st_size == info['size'] and digest(target) == info['sha256']:
                continue
            blob = self.blobs / info['sha256']
            if not blob.is_file() or blob.stat().st_size != info['size'] or digest(blob) != info['sha256']:
                raise RuntimeError(f'Checkpoint missing or corrupted: {relative}. Do not resume this run.')
            print('Restoring', relative, flush=True)
            atomic_copy(blob, target)
        (self.local / 'stage_running.json').unlink(missing_ok=True)
        return state['completed']

    def save(self, completed):
        files = {}
        for sub in ('work', 'output'):
            for path in sorted((self.local / sub).rglob('*')):
                if not path.is_file():
                    continue
                sha = digest(path)
                blob = self.blobs / sha
                if not blob.exists() or blob.stat().st_size != path.stat().st_size:
                    print('Backing up', path.relative_to(self.local), flush=True)
                    atomic_copy(path, blob)
                files[str(path.relative_to(self.local))] = {'sha256': sha, 'size': path.stat().st_size}
        atomic_json(self.manifest, {'signature': self.signature, 'completed': completed,
                                   'files': files, 'saved_at': time.time()})
        # Only after the new manifest is published: remove obsolete blobs.
        used = {v['sha256'] for v in files.values()}
        for blob in self.blobs.iterdir():
            if blob.name not in used:
                blob.unlink()
        (self.local / 'stage_running.json').unlink(missing_ok=True)


def command(python, data, local, command_name, split, mode, batch, workers):
    args = [python, '-u', str(BER / 'src/run.py'), command_name,
            '--data-dir', str(data), '--work-dir', str(local / 'work'),
            '--out-dir', str(local / 'output'), '--split', split,
            '--validator', str(REPO / 'student_resource/utils/validate_submission.py'),
            '--dense', '--dense-batch', str(batch), '--workers', str(workers),
            '--feat-chunk', '1000000', '--join-budget-rows', '20000000',
            '--max-train-rows', '30000000', '--dev-frac', '0.03' if mode == 'dev' else '1.0']
    return args


def run_logged(cmd, path, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as log:
        p = subprocess.Popen(cmd, cwd=BER, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        try:
            for line in p.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            rc = p.wait()
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
        if rc:
            raise subprocess.CalledProcessError(rc, cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--local', type=Path, required=True)
    ap.add_argument('--drive', type=Path, required=True)
    ap.add_argument('--mode', choices=['dev', 'full'], default='full')
    ap.add_argument('--batch', type=int, default=512)
    a = ap.parse_args()
    if a.batch < 1:
        ap.error('--batch must be positive')
    import psutil
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Select a T4 GPU runtime. This pipeline does not support TPU.')
    ram = psutil.virtual_memory().total / GIB
    a.local.mkdir(parents=True, exist_ok=True)
    if a.mode == 'full' and ram < 28:
        raise RuntimeError(f'Only {ram:.1f} GiB system RAM. Full mode requires at least 28 GiB as a conservative preflight (not a guarantee). Select High-RAM if available; dev mode can test setup.')
    print(f'GPU: {torch.cuda.get_device_name(0)} | System RAM: {ram:.1f} GiB', flush=True)
    hashes = {f: digest(a.data / f) for f in FILES}
    h = hashlib.sha256()
    for folder in (BER / 'src', REPO / 'colab'):
        for p in sorted(folder.glob('*.py')):
            h.update(str(p.relative_to(REPO)).encode())
            h.update(p.read_bytes())
    # Batch size is deliberately excluded: it can be reduced after CUDA OOM.
    signature = {'data': hashes, 'code': h.hexdigest(), 'mode': a.mode,
                 'profile': 'colab-t4-v1', 'python': sys.version.split()[0]}
    cp = Checkpoints(a.local, a.drive, signature)
    cp.load()  # Reject incompatible cache before changing local files.
    required = (60 if a.mode == 'full' else 10) * GIB
    # Include existing work in capacity; a resume may already occupy much of it.
    occupied = sum(p.stat().st_size for sub in ('work', 'output')
                   for p in (a.local / sub).rglob('*') if p.is_file())
    if shutil.disk_usage(a.local).free + occupied < required:
        raise RuntimeError(f'Need {(required/GIB):.0f} GiB local capacity for this run. Free disk plus existing run files is insufficient.')
    completed = cp.restore()
    workers = min(4, os.cpu_count() or 2)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = str(workers)
    env.update(CUDA_MODULE_LOADING='LAZY', KMP_DUPLICATE_LIB_OK='TRUE',
               PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    stages = [('prepare_train', 'prepare', 'train'),
              ('features_train', 'features', 'train'), ('train', 'train', 'train')]
    if a.mode == 'full':
        stages += [('prepare_test', 'prepare', 'test'),
                   ('features_test', 'features', 'test'), ('predict', 'predict', 'test')]
    atomic_json(a.drive / 'run_settings.json', dict(signature=signature, batch=a.batch))
    for name, cmd, split in stages:
        if name in completed:
            print('SKIP completed stage:', name, flush=True)
            continue
        print('\nSTART:', name, flush=True)
        atomic_json(a.local / 'stage_running.json', {'stage': name})
        log = a.local / 'logs' / f'{name}.log'
        try:
            run_logged(command(sys.executable, a.data, a.local, cmd, split, a.mode, a.batch, workers), log, env)
            cp.save(completed + [name])
            completed.append(name)
        finally:
            if log.exists():
                atomic_copy(log, a.drive / 'logs' / log.name)
        print('SAVED TO DRIVE:', name, flush=True)
    report = a.local / 'work/models/report.json'
    if report.exists():
        atomic_copy(report, a.drive / 'report.json')
        print(report.read_text(), flush=True)
    if a.mode == 'full':
        out = a.local / 'output'
        # Explicit validation is mandatory even when resuming an already completed prediction.
        run_logged([sys.executable, str(REPO / 'student_resource/utils/validate_submission.py'),
                    '--matching', str(out / 'matching_results.tsv'),
                    '--candidate', str(out / 'candidate_pairs.tsv'), '--test-dir', str(a.data / 'test')],
                   a.local / 'logs/validation.log', env)
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            atomic_copy(out / name, a.drive / 'output' / name)
        archive = a.local / 'ber_submission.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
                z.write(out / name, 'output/' + name)
        atomic_copy(archive, a.drive / archive.name)
        atomic_copy(a.local / 'logs/validation.log', a.drive / 'logs/validation.log')
        print('DONE. Validated TSVs and ber_submission.zip saved to', a.drive, flush=True)
    else:
        print('DEV COMPLETE. This is a 3% training check; no test submission was generated.', flush=True)


if __name__ == '__main__':
    main()
