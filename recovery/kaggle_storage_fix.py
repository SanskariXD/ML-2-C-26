"""Patch a stopped BER checkout to retain unique embeddings + an exact row lookup.
Run with --repo /path/to/business_entity_resolution --work /kaggle/working/work_full.
Does not start training, remove completed embeddings, or change model settings.
"""
from pathlib import Path
import argparse
import ast
import os
import shutil

HELPER = r'''"""Lossless embedding indirection and disk allocation checks."""
from pathlib import Path
import json
import os
import shutil
import numpy as np


def safe_open_memmap(filename, mode='r+', dtype=None, shape=None, **kwargs):
    if mode != 'w+':
        return np.lib.format.open_memmap(filename, mode=mode, dtype=dtype, shape=shape, **kwargs)
    p = Path(filename)
    needed = int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize + 4096
    reclaim = p.stat().st_blocks * 512 if p.exists() else 0
    available = shutil.disk_usage(p.parent).free + reclaim
    if available < needed + 256 * 2**20:
        raise RuntimeError(
            f'DISK SPACE: {p.name} needs {needed/2**30:.2f} GiB plus 0.25 GiB reserve; '
            f'{available/2**30:.2f} GiB available. Existing embedding checkpoints are kept. '
            'Move the work directory to a filesystem with more free space before resuming.')
    mm = np.lib.format.open_memmap(p, mode=mode, dtype=dtype, shape=shape, **kwargs)
    try:
        # Reserve physical blocks before writing through mmap: ENOSPC becomes an exception.
        if hasattr(os, 'posix_fallocate'):
            import errno
            with p.open('r+b') as f:
                try:
                    os.posix_fallocate(f.fileno(), 0, p.stat().st_size)
                except OSError as e:
                    if e.errno not in (errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL):
                        raise
    except OSError:
        mm._mmap.close()
        p.unlink(missing_ok=True)
        raise
    return mm


def embedding_exists(path):
    return Path(path).exists() or Path(str(path) + '.indexed.json').exists()


def save_indexed(path, unique_path, inverse):
    """Keep source intact; atomically publish a small inverse map, then a manifest."""
    p = Path(path)
    base = np.load(unique_path, mmap_mode='r')
    if base.dtype != np.float16 or base.ndim != 2:
        raise ValueError('Expected an fp16 embedding matrix')
    for start in range(0, len(inverse), 100_000):
        block = np.asarray(inverse[start:start+100_000])
        if block.size and (block.min() < 0 or block.max() >= len(base)):
            raise ValueError('Invalid inverse row index; original checkpoint retained')
    dest = Path(str(p) + '.inverse.npy')
    temp = Path(str(dest) + '.partial')
    if shutil.disk_usage(p.parent).free < len(inverse)*4 + 16*2**20:
        raise RuntimeError('Not enough space for the inverse row lookup; checkpoint retained')
    # Ordinary writes report disk errors instead of causing an mmap SIGBUS.
    with temp.open('wb') as f:
        np.lib.format.write_array_header_1_0(f, dict(descr='<i4', fortran_order=False, shape=(len(inverse),)))
        for start in range(0, len(inverse), 100_000):
            f.write(np.asarray(inverse[start:start+100_000], dtype='<i4').tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, dest)
    manifest = dict(version=1, unique=Path(unique_path).name, inverse=dest.name,
                    shape=[len(inverse), base.shape[1]])
    target = Path(str(p) + '.indexed.json')
    staging = Path(str(target) + '.partial')
    with staging.open('w') as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(staging, target)


class IndexedEmbeddings:
    def __init__(self, path):
        self.filename = str(path)
        p = Path(path)
        meta = json.loads(Path(str(p)+'.indexed.json').read_text())
        self.unique = np.load(p.parent / meta['unique'], mmap_mode='r')
        self.inverse = np.load(p.parent / meta['inverse'], mmap_mode='r')
        self.shape = tuple(meta['shape'])
        self.dtype = self.unique.dtype
        self.ndim = 2
        if self.shape != (len(self.inverse), self.unique.shape[1]):
            raise ValueError('Incomplete indexed embedding cache')
    def __len__(self):
        return self.shape[0]
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.unique[(self.inverse[key[0]],) + key[1:]]
        return self.unique[self.inverse[key]]
    def release_pages(self):
        import mmap
        for a in (self.unique, self.inverse):
            try:
                a._mmap.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError, ValueError):
                pass


def load_embedding(path):
    if Path(str(path)+'.indexed.json').exists():
        return IndexedEmbeddings(path)
    return np.load(path, mmap_mode='r')
'''


def transform_dense(source):
    if '# BER_COMPACT_STORAGE_V1' in source:
        return source
    anchor = 'from utils import LOG, timed'
    if source.count(anchor) != 1:
        raise RuntimeError('Unrecognized dense.py; no files changed')
    source = source.replace(anchor, anchor + '\n# BER_COMPACT_STORAGE_V1\nfrom compact_storage import embedding_exists, load_embedding, save_indexed, safe_open_memmap')
    old = 'if not (os.path.exists(path) and not cfg.force):'
    if source.count(old) != 1:
        raise RuntimeError('Unrecognized embedding cache check; no files changed')
    source = source.replace(old, 'if not (embedding_exists(path) and not cfg.force):\n            from pathlib import Path\n            Path(path + ".indexed.json").unlink(missing_ok=True)')
    old = 'out.append(np.load(path, mmap_mode="r"))'
    if source.count(old) != 1:
        raise RuntimeError('Unrecognized embedding loader; no files changed')
    source = source.replace(old, 'out.append(load_embedding(path))')
    a = source.index('    # Scatter unique')
    b = source.index('\ndef _knn_faiss', a)
    source = source[:a] + '''    # No second N x 384 file: preserve exact values through a row lookup.
    save_indexed(path, tmp_u, inv)
    LOG.info("    dense compact cache ready: %s (%d rows; scatter skipped)", path, n)
    del mm_u

''' + source[b:]
    source = source.replace('np.lib.format.open_memmap(', 'safe_open_memmap(')
    return source


def patch_repo(repo):
    src = repo / 'src'
    changes = {src/'dense.py': transform_dense((src/'dense.py').read_text()), src/'compact_storage.py': HELPER}
    worker = src/'dense_exact_worker.py'
    if worker.exists():
        s = worker.read_text()
        if 'from compact_storage import load_embedding' not in s:
            s += '\n'  # import must precede the __main__ block
            s = s.replace("if __name__=='__main__':", "from compact_storage import load_embedding\n\nif __name__=='__main__':")
            s = s.replace("np.load(a.query,mmap_mode='r'),np.load(a.index,mmap_mode='r')", 'load_embedding(a.query),load_embedding(a.index)')
        if 'load_embedding(a.query)' not in s:
            raise RuntimeError('Unrecognized dense worker; no files changed')
        changes[worker] = s
    low = src/'low_memory.py'
    if low.exists():
        s = low.read_text()
        if 'a.release_pages()' not in s:
            s = s.replace('    for a in arrays:\n', '    for a in arrays:\n        if hasattr(a, "release_pages"):\n            a.release_pages()\n            continue\n', 1)
        changes[low] = s
    for name in ('features.py', 'pipeline.py'):
        p = src/name
        s = p.read_text()
        if 'from compact_storage import safe_open_memmap' not in s:
            s = s.replace('import numpy as np', 'import numpy as np\nfrom compact_storage import safe_open_memmap', 1)
            s = s.replace('np.lib.format.open_memmap(', 'safe_open_memmap(')
        changes[p] = s
    for p, s in changes.items():
        ast.parse(s, filename=str(p))
    for p, s in changes.items():
        if p.exists() and p.read_text() == s:
            continue
        backup = Path(str(p)+'.before_storage_fix')
        if p.exists() and not backup.exists():
            shutil.copy2(p, backup)
        temp = Path(str(p)+'.patching')
        temp.write_text(s)
        os.replace(temp, p)
    print('PATCH INSTALLED:', repo)


def different_filesystem(a, b):
    return Path(a).stat().st_dev != Path(b).stat().st_dev


def relocate_work(work, destination, reserve_gib=50):
    """Copy and verify checkpoints before replacing the old path with a symlink."""
    import hashlib
    work, destination = Path(work), Path(destination)
    if work.is_symlink():
        print('Work already redirected:', work.resolve())
        return False
    if not work.is_dir() or not different_filesystem(work, destination.parent):
        return False
    files = [p for p in work.rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in work.rglob('*')):
        raise RuntimeError('Work directory contains symlinks; automatic move skipped for safety')
    size = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(destination.parent).free < size + reserve_gib*2**30:
        return False
    staging = Path(str(destination)+'.copying')
    backup = work.with_name(work.name+'.before_scratch_move')
    if any(p.exists() for p in (destination, staging, backup)):
        raise RuntimeError('A previous scratch move exists; keep its files and inspect before retrying')
    print(f'Copying {size/2**30:.2f} GiB to {destination}; original retained until verification.', flush=True)
    shutil.copytree(work, staging)
    def digest(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8*2**20), b''):
                h.update(block)
        return h.digest()
    for p in files:
        copied = staging / p.relative_to(work)
        if p.stat().st_size != copied.stat().st_size or digest(p) != digest(copied):
            raise RuntimeError(f'Copy verification failed: {p}; original files untouched')
    os.rename(staging, destination)
    os.rename(work, backup)
    try:
        work.symlink_to(destination, target_is_directory=True)
    except BaseException:
        os.rename(backup, work)
        raise
    shutil.rmtree(backup)
    print('Verified work moved; original WORK path preserved through a symlink.', flush=True)
    print('Scratch checkpoints last only for this runtime. Final output stays in /kaggle/working/output.')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path)
    ap.add_argument('--work', type=Path, default=Path('/kaggle/working/work_full'))
    ap.add_argument('--try-scratch', action='store_true', help='Use /tmp only if it is a different filesystem with at least 50 GiB free after copying work')
    a = ap.parse_args()
    # Refuse to patch or clean a live pipeline, including an orphan subprocess.
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'cmdline']):
            cmd = proc.info['cmdline'] or []
            if any(Path(x).name == 'run.py' for x in cmd):
                raise RuntimeError(f'Pipeline PID {proc.pid} is still running. Stop that process first; keep the session.')
    except ImportError:
        raise RuntimeError('Install psutil before applying this recovery patch')
    repo = a.repo
    if repo is None:
        matches = sorted({p.parent.parent.resolve() for p in Path('/kaggle/working').rglob('src/dense.py')})
        if not matches:
            matches = sorted({p.parent.parent.resolve() for p in Path('/kaggle/input').rglob('src/dense.py')})
        if len(matches) != 1:
            raise RuntimeError(f'Found {matches}; supply --repo with the directory containing src/dense.py')
        repo = matches[0]
    # Clean only abandoned scatter outputs backed by a resumable unique checkpoint.
    for split in ('train', 'test'):
        for tag in ('s1', 'idx'):
            p = a.work/split/f'emb_{tag}.npy'
            unique = Path(str(p)+'.uniq.tmp.npy')
            progress = Path(str(unique)+'.progress')
            partial = Path(str(p)+'.tmp.npy')
            if unique.exists() and progress.exists() and partial.exists():
                if not progress.read_text().strip().isdigit():
                    raise RuntimeError('Invalid checkpoint progress; no scatter file removed')
                size = partial.stat().st_blocks*512/2**30
                partial.unlink()
                print(f'Removed incomplete scatter only: {partial.name} ({size:.2f} GiB)')
    if str(repo.resolve()).startswith('/kaggle/input/'):
        target = Path('/kaggle/working/ber_runtime_patched')
        target.mkdir(exist_ok=True)
        for folder in ('src', 'scripts'):
            shutil.copytree(repo/folder, target/folder, dirs_exist_ok=True)
        repo = target
        print('Copied read-only input code into a writable runtime directory.')
    patch_repo(repo)
    print('\nStorage available in THIS session:')
    for path in ('/kaggle/working', '/tmp'):
        p = Path(path)
        if p.exists():
            d = shutil.disk_usage(p)
            print(f'{path}: free={d.free/2**30:.2f} GiB, total={d.total/2**30:.2f} GiB, device={p.stat().st_dev}')
    if a.try_scratch and a.work.exists():
        moved = relocate_work(a.work, Path('/tmp/ber-kaggle-work'))
        if not moved:
            print('No qualifying larger scratch filesystem: compact storage installed, full-run capacity remains unverified.')
    import json
    Path('/kaggle/working/ber_storage_recovery.json').write_text(json.dumps(dict(repo=str(repo.resolve()), work=str(a.work.absolute()))))
    print('\nCompleted .npy files and all unique embedding checkpoints were preserved.')
    print('Resume with the SAME work directory and settings, without --force.')
    print('This fixes the duplicate embedding storage. Full-run feature storage is still measured at runtime.')


if __name__ == '__main__':
    main()
