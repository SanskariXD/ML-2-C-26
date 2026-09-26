"""Compare the standard and memory-saving paths on the SAME real training slice.

Run inside Colab against the already staged dataset. No dataset is downloaded by
this script. This is a regression gate, not an estimate of leaderboard score.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from runner import BER, REPO, FILES, digest, atomic_json


def signature(data, fraction):
    h=hashlib.sha256()
    for folder in (BER/'src',REPO/'colab'):
        for p in sorted(folder.glob('*.py')):
            h.update(str(p.relative_to(REPO)).encode());h.update(p.read_bytes())
    return dict(code=h.hexdigest(),data={f:digest(data/f) for f in FILES},fraction=fraction)


def compare_files(low, standard):
    checked=0
    for path in sorted((low/'train').glob('pairs_*.npz')):
        with np.load(path) as a,np.load(standard/'train'/path.name) as b:
            if set(a.files)!=set(b.files):raise RuntimeError(f'Candidate columns differ: {path.name}')
            for name in a.files:
                # IDs, ranks, scores and tie ordering must all match exactly.
                np.testing.assert_array_equal(a[name],b[name],err_msg=f'{path.name}/{name}')
        checked+=1
    if not checked:raise RuntimeError('No candidate partitions were compared')
    for path in sorted((low/'train').glob('feat_*.npy')):
        a=np.load(path,mmap_mode='r');b=np.load(standard/'train'/path.name,mmap_mode='r')
        if a.shape!=b.shape:raise RuntimeError(f'Feature shape differs: {path.name}')
        for start in range(0,len(a),25_000):
            np.testing.assert_array_equal(a[start:start+25_000],b[start:start+25_000],err_msg=path.name)
    return checked


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--fraction',type=float,default=.03);ap.add_argument('--batch',type=int,default=512)
    a=ap.parse_args()
    if not 0<a.fraction<1:ap.error('--fraction must lie between 0 and 1')
    sig=signature(a.data,a.fraction)
    if a.report.exists():
        old=json.loads(a.report.read_text())
        if old.get('signature')==sig and old.get('passed'):
            print('PASS: matching regression report already saved.');return
    a.work.mkdir(parents=True,exist_ok=True)
    marker=a.work/'signature.json'
    if marker.exists() and json.loads(marker.read_text())!=sig:
        raise RuntimeError('Validation work belongs to different code/data; choose a new RUN_NAME.')
    atomic_json(marker,sig)
    low=a.work/'low';ref=a.work/'reference'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def run(stage,work,lowram):
        cmd=[sys.executable,'-u',str(BER/'src/run.py'),stage,'--data-dir',str(a.data),
             '--work-dir',str(work),'--out-dir',str(work/'output'),'--split','train',
             '--workers','2','--dense','--dense-batch',str(a.batch),
             '--dev-frac',str(a.fraction),'--keep-intermediates']
        if lowram:cmd+=['--low-memory']
        subprocess.run(cmd,check=True,env=env)
    run('prepare',low,True)
    run('features',low,True)
    # Share only preparation and encoder output. Each path independently builds
    # candidates and features, then trains using its own matrix-loading path.
    (ref/'train').mkdir(parents=True,exist_ok=True)
    for name in ['s1.parquet','idx.parquet','partitions.pkl','gt.npz','emb_s1.npy','emb_idx.npy']:
        dest=ref/'train'/name
        if not dest.exists():shutil.copyfile(low/'train'/name,dest)
    run('features',ref,False)
    n=compare_files(low,ref)
    run('train',low,True);run('train',ref,False)
    l=json.loads((low/'models/report.json').read_text());r=json.loads((ref/'models/report.json').read_text())
    metrics=['holdout_stage2_f05','holdout_stage1_f05','holdout_blocking_recall','holdout_oracle_f05']
    checks={key:bool(np.isfinite(l[key]) and np.isfinite(r[key]) and l[key]>=r[key]) for key in metrics}
    unscored=[]
    for key,value in r['per_country_holdout_f05'].items():
        score=l['per_country_holdout_f05'][key]
        if np.isnan(value) and np.isnan(score):
            unscored.append(key)  # no holdout entities for this country in the slice
        else:
            checks[f'country:{key}']=bool(np.isfinite(score) and np.isfinite(value) and score>=value)
    passed=all(checks.values())
    atomic_json(a.report,dict(passed=passed,signature=sig,partitions_compared=n,
                candidates_identical=True,features_identical=True,checks=checks,unscored_countries=unscored,
                baseline=r,low_memory=l,delta=l['holdout_stage2_f05']-r['holdout_stage2_f05'],
                limitation='Training slice regression only; not a full-data/private leaderboard guarantee.'))
    if not passed:raise RuntimeError('Validation score decreased. Full run is blocked; share the regression report.')
    print('PASS: identical candidates/features and no validation score decrease. Report:',a.report)
    # Work can be reproduced. Retain both complete reports, release matrices before full run.
    for name,report in [('baseline',r),('low_memory',l)]:atomic_json(a.report.parent/(name+'_validation_metrics.json'),report)
    shutil.rmtree(a.work)

if __name__=='__main__':main()
