"""Reproduce the 1.5M-row Dataset memory test in separate Python processes.

python colab/tests/memory_probe.py original /tmp/ber-memory-probe
python colab/tests/memory_probe.py low /tmp/ber-memory-probe
Compare printed peak RSS and SHA256 of original.bin / low.bin.
This isolates matrix ingestion; it does not measure full-pipeline RAM.
"""
import sys,os,json,resource,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code/business_entity_resolution/src'))
from config import Config
from features import ALL_FEATURES,feat_path
from low_memory import release,training_sequence
import lightgbm as lgb
root=Path(sys.argv[2]) if len(sys.argv)>2 else Path('/tmp/ber-memory-probe');root.mkdir(exist_ok=True);(root/'train').mkdir(exist_ok=True)
mode=sys.argv[1];cfg=Config(work_dir=str(root),workers=2)
n=750000;cols=len(ALL_FEATURES);meta={}
for j,ck in enumerate(['a','b']):
 path=feat_path(cfg,'train',ck)
 if not Path(path).exists():
  mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(n,cols));rng=np.random.default_rng(j)
  for s in range(0,n,25000):mm[s:s+25000]=rng.normal(size=(min(25000,n-s),cols));release(mm)
  del mm
 rng=np.random.default_rng(j)
 meta[ck]=dict(use=True,fold=np.arange(n,dtype=np.int8)%3,y=rng.integers(0,2,size=n,dtype=np.int8),
               qg=np.arange(n,dtype=np.int32),P=dict(bscore=np.ones(n,np.float32),brank=np.zeros(n,np.int16),dcos=np.ones(n,np.float32)))
t=time.time()
if mode=='low':X,y,fold=training_sequence(meta,cfg,'stage1')
else:
 Xs=[np.asarray(np.load(feat_path(cfg,'train',ck),mmap_mode='r')[np.arange(n)],dtype=np.float32) for ck in ['a','b']]
 X=np.concatenate(Xs);del Xs;y=np.concatenate([m['y'] for m in meta.values()])
ds=lgb.Dataset(X,label=y,feature_name=ALL_FEATURES,params={'max_bin':255,'verbose':-1,'num_threads':2})
ds.construct();(root/(mode+'.bin')).unlink(missing_ok=True);ds.save_binary(str(root/(mode+'.bin')))
print(json.dumps(dict(mode=mode,rows=ds.num_data(),features=ds.num_feature(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seconds=time.time()-t)))
