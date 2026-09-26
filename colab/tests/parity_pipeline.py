"""Full synthetic regression, including FAISS, both GBM stages and TSV equality.

Uses deterministic synthetic embeddings, not downloaded E5 weights. Set BER_BASELINE
to a checkout of 1be6c0646cb5d1af93424fe9b5838c4a9684af67 for the historical control.
BER_PARITY_WORK must be a fresh temporary directory. Needs pipeline + faiss packages.
"""
import sys,importlib.util,random,os,subprocess,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]; old=Path(os.environ.get('BER_BASELINE',str(root)));out=Path(os.environ.get('BER_PARITY_WORK','/tmp/ber-parity-dense'))
if out.exists() and any(out.iterdir()):raise SystemExit('Use a fresh BER_PARITY_WORK directory to prevent stale cache comparisons')
out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('smoke',root/'code/business_entity_resolution/tests/smoke_test.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=random.Random(43)
for split,n in [('train',900),('test',250)]:
 d=out/'data'/split;d.mkdir(parents=True,exist_ok=True);s1,s2,s3,gt=m.gen(rng,n)
 for i,rows in enumerate([s1,s2,s3],1):m.write_tsv(str(d/f'{split}_source{i}.tsv'),rows,['entity_id','business_name','business_address','country'])
 if split=='train':m.write_tsv(str(d/'train_ground_truth.tsv'),[(a,','.join(b)) for a,b in gt],['source1_entity_id','matched_entity_ids'])
shim=out/'shim';shim.mkdir(exist_ok=True)
(shim/'sitecustomize.py').write_text('import sys,os\nsys.path.insert(0,os.environ["BER_SRC"])\nimport utils,dense\nutils.psutil=None\ndense._pick_device=lambda:("cpu",False)\n')
for label,repo in [('original',old),('low',root)]:
 env=dict(os.environ,PYTHONPATH=str(shim),BER_SRC=str(repo/'code/business_entity_resolution/src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 cmd=[sys.executable,str(repo/'code/business_entity_resolution/src/run.py'),'all','--data-dir',str(out/'data'),'--work-dir',str(out/label),'--out-dir',str(out/(label+'_output')),'--workers','2','--dense','--keep-intermediates','--lgb-rounds','80','--min-leaf','10','--feat-chunk','25000','--join-budget-rows','500000']
 if label=='low':cmd+=['--low-memory']
 prep=cmd.copy();prep[2]='prepare'
 with open(out/(label+'.log'),'w') as f:
  subprocess.run(prep,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  preptest=prep+['--split','test'];subprocess.run(preptest,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 import numpy as np,pyarrow.parquet as pq
 for split in ['train','test']:
  for tag in ['s1','idx']:
   n=pq.ParquetFile(out/label/split/(tag+'.parquet')).metadata.num_rows
   x=np.random.default_rng(n).normal(size=(n,384)).astype(np.float32)
   x/=np.linalg.norm(x,axis=1,keepdims=True)
   np.save(out/label/split/('emb_'+tag+'.npy'),x.astype(np.float16))
 cmd+=['--max-train-rows','3000']
 with open(out/(label+'.log'),'a') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
 if r.returncode:print((out/(label+'.log')).read_text()[-6000:]);sys.exit(r.returncode)
import numpy as np
for split in ['train','test']:
 for p in (out/'original'/split).glob('*.npz'):
  a=np.load(p);b=np.load(out/'low'/split/p.name)
  for k in a.files:np.testing.assert_array_equal(a[k],b[k],err_msg=f'{split}/{p.name}:{k}')
 for p in (out/'original'/split).glob('feat_*.npy'):
  np.testing.assert_array_equal(np.load(p),np.load(out/'low'/split/p.name),err_msg=p.name)
a=json.loads((out/'original/models/report.json').read_text());b=json.loads((out/'low/models/report.json').read_text());assert a==b,(a,b)
for f in ['matching_results.tsv','candidate_pairs.tsv']:assert (out/'original_output'/f).read_bytes()==(out/'low_output'/f).read_bytes(),f
print('EXACT PARITY PASS',a['holdout_stage2_f05'],flush=True)
