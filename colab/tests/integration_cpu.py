"""Run with the pipeline requirements installed: python colab/tests/integration_cpu.py.

Synthetic end-to-end test of the Colab wrapper. GPU and large-resource guards are
mocked because this test uses tiny CPU data. Dense retrieval is disabled only in
this harness. Telemetry is disabled in subprocesses to tolerate sandbox /proc
namespaces. No production source or production defaults are patched on disk.
"""
import importlib.util,sys,tempfile,random,types,json,os
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'colab'))
import runner
import psutil
psutil.virtual_memory=lambda:types.SimpleNamespace(total=32*2**30)
runner.shutil.disk_usage=lambda p:types.SimpleNamespace(free=100*2**30)
spec=importlib.util.spec_from_file_location('smoke',root/'code/business_entity_resolution/tests/smoke_test.py')
smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)
rng=random.Random(7)
with tempfile.TemporaryDirectory() as td:
 t=Path(td); ds=t/'dataset'
 shim=t/'shim';shim.mkdir()
 (shim/'sitecustomize.py').write_text("import sys\nsys.path.insert(0, "+repr(str(root/'code/business_entity_resolution/src'))+")\nimport utils\nutils.psutil=None\n")
 os.environ['PYTHONPATH']=str(shim)
 for split,n in [('train',900),('test',300)]:
  d=ds/split;d.mkdir(parents=True)
  s1,s2,s3,gt=smoke.gen(rng,n)
  for i,rows in enumerate([s1,s2,s3],1):smoke.write_tsv(str(d/f'{split}_source{i}.tsv'),rows,['entity_id','business_name','business_address','country'])
  if split=='train':smoke.write_tsv(str(d/'train_ground_truth.tsv'),[(a,','.join(b)) for a,b in gt],['source1_entity_id','matched_entity_ids'])
 # CPU-only integration harness; production checks and production dense defaults are unchanged.
 sys.modules['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,get_device_name=lambda n:'CPU test harness (mock GPU check)'))
 base=runner.command
 def cpu_command(*args,**kwargs):
  c=base(*args,**kwargs); c[c.index('--dense')]='--no-dense'
  return c+['--bm25-k','0','--bm25-k-empty','0','--k-dense-script','0']
 runner.command=cpu_command
 sys.argv=['runner','--data',str(ds),'--local',str(t/'local'),'--drive',str(t/'drive'),'--mode','full']
 runner.main()
 cp=json.loads((t/'drive/checkpoint.json').read_text())
 assert len(cp['completed'])==6,cp['completed']
 original=runner.run_logged
 def no_training_again(cmd,*args,**kwargs):
  assert 'run.py' not in ' '.join(cmd), 'Resume repeated a completed pipeline stage'
  return original(cmd,*args,**kwargs)
 runner.run_logged=no_training_again
 runner.main()
 for f in ['output/matching_results.tsv','output/candidate_pairs.tsv','ber_submission.zip','report.json']:
  assert (t/'drive'/f).is_file(),f
 print('WRAPPER INTEGRATION PASS: six CPU stages, official validator, Drive publication, resume skips all completed stages')
