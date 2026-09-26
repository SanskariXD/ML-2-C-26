"""Behavioral regression checks for the 12.7 GiB execution path (CPU, no dataset)."""
import os, sys, tempfile, unittest, pickle
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/business_entity_resolution/src'))
from config import Config
from prepare import prepare,load_split,load_gt
from normalize import normalize_frame
from blocking import block_partition,bm25_topk
from blocking_stream import bm25_stream
from features import PartitionFeaturizer,ALL_FEATURES,context_features,BASE_FEATURES,CONTEXT_FEATURES
from low_memory import stage2_stream,unique_disk,write_context
from stage2 import stage2_matrix,S2_FEATURES


class LowMemoryTests(unittest.TestCase):
    def test_prepare_records_partitions_labels_and_dev(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'data/train';d.mkdir(parents=True)
            head='entity_id\tbusiness_name\tbusiness_address\tcountry\n'
            (d/'train_source1.tsv').write_text(head+''.join(f'S1-{i}\tName {i}\tStreet {i}\tUS\n' for i in range(60))+'S1-1\tDuplicate\tOther\tIndia\n')
            (d/'train_source2.tsv').write_text(head+''.join(f'S2-{i}\tName {i}\tStreet\t{i}\tUS\n' for i in range(60))+'S2-0\tDuplicate\tOther\tIndia\n')
            (d/'train_source3.tsv').write_text(head+'S2-1\tCross duplicate\tOther\tIndia\nS3-0\tOrphan\t\t\nS3-1\tMissing columns\n')
            (d/'train_ground_truth.tsv').write_text('source1_entity_id\tmatched_entity_ids\n'+''.join(f'S1-{i}\tS2-{i}\n' for i in range(60))+'S1-0\tS2-0,S3-1\n')
            for frac in (1.,.3):
                cfgs=[Config(data_dir=str(d.parent),work_dir=str(Path(t)/f'{frac}_{low}'),low_memory=low,dev_frac=frac,workers=1) for low in (False,True)]
                for c in cfgs:prepare(c,'train')
                for name in ('s1','idx'):
                    a=pd.read_parquet(Path(cfgs[0].work_dir)/'train'/f'{name}.parquet');b=pd.read_parquet(Path(cfgs[1].work_dir)/'train'/f'{name}.parquet')
                    pd.testing.assert_frame_equal(a,b,check_dtype=False)
                # Dev dictionary iteration order can differ; compare pair identity.
                a=load_gt(cfgs[0]);b=load_gt(cfgs[1]);self.assertEqual(set(zip(*a)),set(zip(*b)))
                pa=load_split(cfgs[0],'train')[2];pb=load_split(cfgs[1],'train')[2]
                self.assertEqual(list(pa),list(pb))
                for k in pa:
                    for a,b in zip(pa[k],pb[k]):np.testing.assert_array_equal(a,b)
    def frames(self):
        rows=[('Apex media LLC','12 Maple Road US'),('apex media','12 maple rd'),('Cornerstone',''),('भारत मीडिया','12 Maple'),('Apex','99 Maple Road')]*8
        def make(items):
            f=pd.DataFrame(items,columns=['business_name','business_address']);f['entity_id']=[str(i) for i in range(len(f))];f['country']='US';f['src']=2
            return pd.concat([f,normalize_frame(f,1)],axis=1)
        return make(rows[:25]),make(rows)
    def test_keys_bm25_features_context_stage2(self):
        with tempfile.TemporaryDirectory() as t:
            Q,I=self.frames();c=Config(work_dir=t,workers=1,hash_features=1024,feat_chunk=7)
            a=block_partition(Q,I,c);c.low_memory=True;b=block_partition(Q,I,c)
            for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
            text=[D[col].to_numpy(object) for D in (Q,I) for col in ('business_name','business_address')]
            for a,b in zip(bm25_topk(*text,15,chunk=9),bm25_stream(*text,15,chunk=7)):np.testing.assert_array_equal(a,b)
            P=b if isinstance(b,dict) else block_partition(Q,I,c)
            P['dcos']=np.full(len(P['q']),np.nan,np.float32);sl=slice(None)
            c.low_memory=False;f=PartitionFeaturizer(Q,I,c);A=f.chunk(P['q'],P['i'],P,sl)
            c.low_memory=True;f=PartitionFeaturizer(Q,I,c);B=f.chunk(P['q'],P['i'],P,sl)
            np.testing.assert_array_equal(A,B)
            X=np.zeros((len(A),len(ALL_FEATURES)),np.float16);X[:,:len(BASE_FEATURES)]=A
            nt=A[:,BASE_FEATURES.index('n_tset')];at=A[:,BASE_FEATURES.index('a_tset')]
            ctx=context_features(P['q'].astype(np.int64),P['i'].astype(np.int64),nt,at,P['bscore'])
            Y=X.copy();X[:,len(BASE_FEATURES):]=np.column_stack([ctx[k] for k in CONTEXT_FEATURES])
            write_context(Y,P['q'],P['i'],nt,at,P['bscore']);np.testing.assert_array_equal(X,Y)
            p=np.random.default_rng(2).random(len(A)).astype(np.float32)
            names=I['n_core'].to_numpy(object);addrs=I['a_full'].to_numpy(object)
            A=np.zeros((len(p),len(S2_FEATURES)),np.float16);B=A.copy()
            stage2_matrix(P['q'],P['i'],p,X,names,addrs,out=A)
            stage2_stream(P['q'],P['i'],p,X,names,addrs,out=B);np.testing.assert_array_equal(A,B)
    def test_dense_full_index_and_duplicate_ties(self):
        import faiss
        faiss.omp_set_num_threads(2)
        rng=np.random.default_rng(14);I=rng.normal(size=(2051,64)).astype(np.float32);Q=rng.normal(size=(259,64)).astype(np.float32)
        faiss.normalize_L2(I);faiss.normalize_L2(Q);I=I.astype(np.float16);Q=Q.astype(np.float16)
        for duplicates in (False,True):
            if duplicates:I[:100]=I[0];Q[:3]=I[0]
            a=I.astype(np.float32);b=Q.astype(np.float32);faiss.normalize_L2(a);faiss.normalize_L2(b)
            index=faiss.IndexFlatIP(a.shape[1]);index.add(a);v,ix=index.search(b,50)
            from dense_exact_worker import exact_search
            with tempfile.TemporaryDirectory() as td:
                exact_search(Q,I,np.arange(len(Q)),np.arange(len(I)),50,td)
                ii=np.load(Path(td)/'i.npy');ss=np.load(Path(td)/'s.npy')
            np.testing.assert_array_equal(ix,ii.reshape(ix.shape))
            np.testing.assert_array_equal(v,ss.reshape(v.shape))
    def test_disk_dedup_preserves_encoder_order(self):
        from dense import _unique_order
        values=['abc','def','abc','भारत','','def']*5000
        a,ai=_unique_order(values);b,bi=unique_disk(values)
        self.assertEqual(a,b[:]);np.testing.assert_array_equal(ai,bi)

if __name__=='__main__':unittest.main()
