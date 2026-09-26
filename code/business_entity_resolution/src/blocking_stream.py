"""Disk-backed keys and streamed BM25; frequencies use complete country partitions."""
from pathlib import Path
from array import array
import tempfile
import numpy as np
import scipy.sparse as sp
from low_memory import STEP, release, spill


class RawWriter:
    def __init__(self,path,dtype):
        self.path=Path(path);self.dtype=np.dtype(dtype);self.file=self.path.open('wb');self.n=0
    def add(self,a):
        a=np.asarray(a,dtype=self.dtype); a.tofile(self.file);self.n+=len(a)
    def finish(self):
        self.file.close()
        return np.memmap(self.path,dtype=self.dtype,mode='r',shape=(self.n,)) if self.n else np.empty(0,self.dtype)


def keys_disk(Q,I,cfg,directory):
    from blocking import _topk_rare, _cross_keys, _single_keys, _M28
    nq,ni=len(Q),len(I)
    def family(col,variants,first=None):
        # First-occurrence codes and record-local code order match pandas.factorize
        # followed by unique(rec*(U+1)+code) in the original implementation.
        vocab={};df=[];rw=RawWriter(Path(directory)/(col+'r'),np.int64);cw=RawWriter(Path(directory)/(col+'c'),np.int64)
        for offset,D in ((0,Q),(nq,I)):
            for s in range(0,len(D),STEP):
                rr=[];cc=[]
                for j,text in enumerate(D[col].iloc[s:s+STEP]):
                    codes=[]
                    for tok in str(text).split()[:first]:
                        c=vocab.get(tok)
                        if c is None: c=len(vocab);vocab[tok]=c;df.append(0)
                        codes.append(c)
                    for c in sorted(set(codes)):
                        rr.append(offset+s+j);cc.append(c);df[c]+=1
                rw.add(rr);cw.add(cc)
        if len(vocab)>=_M28: raise OverflowError('token vocabulary exceeds 2^28')
        del vocab
        rec,code=rw.finish(),cw.finish();df=np.asarray(df,np.int64)
        out={}
        for name,k in variants:
            if k is None: out[name]=(rec,code);continue
            ro=RawWriter(Path(directory)/(name+'r'),np.int64);co=RawWriter(Path(directory)/(name+'c'),np.int64)
            for start in range(0,nq+ni,STEP):
                a,b=np.searchsorted(rec,[start,start+STEP])
                r,c=_topk_rare(rec[a:b],code[a:b],df,k);ro.add(r);co.add(c)
            out[name]=(ro.finish(),co.finish())
        release(rec,code)
        return out
    fam={}
    for col,variants,first in [
        ('n_core',[('n',cfg.name_topk),('np',cfg.name_topk+1)],None),
        ('a_alpha',[('a',cfg.addr_topk),('a3',max(1,cfg.addr_topk-1))],None),
        ('n_skel',[('ns',cfg.name_topk)],None),('n_phon',[('ph',cfg.name_topk)],None),
        ('a_skel',[('as',max(1,cfg.addr_topk-1))],None),
        ('a_nums',[('u',None)],cfg.num_topk),('n_concat',[('c',None)],None)]:
        fam.update(family(col,variants,first))
    # Same insertion order as build_keys (including PH before AN).
    for t,left,right,symmetric in [(0,'n','a',False),(1,'ns','as',False),(6,'ph','as',False),
                                    (2,'u','a3',False),(3,'n','u',False),(4,'np','np',True),(5,'c',None,False)]:
        writers=[RawWriter(Path(directory)/f'key{t}_{i}',dt) for i,dt in enumerate([np.int64,np.uint64,np.int64,np.uint64])]
        ar,ac=fam[left]
        for start in range(0,nq+ni,STEP):
            a,b=np.searchsorted(ar,[start,start+STEP]); rr,cc=ar[a:b],ac[a:b]
            if right is None: r,key=_single_keys(rr,cc,t)
            else:
                br,bc=fam[right]; c,d=np.searchsorted(br,[start,start+STEP])
                r,key=_cross_keys(rr,cc,br[c:d],bc[c:d],t,symmetric)
            mask=r<nq
            for w,v in zip(writers,[r[mask],key[mask],r[~mask]-nq,key[~mask]]):w.add(v)
        yield t,tuple(w.finish() for w in writers)


def bm25_stream(q_names,q_addrs,i_names,i_addrs,k,chunk=0,max_df_ratio=.20):
    from blocking import _bm25_tokenize
    from collections import Counter
    nq,ni=len(q_names),len(i_names);k=min(k,ni)
    if not nq or not k: return np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0,np.float32)
    with tempfile.TemporaryDirectory(prefix='ber-bm25-') as td:
        iw=RawWriter(Path(td)/'indices',np.int32);vw=RawWriter(Path(td)/'tf',np.float32)
        vocab={};dfs=[];ptr=np.zeros(ni+1,np.int64);dl=np.zeros(ni,np.float32)
        for start in range(0,ni,STEP):
            indices=[];values=[]
            for row,(name,addr) in enumerate(zip(i_names[start:start+STEP],i_addrs[start:start+STEP]),start):
                counts=Counter(_bm25_tokenize(name,addr)); pairs=[]
                for tok,count in counts.items():
                    j=vocab.get(tok)
                    if j is None:j=len(vocab);vocab[tok]=j;dfs.append(0)
                    dfs[j]+=1;pairs.append((j,count))
                pairs.sort()
                indices.extend(j for j,_ in pairs);values.extend(v for _,v in pairs)
                ptr[row+1]=ptr[row]+len(pairs);dl[row]=sum(counts.values())
            iw.add(indices);vw.add(values)
        ix,vals=iw.finish(),vw.finish();df=np.asarray(dfs,np.int64);nv=len(vocab)
        avgdl=max(1e-9,float(dl.mean()));k1,b=1.2,.75
        idf=np.log(1+(ni-df+.5)/(df+.5)).astype(np.float32)
        idf=np.where(df<=max(50,int(max_df_ratio*ni)),idf,0).astype(np.float32)
        ww=RawWriter(Path(td)/'weight',np.float32)
        for start in range(0,ni,STEP):
            end=min(ni,start+STEP);a,bp=ptr[start],ptr[end]
            rows=np.repeat(np.arange(start,end),np.diff(ptr[start:end+1]))
            denom=vals[a:bp]+k1*(1-b+b*dl[rows]/avgdl)
            ww.add(idf[ix[a:bp]]*vals[a:bp]*(k1+1)/denom)
        weights=ww.finish();D=sp.csr_matrix((weights,ix,ptr),shape=(ni,nv),copy=False)
        # CSR conversion of the transpose is needed for the same sparse summation order.
        Dt=D.T.tocsr();Dt.eliminate_zeros()
        del D;release(ix,vals,weights)
        # Original 400 x millions dense scores plus argpartition can exceed RAM.
        chunk=max(1,min(chunk or 400,nq,(128*2**20)//max(1,ni*16)))
        oq,oi,ov=[],[],[]
        for start in range(0,nq,chunk):
            qr=[];qc=[]
            for row,(name,addr) in enumerate(zip(q_names[start:start+chunk],q_addrs[start:start+chunk])):
                for tok in set(_bm25_tokenize(name,addr)):
                    j=vocab.get(tok)
                    if j is not None and idf[j]>0:qr.append(row);qc.append(j)
            Q=sp.csr_matrix((np.ones(len(qr),np.float32),(qr,qc)),shape=(min(chunk,nq-start),nv))
            S=(Q@Dt).toarray()
            part=np.argpartition(-S,k-1,axis=1)[:,:k];rr=np.arange(len(S))[:,None]
            top=part[rr,np.argsort(-S[rr,part],axis=1)];sc=S[rr,top];r,c=np.nonzero(sc>0)
            oq.append((start+r).astype(np.int32));oi.append(top[r,c].astype(np.int32));ov.append(sc[r,c].astype(np.float32))
        return np.concatenate(oq),np.concatenate(oi),np.concatenate(ov)
