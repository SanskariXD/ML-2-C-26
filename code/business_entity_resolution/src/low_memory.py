"""Memory-saving execution helpers. Retrieval/training budgets are unchanged.

The original path remains available for equivalence tests. Disk arrays are temporary
execution artifacts; they are never used to shrink the candidate universe.
"""
from __future__ import annotations
import gc
import os
import tempfile
from pathlib import Path
import numpy as np

STEP = 25_000


def release(*arrays):
    """Release clean mapped pages; subsequent reads fault them back from disk."""
    import mmap
    for a in arrays:
        if isinstance(a, np.memmap):
            a.flush()
            try:
                a._mmap.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError, ValueError):
                pass


def spill(a, directory, name):
    p = Path(directory) / (name + '.npy')
    p.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(p, mode='w+', dtype=a.dtype, shape=a.shape)
    for s in range(0, len(a), STEP):
        mm[s:s+STEP] = a[s:s+STEP]
    release(mm)
    return mm


class ArrowStrings:
    """Compact Arrow storage; materialise Python strings only for requested rows."""
    def __init__(self, series, truncate=None):
        import pandas as pd
        self.data = series.reset_index(drop=True).astype('string[pyarrow]')
        if truncate is not None:
            self.data = self.data.str.slice(0, truncate)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, key):
        value = self.data.iloc[key]
        if isinstance(key, (int, np.integer)):
            return str(value)
        return value.to_numpy(dtype=object)
    def __iter__(self):
        for s in range(0, len(self), STEP):
            yield from self[s:s+STEP]


class HashRows:
    """Lazy sparse rows with IDF fitted on ALL records in the original partition."""
    def __init__(self, texts, hv, idf=None, binary=False):
        self.texts, self.hv, self.idf, self.binary = texts, hv, idf, binary
        self.shape = (len(texts), hv.n_features)
    def __getitem__(self, rows):
        from sklearn.preprocessing import normalize
        X = self.hv.transform(self.texts[rows]).tocsr()
        if self.binary:
            X.data[:] = 1.0
        else:
            X.data = (1.0 + np.log(X.data)) * self.idf[X.indices]
            normalize(X, norm='l2', copy=False)
        return X
    def counts(self):
        out = np.empty(len(self.texts), np.float32)
        for s in range(0, len(out), STEP):
            out[s:s+STEP] = np.diff(self[s:s+STEP].indptr)
        return out


def lazy_tfidf(texts_q, texts_i, kind, nf, binary=False):
    from features import _hv
    hv = _hv(kind, nf)
    idf = None
    if not binary:
        df = np.zeros(nf, np.int64)
        for texts in (texts_q, texts_i):
            for s in range(0, len(texts), STEP):
                X = hv.transform(texts[s:s+STEP]).tocsr()
                df += np.bincount(X.indices, minlength=nf)
        idf = (np.log((1.0 + len(texts_q) + len(texts_i)) / (1.0 + df)) + 1).astype(np.float32)
    return (HashRows(texts_q, hv, idf, binary), HashRows(texts_i, hv, idf, binary))


def init_featurizer(self, Q, I, cfg, emb_q, emb_i):
    from features import STR_COLS, FLAG_COLS, MAXLEN
    self.cfg, self.W = cfg, max(1, cfg.workers)
    self.q = {c: ArrowStrings(Q[c], MAXLEN) for c in STR_COLS}
    self.i = {c: ArrowStrings(I[c], MAXLEN) for c in STR_COLS}
    for side, D in ((self.q, Q), (self.i, I)):
        for c in FLAG_COLS:
            side[c] = D[c].to_numpy(dtype=np.float32)
    self.i_src3 = (I['src'].to_numpy(dtype=np.int16) == 3).astype(np.float32)
    for attr, col, kind, binary in [
        ('m_nchar','n_core','char',False), ('m_nword','n_core','word',False),
        ('m_aword','a_full','word',False), ('m_nbin','n_core','word',True),
        ('m_numbin','a_nums','word',True), ('m_fnum','a_first_num','word',True)]:
        setattr(self, attr, lazy_tfidf(self.q[col], self.i[col], kind, cfg.hash_features, binary))
    self.nb_q, self.nb_i = [x.counts() for x in self.m_nbin]
    self.nn_q, self.nn_i = [x.counts() for x in self.m_numbin]
    self.len_nq = np.fromiter(map(len, self.q['n_core']), np.float32, len(Q))
    self.len_ni = np.fromiter(map(len, self.i['n_core']), np.float32, len(I))
    self.len_ai = np.fromiter(map(len, self.i['a_full']), np.float32, len(I))
    self.ntok_q = np.fromiter((s.count(' ')+1 if s else 0 for s in self.q['n_core']), np.float32, len(Q))
    self.ntok_i = np.fromiter((s.count(' ')+1 if s else 0 for s in self.i['n_core']), np.float32, len(I))
    self.first_q = ArrowStrings(self.q['n_core'].data.str.extract(r'^([^ ]*)', expand=False))
    self.first_i = ArrowStrings(self.i['n_core'].data.str.extract(r'^([^ ]*)', expand=False))
    qc = self.q['n_core'].data.value_counts()
    ic = self.i['n_core'].data.value_counts()
    def counts(texts, lookup):
        out = np.empty(len(texts), np.float32)
        for s in range(0,len(texts),STEP):
            out[s:s+STEP] = texts.data.iloc[s:s+STEP].map(lookup).fillna(0).to_numpy(np.float32)
        return out
    self.q_name_df_s1 = counts(self.q['n_core'], qc)
    self.i_name_df_idx = counts(self.i['n_core'], ic)
    self.i_name_df_s1 = counts(self.i['n_core'], qc)
    self.emb_q, self.emb_i = emb_q, emb_i


def training_sequence(meta, cfg, tag):
    """Keep exactly the original row order and hard-negative selection, load X lazily."""
    import lightgbm as lgb
    from pipeline import _subsample_hard_neg
    from features import load_features
    from prepare import split_dir
    from utils import safe_name
    folds = cfg.stage1_folds if tag == 'stage1' else cfg.stage2_folds
    blocks, ys, fs, bs, br, dc = [], [], [], [], [], []
    for ck, m in meta.items():
        if not m['use']:
            continue
        sel = np.flatnonzero(np.isin(m['fold'], folds))
        if not len(sel):
            continue
        mm = (load_features(cfg,'train',ck) if tag=='stage1' else
              np.load(os.path.join(split_dir(cfg,'train'),f's2_{safe_name(ck)}.npy'),mmap_mode='r'))
        blocks.append((mm, sel))
        ys.append(m['y'][sel]); fs.append(m['fold'][sel])
        bs.append(m['P']['bscore'][sel]); br.append(m['P']['brank'][sel]); dc.append(m['P']['dcos'][sel])
    y, fold = np.concatenate(ys), np.concatenate(fs)
    keep = _subsample_hard_neg(y, np.concatenate(bs), np.concatenate(br), np.concatenate(dc),
                              cfg.max_train_rows, cfg.seed + (tag=='stage2'))
    del ys, fs, bs, br, dc
    offset = 0
    filtered = []
    for mm, rows in blocks:
        selected = rows[keep[offset:offset+len(rows)]]
        offset += len(rows)
        if len(selected):
            filtered.append((mm,selected))
    class MatrixSequence(lgb.Sequence):
        batch_size = 8192
        def __init__(self, parts):
            self.parts = parts
            self.ends = np.cumsum([len(rows) for _,rows in parts])
            self.shape = (int(self.ends[-1]), parts[0][0].shape[1])
        def __len__(self): return self.shape[0]
        def __getitem__(self, key):
            scalar = isinstance(key,(int,np.integer))
            ix = np.asarray([key] if scalar else (np.arange(*key.indices(len(self))) if isinstance(key,slice) else key))
            out = np.empty((len(ix),self.shape[1]),np.float64)
            bid = np.searchsorted(self.ends,ix,side='right')
            for b in np.unique(bid):
                mask = bid==b
                mm,rows = self.parts[b]
                off = 0 if b==0 else self.ends[b-1]
                out[mask] = mm[rows[ix[mask]-off]]
                release(mm)
            return out[0] if scalar else out
    return MatrixSequence(filtered), y[keep], fold[keep]


class ParquetFrame:
    """Read selected parquet rows/columns on demand, preserving global row numbers."""
    def __init__(self,path,rows=None):
        import pyarrow.parquet as pq
        self.path=str(path);self.rows=rows
        self.pf=pq.ParquetFile(path)
        self.columns=self.pf.schema_arrow.names
    def __len__(self):return self.pf.metadata.num_rows if self.rows is None else len(self.rows)
    @property
    def iloc(self):
        owner=self
        class Indexer:
            def __getitem__(self,key):
                base=owner.rows
                if base is None:
                    if isinstance(key,slice): base=np.arange(*key.indices(len(owner)),dtype=np.int64)
                    else: base=np.asarray(key)
                else: base=base[key]
                return ParquetFrame(owner.path,base)
        return Indexer()
    def read(self,columns=None):
        import pandas as pd
        import pyarrow as pa
        chunks=[];offset=0
        for batch in self.pf.iter_batches(batch_size=STEP,columns=columns):
            if self.rows is None: chunks.append(batch)
            else:
                a,b=np.searchsorted(self.rows,[offset,offset+len(batch)])
                if b>a:chunks.append(batch.take(pa.array(self.rows[a:b]-offset)))
            offset+=len(batch)
        schema=self.pf.schema_arrow
        if columns is not None:schema=pa.schema([schema.field(c) for c in columns])
        table=pa.Table.from_batches(chunks,schema=schema)
        return table.to_pandas(types_mapper=pd.ArrowDtype)
    def __getitem__(self,columns):
        if isinstance(columns,str):return self.read([columns])[columns]
        return self.read(columns)
    def reset_index(self,drop=True):return self.read().reset_index(drop=drop)


class EmbeddingTexts:
    def __init__(self,frame):self.frame=frame
    def __len__(self):return len(self.frame)
    def __getitem__(self,key):
        from dense import _texts
        return _texts(self.frame.iloc[key][['business_name','business_address']])


def unique_disk(texts):
    """Same first-occurrence dedup as _unique_order, with text lookup on disk."""
    import sqlite3
    from array import array
    temp=tempfile.TemporaryDirectory(prefix='ber-text-')
    db=sqlite3.connect(str(Path(temp.name)/'texts.sqlite'))
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-32768;'
                    'CREATE TABLE texts(id INTEGER PRIMARY KEY,text TEXT UNIQUE);')
    inv=np.lib.format.open_memmap(Path(temp.name)/'inverse.npy',mode='w+',dtype=np.int32,shape=(len(texts),))
    n=0
    for start in range(0,len(texts),STEP):
        for offset,text in enumerate(texts[start:start+STEP]):
            row=db.execute('SELECT id FROM texts WHERE text=?',(text,)).fetchone()
            if row is None:
                j=n;n+=1;db.execute('INSERT INTO texts VALUES(?,?)',(j,text))
            else:j=row[0]
            inv[start+offset]=j
        db.commit();release(inv)
    class UniqueTexts:
        def __init__(self):self.temp=temp;self.db=db
        def __len__(self):return n
        def __getitem__(self,key):
            if isinstance(key,slice):
                a,b,step=key.indices(n)
                if step!=1:raise ValueError('only contiguous text slices supported')
                return [r[0] for r in db.execute('SELECT text FROM texts WHERE id>=? AND id<? ORDER BY id',(a,b))]
            return db.execute('SELECT text FROM texts WHERE id=?',(int(key),)).fetchone()[0]
        def __del__(self):
            self.db.close()
            self.temp.cleanup()
    return UniqueTexts(),inv


def write_context(mm,q,i,nt,at,bscore):
    from features import BASE_FEATURES, CONTEXT_FEATURES
    from utils import group_max,group_rank_desc
    def put(name,v):
        mm[:,len(BASE_FEATURES)+CONTEXT_FEATURES.index(name)]=v
        release(mm)
    combo=.5*np.nan_to_num(nt)+.5*np.nan_to_num(at)
    for name,v in [('n_tset',nt),('a_tset',at),('combo',combo)]:
        mx=group_max(q,v);put('g_max_'+name,mx)
        put('g_diff_'+name,(np.nan_to_num(v,nan=0)-np.nan_to_num(mx,nan=0)).astype(np.float32))
        del mx
        put('g_rank_'+name,group_rank_desc(q,v))
    put('g_rank_bscore',group_rank_desc(q,bscore))
    put('q_ncand',np.bincount(q)[q].astype(np.float32))
    put('i_ncand_s1',np.bincount(i)[i].astype(np.float32))
    put('i_rank_combo',group_rank_desc(i,combo))
    put('i_diff_combo',(combo-group_max(i,combo)).astype(np.float32))
    put('i_rank_bscore',group_rank_desc(i,bscore))


def stage2_stream(q,i,p1,mm,i_ncore,i_afull,out):
    from stage2 import S2_FEATURES,S2_CORE,_cp_masked
    from features import FIDX
    from utils import group_top2,rank_in_sorted_groups
    n=len(q);col={name:j for j,name in enumerate(S2_FEATURES)}
    def put(name,v):out[:,col[name]]=v;release(out)
    for s in range(0,n,STEP):
        pc=np.clip(p1[s:s+STEP].astype(np.float64),1e-6,1-1e-6)
        out[s:s+STEP,col['p1_logit']]=np.log(pc/(1-pc))
    rk,t1,t2=group_top2(q,p1)
    put('g_rank_p1',rk+1);put('g_max_p1',t1);put('g_second_p1',t2);put('g_diff_p1',p1-t1)
    del rk,t1,t2
    put('g_sum_p1',np.bincount(q,weights=p1)[q])
    put('g_cnt50_p1',np.bincount(q,weights=(p1>.5).astype(np.float64))[q])
    put('q_ncand',np.bincount(q)[q]);put('i_ncand_s1',np.bincount(i)[i])
    rk,t1,t2=group_top2(i,p1);other=np.where(rk==0,t2,t1)
    put('i_rank_p1',rk+1);put('i_max_other_p1',other);put('i_diff_other_p1',p1-np.nan_to_num(other,nan=0))
    del rk,t1,t2,other
    nq=int(q.max())+1 if n else 0
    order=np.lexsort((-p1,q));qs=q[order];rk=rank_in_sorted_groups(qs)
    T=np.full((nq,3),-1,np.int64);mask=rk<3;T[qs[mask],rk[mask]]=order[mask]
    del order,qs,rk,mask
    for start in range(0,n,STEP):
        sl=slice(start,min(n,start+STEP));rows=np.arange(start,sl.stop);qc=q[sl];ic=i[sl]
        T0,T1,T2=T[qc,0],T[qc,1],T[qc,2]
        ref1=np.where(T0!=rows,T0,T1);ref2=np.where((T0!=rows)&(T1!=rows),T1,T2)
        v1,v2=ref1>=0,ref2>=0;r1,r2=np.where(v1,ref1,0),np.where(v2,ref2,0)
        c_n1=_cp_masked(ic,i[r1],i_ncore,v1)
        c_a1=_cp_masked(ic,i[r1],i_afull,v1&(i_afull[ic]!='')&(i_afull[i[r1]]!=''))
        c_n2=_cp_masked(ic,i[r2],i_ncore,v2)
        c_a2=_cp_masked(ic,i[r2],i_afull,v2&(i_afull[ic]!='')&(i_afull[i[r2]]!=''))
        for name,v in [('c2t_n1',c_n1),('c2t_a1',c_a1),('c2t_n2',c_n2),('c2t_a2',c_a2)]:out[sl,col[name]]=v
        w1=np.where(v1,p1[r1],0);w2=np.where(v2,p1[r2],0);den=w1+w2+1e-6
        out[sl,col['support_n']]=(w1*np.nan_to_num(c_n1)+w2*np.nan_to_num(c_n2))/den
        out[sl,col['support_a']]=(w1*np.nan_to_num(c_a1)+w2*np.nan_to_num(c_a2))/den
        for name in S2_CORE:out[sl,col[name]]=mm[sl,FIDX[name]].astype(np.float32)
        release(mm,out)
    return out

