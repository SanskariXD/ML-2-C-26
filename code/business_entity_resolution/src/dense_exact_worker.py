"""Isolated exact FAISS search; same index, query batches, and tie handling as baseline.

Preallocating the flat vector once avoids the baseline's complete float32 input
copy and FAISS vector growth. All reference rows remain in the index.
"""
import argparse
from pathlib import Path
import numpy as np
from low_memory import release


def exact_search(emb_q,emb_i,q_rows,i_rows,k,output):
    import faiss
    nq,ni=len(q_rows),len(i_rows);k=min(int(k),ni);dim=emb_i.shape[1]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not nq or not k:
        for name,dtype in [('q',np.int32),('i',np.int32),('s',np.float32)]:np.save(output/f'{name}.npy',np.empty(0,dtype))
        return
    index=faiss.IndexFlatIP(dim)
    # FlatCodes stores bytes. One allocation avoids add() repeatedly reallocating
    # a growing std::vector (which temporarily holds old + new index buffers).
    index.codes.resize(ni*dim*4)
    index.ntotal=ni
    target=faiss.rev_swig_ptr(index.get_xb(),ni*dim).reshape(ni,dim)
    for start in range(0,ni,25_000):
        block=np.ascontiguousarray(emb_i[i_rows[start:start+25_000]],dtype=np.float32)
        faiss.normalize_L2(block);target[start:start+len(block)]=block
        release(emb_i)
    del block,target
    result={name:np.lib.format.open_memmap(output/f'{name}.npy',mode='w+',dtype=dtype,shape=(nq*k,))
            for name,dtype in [('q',np.int32),('i',np.int32),('s',np.float32)]}
    batch=min(nq,262_144) # same CPU IndexFlatIP query batch as the baseline
    for start in range(0,nq,batch):
        stop=min(nq,start+batch)
        Q=np.ascontiguousarray(emb_q[q_rows[start:stop]],dtype=np.float32)
        faiss.normalize_L2(Q);D,I=index.search(Q,k)
        sl=slice(start*k,stop*k)
        result['q'][sl]=np.repeat(np.arange(start,stop,dtype=np.int32),k)
        result['i'][sl]=I.reshape(-1).astype(np.int32)
        result['s'][sl]=D.reshape(-1)
        release(emb_q,*result.values())
    del index


def isolated_knn(cfg,eq,ei,q_rows,i_rows,k,directory):
    import os,sys,subprocess,gc
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    np.save(path/'query_rows.npy',q_rows);np.save(path/'index_rows.npy',i_rows)
    index_gib=len(i_rows)*ei.shape[1]*4/2**30
    # This is a measured allocation calculation, not a score-changing fallback.
    if index_gib>9:
        raise RuntimeError(f'Exact index alone needs {index_gib:.2f} GiB. This partition exceeds the 12.7 GiB profile; candidate budgets were NOT reduced.')
    release(eq,ei);gc.collect()
    env=dict(os.environ,OMP_NUM_THREADS=str(max(1,cfg.workers)),OPENBLAS_NUM_THREADS=str(max(1,cfg.workers)))
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--query',str(eq.filename),
                    '--index',str(ei.filename),'--rows',str(path),'--k',str(k)],check=True,env=env)
    return tuple(np.load(path/f'{name}.npy',mmap_mode='r') for name in ('q','i','s'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--query',required=True);ap.add_argument('--index',required=True)
    ap.add_argument('--rows',required=True);ap.add_argument('--k',type=int,required=True);a=ap.parse_args()
    p=Path(a.rows)
    exact_search(np.load(a.query,mmap_mode='r'),np.load(a.index,mmap_mode='r'),
                 np.load(p/'query_rows.npy'),np.load(p/'index_rows.npy'),a.k,p)
