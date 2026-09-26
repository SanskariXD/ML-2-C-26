"""Streaming equivalent of prepare, including source dedupe and dev ownership rules."""
import os
import pickle
import sqlite3
import zlib
from array import array
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io_utils import COLS
from normalize import normalize_frame
from utils import LOG


def source_rows(path):
    with open(path,encoding='utf-8-sig',errors='replace',newline='') as f:
        header=[h.strip().lower() for h in f.readline().rstrip('\r\n').split('\t')]
        if 'entity_id' not in header: raise ValueError(f'{path}: no entity_id column')
        pos={c:header.index(c) if c in header else -1 for c in COLS}
        n=len(header); standard=[pos[c] for c in COLS]==[0,1,2,3] and n==4
        for line in f:
            line=line.rstrip('\r\n')
            if not line.strip(): continue
            p=line.split('\t')
            if len(p)>n and standard: p=[p[0],p[1],' '.join(p[2:-1]),p[-1]]
            elif len(p)<n: p += ['']*(n-len(p))
            row=tuple(p[pos[c]].strip() if pos[c]>=0 else '' for c in COLS)
            if row[0]: yield row


def prepare_stream(cfg,split):
    from prepare import split_dir
    d=Path(split_dir(cfg,split)); done=d/'partitions.pkl'
    if done.exists() and not cfg.force: return
    dbpath=d/'prepare.partial.sqlite'; dbpath.unlink(missing_ok=True)
    db=sqlite3.connect(dbpath)
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-32768;'
                    'CREATE TABLE ids(side TEXT,id TEXT,r INTEGER,PRIMARY KEY(side,id)) WITHOUT ROWID;'
                    'CREATE TABLE gt(s TEXT PRIMARY KEY,ms TEXT,seq INTEGER);'
                    'CREATE TABLE owners(id TEXT PRIMARY KEY,s TEXT) WITHOUT ROWID;')
    src=Path(cfg.data_dir)/split
    if split=='train':
        with open(src/'train_ground_truth.tsv',encoding='utf-8-sig',errors='replace') as f:
            next(f)
            for seq,line in enumerate(f):
                line=line.rstrip('\r\n')
                if not line.strip(): continue
                s,_,rest=line.partition('\t')
                db.execute('INSERT INTO gt VALUES(?,?,?) ON CONFLICT(s) DO UPDATE SET ms=excluded.ms',
                           (s.strip(),rest,seq))
        if cfg.dev_frac<1:
            for s,ms in db.execute('SELECT s,ms FROM gt ORDER BY seq'):
                db.executemany('INSERT OR REPLACE INTO owners VALUES(?,?)',
                               [(m.strip(),s) for m in ms.split(',') if m.strip()])
        db.commit()
    groups={'s1':{},'idx':{}}; counts={'s1':0,'idx':0}; writers={}
    def flush(rows,side,source):
        if not rows: return
        df=pd.DataFrame(rows,columns=COLS)
        norm=normalize_frame(df,1)
        for c in COLS: df[c]=df[c].astype('string[pyarrow]')
        if side=='idx': df['src']=np.int8(source)
        df=pd.concat([df,norm],axis=1)
        df['ckey']=df['country'].astype(str).str.strip().str.casefold().astype('string[pyarrow]')
        start=counts[side]-len(rows)
        for ck,ix in df.groupby('ckey',sort=True).indices.items():
            groups[side].setdefault(ck,array('i')).extend((ix+start).tolist())
        table=pa.Table.from_pandas(df,preserve_index=False)
        if side not in writers: writers[side]=pq.ParquetWriter(d/f'{side}.partial.parquet',table.schema)
        writers[side].write_table(table)
    try:
        for source in (1,2,3):
            side='s1' if source==1 else 'idx'; rows=[]
            for row in source_rows(src/f'{split}_source{source}.tsv'):
                eid=row[0]
                keep=True
                if split=='train' and cfg.dev_frac<1:
                    if source==1: keep=zlib.crc32(b'dev'+eid.encode())%1000<int(cfg.dev_frac*1000)
                    else:
                        owner=db.execute('SELECT s FROM owners WHERE id=?',(eid,)).fetchone()
                        if owner:
                            found=db.execute("SELECT r FROM ids WHERE side='s1' AND id=?",owner).fetchone()
                            keep=found is not None and found[0] is not None
                        else: keep=zlib.crc32(b'devx'+eid.encode())%1000<int(cfg.dev_frac*1000)
                cur=db.execute('INSERT OR IGNORE INTO ids VALUES(?,?,?)',(side,eid,counts[side] if keep else None))
                if not cur.rowcount or not keep: continue
                rows.append(row); counts[side]+=1
                if len(rows)>=25_000:
                    flush(rows,side,source); rows=[]; db.commit()
                    LOG.info('prepare %s %s: %d records',split,side,counts[side])
            flush(rows,side,source); db.commit()
        # Preserve valid empty-source schemas too.
        for side in ('s1','idx'):
            if side not in writers:
                df=pd.DataFrame({c:pd.Series([],dtype='string[pyarrow]') for c in COLS})
                if side=='idx': df['src']=pd.Series([],dtype='int8')
                df=pd.concat([df,normalize_frame(df,1)],axis=1)
                df['ckey']=pd.Series([],dtype='string[pyarrow]')
                pq.write_table(pa.Table.from_pandas(df,preserve_index=False),d/f'{side}.partial.parquet')
        if split=='train':
            aa,bb=array('i'),array('i')
            for s,ms in db.execute('SELECT s,ms FROM gt ORDER BY seq'):
                found=db.execute("SELECT r FROM ids WHERE side='s1' AND id=?",(s,)).fetchone()
                if not found or found[0] is None: continue
                for m in ms.split(','):
                    if not m.strip(): continue
                    dest=db.execute("SELECT r FROM ids WHERE side='idx' AND id=?",(m.strip(),)).fetchone()
                    if dest and dest[0] is not None: aa.append(found[0]);bb.append(dest[0])
            np.savez(d/'gt.npz',s1=np.asarray(aa,np.int32),idx=np.asarray(bb,np.int32))
    finally:
        for w in writers.values(): w.close()
        db.close()
    parts={}; blank=np.asarray(groups['idx'].get('',array('i')),np.int32)
    for ck in sorted(groups['s1']):
        ii=np.asarray(groups['idx'].get(ck,array('i')),np.int32)
        if ck and len(blank): ii=np.union1d(ii,blank)
        parts[ck]=(np.asarray(groups['s1'][ck],np.int32),ii)
    for side in ('s1','idx'): os.replace(d/f'{side}.partial.parquet',d/f'{side}.parquet')
    with open(d/'partitions.partial.pkl','wb') as f: pickle.dump(parts,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(d/'partitions.partial.pkl',done)
    dbpath.unlink(missing_ok=True)
