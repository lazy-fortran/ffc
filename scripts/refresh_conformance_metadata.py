#!/usr/bin/env python3
"""Refresh xfail/skip/fail_owners manifests from a locked epoch ( regeneration,
never hand-editing ).  Deterministic: PASS rows drop from all manifests,
XPASS rows drop from xfail (promotion), XFAIL rows keep-or-default their
owner, SKIP rows get scope=harness skip entries carrying the walker note,
FAIL rows keep-or-default owner=#540 pending triage.
Usage: refresh_conformance_metadata.py --epoch-dir DIR --manifest-dir DIR
"""
import argparse, json, os, re

SUITES = ['fortfront-f90','fortfront-lf','lfortran','gfortran-dg']

def safe(s): return s.replace('-','_')

def parse_manifest(path):
    entries={}   # file -> full suffix after ' # '
    order=[]
    if not os.path.exists(path): return entries, order
    for line in open(path).read().split('\n'):
        if not line.strip(): continue
        if ' # ' in line:
            f,meta=line.split(' # ',1)
            entries[f.strip()]=meta
        order.append(line)
    return entries, order

def rebuild(path, entries, comments):
    lines=list(comments)
    for f in sorted(entries):
        lines.append(f"{f} # {entries[f]}")
    open(path,'w').write('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epoch-dir',required=True)
    ap.add_argument('--manifest-dir',required=True)
    a=ap.parse_args()
    for suite in SUITES:
        rep=os.path.join(a.epoch_dir,f"{suite}.jsonl")
        if not os.path.exists(rep): continue
        md=os.path.join(a.manifest_dir,f"fail_owners_{safe(suite)}.txt")
        xf=os.path.join(a.manifest_dir,f"xfail_{safe(suite)}.txt")
        sk=os.path.join(a.manifest_dir,f"skip_{safe(suite)}.txt")
        fo,_=parse_manifest(md); xfe,_=parse_manifest(xf); ske,_=parse_manifest(sk)
        def comments(p):
            if not os.path.exists(p): return []
            return [l for l in open(p).read().split('\n') if l.strip() and ' # ' not in l]
        cf,cx,cs=comments(md),comments(xf),comments(sk)
        n={'drop_pass':0,'promote':0,'xfail':0,'skip':0,'fail':0}
        for line in open(rep):
            try: j=json.loads(line)
            except Exception: continue
            f,s=j.get('file'),j.get('status')
            if not f or not s or s=='SUMMARY': continue
            if s=='PASS':
                for d in (fo,xfe,ske):
                    if f in d: del d[f]
                n['drop_pass']+=1
            elif s=='XPASS':
                if f in xfe: del xfe[f]
                n['promote']+=1
            elif s=='XFAIL':
                xfe.setdefault(f, 'owner=lazy-fortran/ffc#540; reason=pending triage')
                n['xfail']+=1
            elif s=='SKIP':
                note=j.get('note') or 'skipped by walker'
                ske.setdefault(f, f"scope=harness; reason={note}")
                n['skip']+=1
            elif s=='FAIL':
                fo.setdefault(f, 'owner=lazy-fortran/ffc#540; reason=fresh full-epoch failure pending triage')
                n['fail']+=1
        rebuild(md,fo,cf); rebuild(xf,xfe,cx); rebuild(sk,ske,cs)
        print(suite,n)

main()
