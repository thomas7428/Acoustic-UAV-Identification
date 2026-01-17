#!/usr/bin/env python3
"""Produce a small dataset report for v4 datasets.

Usage: tools/dataset_report_v4.py --dataset datasets/<id>
"""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize_meta_list(metas):
    vals = {}
    # peak/rms/clip
    peak = [m.get('train_meta',{}).get('peak_dbfs') for m in metas if m.get('train_meta',{}).get('peak_dbfs') is not None]
    rms = [m.get('train_meta',{}).get('rms_dbfs') for m in metas if m.get('train_meta',{}).get('rms_dbfs') is not None]
    clip = [m.get('train_meta',{}).get('clip_count',0) for m in metas]
    vals['count'] = len(metas)
    vals['peak_mean'] = float(np.mean(peak)) if peak else None
    vals['rms_mean'] = float(np.mean(rms)) if rms else None
    vals['clip_total'] = int(np.sum(clip)) if clip else 0
    return vals


def load_jsonl(p:Path):
    out=[]
    if not p.exists():
        return out
    with p.open('r',encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--save_csv', default=None)
    args = p.parse_args()

    ds = Path(args.dataset)
    if not ds.exists():
        print('Dataset not found:', ds)
        return 2

    # detect splits by listing dirs
    splits = [d for d in ds.iterdir() if d.is_dir()]
    report = {}
    for s in sorted(splits):
        meta_path = s / 'augmentation_samples.jsonl'
        metas = load_jsonl(meta_path)
        summary = summarize_meta_list(metas)
        # additional debug stats
        debug_rirs = [m.get('debug_meta',{}).get('rir_id') for m in metas if m.get('debug_meta')]
        summary['unique_rir'] = len(set([r for r in debug_rirs if r is not None]))
        # distance stats
        dists = [m.get('debug_meta',{}).get('distance_m') for m in metas if m.get('debug_meta',{}).get('distance_m') is not None]
        summary['distance_count'] = len(dists)
        summary['distance_mean'] = float(np.mean(dists)) if dists else None
        report[s.name] = summary

    print('\nDataset report for', ds)
    for k,v in report.items():
        print(f"\nSplit: {k}")
        for kk,vv in v.items():
            print(f"  {kk}: {vv}")

    if args.save_csv:
        import csv
        with open(args.save_csv,'w',newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['split','count','peak_mean','rms_mean','clip_total','unique_rir','distance_count','distance_mean'])
            for k,v in report.items():
                w.writerow([k,v['count'],v['peak_mean'],v['rms_mean'],v['clip_total'],v['unique_rir'],v['distance_count'],v['distance_mean']])
        print('Saved CSV to', args.save_csv)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
