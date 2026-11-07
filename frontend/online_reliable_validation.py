"""Repeat actual-scan 10 Hz transport/deadline checks with reliable input."""
import argparse
import hashlib
import json
from pathlib import Path
from frontend.online_admission import REPO, WARM, MEASURE, recover, direct, bridge


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    files=['frontend/input_bridge.py','frontend/online_admission.py','frontend/online_reliable_validation.py',
           'src/ptv3_client.py','src/ptv3_worker.py','src/ptv3_loader_verified.py','src/ptv3_fast.py']
    protocol=dict(warm_frames=WARM,measure_frames=MEASURE,repetitions=3,deadline_ms=80,
                  input_reliability='reliable',middleware='rmw_cyclonedds_cpp',required_semantic_fraction=.95,
                  required_coverage='all warm and measured scans, exact timestamps/order/bytes, every run',
                  raw_data_persisted=False,full_lio=False,
                  source_sha256={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in files})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    try:
        scans,meta=recover()
        (out/'scan-hashes.json').write_text(json.dumps({f:dict(points=len(v[0]),sha256=v[1]) for f,v in scans.items()},indent=2)+'\n')
        inference=direct(scans,out)
        records=[]
        for rep in range(1,4):
            target=out/f'r{rep}';target.mkdir()
            modes=['geometry','online'] if rep%2 else ['online','geometry']
            for mode in modes:
                result=bridge(scans,meta,target,mode,'reliable');result['repetition']=rep
                records.append(result)
                (out/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
                print('BRIDGE_RUN',json.dumps({k:v for k,v in result.items() if not k.endswith('stamp_ns')}),flush=True)
        passed=all(r['complete_ordered_coverage'] and r['point_bytes_unchanged'] and
                   r['all_timely_semantics'] and (r['mode']=='geometry' or r['semantic_fraction']>=.95) for r in records)
        result=dict(admitted=passed,inference=inference,runs=records,full_lio_online_validated=False,
                    scope='80 ms deadline starts at bridge receipt, not at sensor publication; complete paired reliable transport required')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('BRIDGE_ADMISSION',passed,flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)[:300]),indent=2)+'\n')
        raise


if __name__=='__main__':main()
