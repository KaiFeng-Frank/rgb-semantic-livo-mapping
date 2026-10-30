"""Export small reproducibility artifacts, never sensor data or model tensors."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path('/home/wuyou/stocc_reference/history-diagnostic')


def main():
    paths={name:name for name in [
        'protocol.json','report.json','report-verification.json','analysis-correction.json',
        'frame-point-effects.json','pedestrian-first-observation-harms.json',
        'target-audit.json','point-target-manifest.json','panoptic-acquisition.json',
        'panoptic-files.json','paired-launch.json','checkpoint-audit.json','object-frames.json.gz']}
    for name in ['events.json','report-initial-present-class.json','report-verification-initial.json']:
        paths[name]=name+'.gz'
    for name in ['native_history.log','cold_history.log','smoke-cold.log',
                 'target-preparation.log','analysis-final.log','analysis-tests-final.log']:
        paths[name]='logs/'+name+'.gz'
    for arm in ['native_history','cold_history','smoke-cold']:
        for name in ['summary.json','frames.jsonl','effective-config.py','evaluate-as-run.py',
                     'history-control-as-run.py','official-metrics/results.csv']:
            source=ROOT/arm/name
            if source.is_file():paths[str(source.relative_to(ROOT))]=str(source.relative_to(ROOT))
    for source in sorted((ROOT/'sources-as-run').iterdir()):
        if source.suffix in ('.py','.sh'):paths[str(source.relative_to(ROOT))]=str(source.relative_to(ROOT))
    payloads={}
    manifest={}
    for src,dest in sorted(paths.items()):
        data=(ROOT/src).read_bytes()
        if dest.endswith('.gz') and not src.endswith('.gz'):data=gzip.compress(data,mtime=0)
        assert len(data)<20*1024**2,src
        payloads[dest]=data
        manifest[dest]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    payloads['artifact-manifest.json']=json.dumps(manifest,indent=2).encode()
    output=ROOT.parent/'history-diagnostic-evidence.tar.gz'
    with tarfile.open(output,'w:gz') as archive:
        for name,data in payloads.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o644
            archive.addfile(entry,io.BytesIO(data))
    print(json.dumps({'path':str(output),'files':len(payloads),'bytes':output.stat().st_size,
                      'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
