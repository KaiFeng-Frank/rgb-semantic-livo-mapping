"""Owned remote supervisor: one extraction followed by CPU-only GLFS fitting."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/home/wuyou/glfs_reference')
project = root/'project'
python = str(root/'venv/bin/python')
state_path = root/'execution-state.json'


def save(state):
    tmp = state_path.with_suffix('.partial')
    tmp.write_text(json.dumps(state, indent=2)+'\n')
    tmp.replace(state_path)


state = dict(supervisor_pid=os.getpid(), started=time.time(), source_commit='a226472',
             protocol_sha256=hashlib.file_digest((root/'protocol.json').open('rb'), 'sha256').hexdigest(),
             status='starting', stages=[])
save(state)
env = os.environ.copy()
env.update(PTV3_POINTCEPT=str(root/'Pointcept_v151'), PTV3_CONFIG=str(root/'inputs/ptv3-config.py'),
           PTV3_CHECKPOINT=str(root/'backbone.pth'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
try:
    free_mib = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free_mib < 6500:
        raise RuntimeError('Insufficient GPU headroom for the capped encoder extraction')
    versions = {}
    for name in ('torch', 'torchvision', 'spconv-cu126', 'flash-attn', 'torch-scatter', 'numpy', 'scipy', 'timm'):
        versions[name] = importlib.metadata.version(name)
    (root/'runtime-versions.json').write_text(json.dumps(dict(python=sys.version, packages=versions,
        initial_free_gpu_mib=free_mib, extractor_gpu_memory_fraction=.18), indent=2))
    commands = [
        ('extraction', [python, '-u', '-m', 'tools.semantic_memory.glfs_extract', '--root', str(root),
                        '--data', '/home/wuyou/taseg_reference/data/nuscenes'], '0'),
        ('fitting', [python, '-u', '-m', 'tools.semantic_memory.glfs_experiment', '--root', str(root)], ''),
    ]
    for name, command, visible in commands:
        stage = dict(name=name, started=time.time(), command=command)
        state.update(status='running', stage=name)
        state['stages'].append(stage)
        with (root/f'{name}.log').open('w') as log:
            child = subprocess.Popen(command, cwd=project, env={**env, 'CUDA_VISIBLE_DEVICES':visible},
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            stage['pid'] = child.pid
            stage['proc_start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().split(') ')[1].split()[19]
            save(state)
            stage['returncode'] = child.wait()
        stage['finished'] = time.time()
        save(state)
        if stage['returncode']:
            raise RuntimeError(f'{name} exited with {stage["returncode"]}; original logs retained')
    state.update(status='complete', finished=time.time())
    save(state)
except Exception as error:
    state.update(status='failed', error=str(error), finished=time.time())
    save(state)
    raise
