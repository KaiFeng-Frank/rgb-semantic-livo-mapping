"""Real subprocess/SHM round trips, including slot reuse and startup failure."""
from multiprocessing import shared_memory
import os
import sys

import numpy as np
import pytest

from src import ptv3_client as module


WORKER = '''
import os, struct, sys
import numpy as np
from multiprocessing import shared_memory, resource_tracker
shm = shared_memory.SharedMemory(name=os.environ['PTV3_SHM_NAME'])
resource_tracker.unregister(shm._name, 'shared_memory')
s, n, c = [int(os.environ[x]) for x in ('PTV3_SHM_NSLOT','PTV3_SHM_MAXPTS','PTV3_PROBABILITY_CLASSES')]
lab = np.ndarray((s,n),dtype=np.int16,buffer=shm.buf,offset=s*n*16)
conf = np.ndarray((s,n),dtype=np.float32,buffer=shm.buf,offset=s*n*18)
prob = np.ndarray((s,n,c),dtype=np.float32,buffer=shm.buf,offset=s*n*22)
os.write(1,b'RDY0'+struct.pack('<I',c))
while True:
    tag = sys.stdin.buffer.read(1)
    if tag in (b'',b'Q'): break
    assert tag == b'H'
    seq,slot,count = struct.unpack('<IBI',sys.stdin.buffer.read(9))
    lab[slot,:count] = seq%c
    conf[slot,:count] = .9
    prob[slot,:count] = .1/(c-1)
    prob[slot,:count,seq%c] = .9
    os.write(1,b'r'+struct.pack('<IBIdI',seq,slot,count,1.,count))
lab=conf=prob=None
shm.close()
'''


def test_probability_slot_ownership_wraparound_and_legacy_tuple(tmp_path,monkeypatch):
    worker = tmp_path/'worker.py'
    worker.write_text(WORKER)
    monkeypatch.setattr(module,'PTV3_PY',sys.executable)
    monkeypatch.setattr(module,'WORKER',str(worker))
    # Preserve all positional arguments of the existing constructor.
    client = module.PTv3Client(None,.2,None,True,4,8,None,None,None,None,None,
                              None,False,120.,10.,None,probability_classes=16)
    name=client.shm.name
    copies=[]
    try:
        for i in range(10):
            lab,conf,p=client.segment_probabilities(np.zeros((i%7+1,4),np.float32))
            assert np.all(lab==i%16)
            assert np.array_equal(p.argmax(1),lab)
            assert np.allclose(p.max(1),conf)
            assert np.allclose(p.sum(1),1)
            copies.append(p)
        assert all(np.all(p.argmax(1)==i%16) for i,p in enumerate(copies))
        assert len(client.segment(np.zeros((1,4),np.float32)))==2
        client.submit(np.zeros((1,4),np.float32))
        with pytest.raises(RuntimeError):
            client.segment_probabilities(np.zeros((1,4),np.float32))
        client.collect()
    finally:
        client.close()
    assert client.p.returncode==0
    with pytest.raises(FileNotFoundError): shared_memory.SharedMemory(name=name)


def test_vocabulary_mismatch_reaps_and_unlinks(tmp_path,monkeypatch):
    worker=tmp_path/'worker.py'
    worker.write_text("import os,struct,time\nos.write(1,b'RDY0'+struct.pack('<I',19))\ntime.sleep(60)\n")
    monkeypatch.setattr(module,'PTV3_PY',sys.executable)
    monkeypatch.setattr(module,'WORKER',str(worker))
    processes=[]
    original=module.subprocess.Popen
    def launch(*args,**kwargs):
        p=original(*args,**kwargs)
        processes.append((p,kwargs['env']['PTV3_SHM_NAME']))
        return p
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    with pytest.raises(ValueError,match='vocabulary'):
        module.PTv3Client(probability_classes=16,max_pts=8)
    p,name=processes[0]
    assert p.poll() is not None
    assert p.stdin.closed and p.stdout.closed
    with pytest.raises(FileNotFoundError): shared_memory.SharedMemory(name=name)


@pytest.mark.parametrize('kwargs',[dict(probability_classes=16,use_shm=False),
    dict(probability_classes=16,memory_shadow=True),dict(probability_classes=-1)])
def test_invalid_export_contract_rejected_before_spawn(kwargs,monkeypatch):
    def forbidden(*a,**kw): raise AssertionError('must not spawn')
    monkeypatch.setattr(module.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError): module.PTv3Client(**kwargs)


def test_default_does_not_inherit_probability_export(tmp_path,monkeypatch):
    worker=tmp_path/'worker.py'
    worker.write_text("import os,struct\nassert 'PTV3_PROBABILITY_CLASSES' not in os.environ\nos.write(1,b'RDY0'+struct.pack('<I',16))\nassert os.read(0,1)==b'Q'\n")
    monkeypatch.setattr(module,'PTV3_PY',sys.executable)
    monkeypatch.setattr(module,'WORKER',str(worker))
    monkeypatch.setenv('PTV3_PROBABILITY_CLASSES','19')
    client=module.PTv3Client(max_pts=8)
    try:
        with pytest.raises(RuntimeError,match='Enable'):
            client.segment_probabilities(np.zeros((1,4),np.float32))
    finally: client.close()
    assert client.p.returncode==0
