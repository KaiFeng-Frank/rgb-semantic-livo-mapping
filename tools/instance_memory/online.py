"""Opt-in CPU memory worker adapter; the ROS Python never imports torch.

The current map keeps raw independent semantic evidence. Only the current scan
and instance output may use this experimental head. No silent worker restarts.
"""
import argparse
from multiprocessing.connection import Connection
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def observations(xyz,probabilities,association,features,stamp_ns,pose):
    from tools.visual_instances.semantics import Config,component
    cfg=Config();result=[]
    ids=association['ids'];camera_stamp=association['info'].get('camera_stamp_ns',stamp_ns)
    for tid in np.unique(ids[ids>0]):
        tid=int(tid)
        visual=association.get('visual_features',{}).get(tid)
        if visual is None or tid not in features:continue
        at=np.flatnonzero(ids==tid);chosen=component(xyz[at],cfg)
        if chosen is None:continue
        at=at[chosen];points=xyz[at];center=np.median(points,axis=0)
        extent=np.quantile(points,.9,axis=0)-np.quantile(points,.1,axis=0)
        p=probabilities[at];mean=p.mean(0)
        value,available=visual
        quality=np.array([np.log1p(len(at))/8.,len(at)/max(1,np.count_nonzero(ids==tid)),
            np.linalg.norm(center)/80.,np.linalg.norm(extent)/10.,
            -(p*np.log(np.maximum(p,1e-8))).sum(1).mean()/np.log(16),p.std(0).mean(),
            (stamp_ns-camera_stamp)*1e-9,float(available)],np.float32)
        x=np.concatenate([features[tid],value,mean,association['class_scores'][tid],association['appearance'][tid],quality])
        if x.shape!=(320,) or not np.isfinite(x).all():raise ValueError('Invalid online memory feature')
        result.append(dict(x=x,raw_mean=mean,raw_probabilities=p,indices=at,
            visual_track=tid,visual_session=association['info']['session'],
            center=pose[:3,:3]@center+pose[:3,3],extent=extent,sensor_center=center,
            stamp=stamp_ns,camera_stamp=camera_stamp,visual_label=int(association['labels'][at[0]]),
            visual_confidence=float(association['scores'][at[0]])))
    return result


class OnlineMemory:
    def __init__(self,checkpoint,python,log,*,mode='feature_memory',cancel_event=None):
        from multiprocessing import Pipe
        self.conn,child=Pipe();self.cancel=cancel_event;self.log=Path(log).open('w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        # Avoid loading ROS Python 3.12 extension modules in the Torch 3.11 worker.
        env['PYTHONPATH']=''
        self.process=subprocess.Popen([str(python),'-u','-m','tools.instance_memory.online','--fd',str(child.fileno()),
            '--checkpoint',str(checkpoint),'--mode',mode],pass_fds=[child.fileno()],
            cwd=Path(__file__).resolve().parents[2],env=env,stdout=self.log,stderr=subprocess.STDOUT)
        child.close()
        try:self.meta=self.receive(90)
        except BaseException:self.close();raise

    def receive(self,seconds):
        deadline=time.monotonic()+seconds
        while not self.conn.poll(.01):
            if self.cancel is not None and self.cancel.is_set():raise InterruptedError('Memory cancelled')
            if self.process.poll() is not None:raise RuntimeError('Memory worker exited')
            if time.monotonic()>deadline:raise TimeoutError('Memory response deadline')
        result=self.conn.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result

    def configuration(self):return self.meta

    def revise_poses(self,times,poses):
        self.conn.send(dict(op='revise',poses={int(t):p for t,p in zip(times,poses)}))
        self.receive(1.)

    def step(self,xyz,probabilities,association,*,features,stamp_ns,pose):
        started=time.perf_counter()
        obs=observations(xyz,probabilities,association,features,stamp_ns,pose)
        self.conn.send(dict(op='step',frame=dict(stamp=stamp_ns,observations=obs)))
        data=self.receive(1.)
        raw=probabilities.argmax(1).astype(np.uint16);labels=raw.copy();conf=probabilities.max(1).copy()
        n=len(raw)
        result=dict(labels=labels,confidence=conf,raw_labels=raw,raw_confidence=conf.copy(),
            valid=np.zeros(n,np.uint8),epochs=np.zeros(n,np.uint32),ids=np.zeros(n,np.uint32),
            instance_labels=np.full(n,65535,np.uint16),instance_confidence=np.zeros(n,np.float32))
        for o,p,tid in zip(obs,data['probabilities'],data['ids']):
            at=o['indices'];p=np.asarray(p);result['labels'][at]=p.argmax(1);result['confidence'][at]=p.max(1)
            result['epochs'][at]=tid;result['ids'][at]=tid
            # Shared object class is a summary; the point output may retain
            # within-mask variation instead of forcing all mixed points equal.
            mean=p.mean(0);result['instance_labels'][at]=mean.argmax();result['instance_confidence'][at]=mean.max()
            # Existing wire contract: instance_valid means all selected points
            # actually share that class. Mixed point predictions keep valid=0.
            result['valid'][at]=int(np.all(p.argmax(1)==mean.argmax()))
        result['info']=dict(data['info'],changed_points=int((labels!=raw).sum()),
            elapsed_with_ipc_ms=(time.perf_counter()-started)*1000,mode=self.meta['mode'],
            missing_feature_observations=int(len(np.unique(association['ids'][association['ids']>0]))-len(obs)))
        return result

    def close(self):
        try:self.conn.close()
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
            self.log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fd',type=int,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--mode',choices=['feature_memory','matched_current','same_memory_head_without_history','uniform_history'],default='feature_memory')
    a=p.parse_args();conn=Connection(a.fd)
    try:
        import torch
        from .evaluate import checkpoint
        from .runtime import Runtime
        torch.set_num_threads(1)
        model,identity,meta=checkpoint(a.checkpoint)
        rt=Runtime(model,identity,a.mode)
        conn.send(dict(mode=a.mode,seed=meta['seed'],epoch=meta['epoch'],
            checkpoint=str(a.checkpoint),cache_protocol_sha256=meta['cache_protocol_sha256'],
            evidence='raw-only causal instance feature memory; experimental supervised native16'))
        while True:
            message=conn.recv()
            if message['op']=='revise':
                rt.memory.revise(message['poses']);conn.send(dict(revised=True))
            elif message['op']=='step':
                with torch.inference_mode():
                    outputs,ids,info=rt.step(message['frame'])
                    probabilities=[o.softmax(-1).numpy() for o in outputs]
                conn.send(dict(probabilities=probabilities,ids=ids,info=info))
            else:raise ValueError('Unknown memory operation')
    except (EOFError,BrokenPipeError):pass
    except Exception as e:
        try:conn.send(dict(error=f'{type(e).__name__}: {e}'))
        except (EOFError,BrokenPipeError):pass
        raise
    finally:conn.close()


if __name__=='__main__':main()
