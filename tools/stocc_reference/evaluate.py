"""Instrument the pinned official ST-Occ model, pipeline, sampler and metrics.

No training or prediction-layer changes. Batch size one preserves scene memory.
The CUDA allocator cap and loader count are explicit runtime settings. Timing
includes official metric bookkeeping and is not pure deployment latency.
"""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

ROOT = Path('/home/wuyou/stocc_reference')
CONFIG = ROOT/'upstream/occupancy_configs/st_occ/stocc-r50-256x704-36e_flicker_eval.py'
CHECKPOINT = ROOT/'downloads/stocc-r50-256x704.pth'


class OfficialInputParallel(MMDataParallel):
    """Preserve the CPU metadata container expected by upstream forward_test.

    MMCV 1.6 unwraps this DataContainer during scatter, whereas the pinned
    ST-Occ entry point explicitly reads img_metas[0].data[0]. Only that CPU
    metadata representation is preserved; all measurement tensors use the
    regular MMCV CUDA transfer. No feature, prediction or state is changed.
    """
    def scatter(self, inputs, kwargs, device_ids):
        assert len(device_ids)==1
        scattered_inputs, scattered_kwargs=super().scatter(inputs, kwargs, device_ids)
        for payload in scattered_kwargs:
            payload['img_metas']=kwargs['img_metas']
        return scattered_inputs, scattered_kwargs


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))
    temp.replace(path)


def json_safe(value):
    if isinstance(value, dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value, (np.ndarray,np.generic,Path)): return json_safe(plain(value))
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--scenes', nargs='*')
    parser.add_argument('--ann-file', help='Optional official-format metadata subset for an early smoke test.')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--memory-fraction', type=float, default=.40)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--history-mode', choices=['native', 'cold'], default='native')
    parser.add_argument('--label-control', action='store_true', help='Replace evaluation occupancy by a constant and GT speed by zero; tests prediction independence, not metric quality.')
    args = parser.parse_args()
    os.chdir(ROOT/'upstream')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'predictions').mkdir(exist_ok=True)
    if (args.out/'progress.json').exists():
        raise RuntimeError('Use a fresh output directory, preserving previous evidence.')
    source=Path(__file__).read_bytes()
    (args.out/'evaluate-as-run.py').write_bytes(source)
    state = dict(status='starting', started=time.time(), arguments=vars(args),
                 source_commit='1633f62e2e6677a5fa474905977acfeca4e7819e',
                 checkpoint_sha256='68491f55a2208fad884e3b664d0e34b8311688376380f7722e30cac6bde5ffa0',
                 timing_scope='CUDA-synchronized model call; includes official metric bookkeeping, excludes data load. Resource concurrency is recorded separately.',
                 label_control=args.label_control)
    state['evaluator_sha256']=hashlib.sha256(source).hexdigest()
    state['compatibility']=json.loads((ROOT/'compatibility.json').read_text())
    save(args.out/'progress.json',state)
    try:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
        set_random_seed(0, deterministic=False)
        logging.getLogger('mmcv').setLevel(logging.ERROR)
        cfg=Config.fromfile(str(CONFIG))
        cfg.model.pretrained=None
        cfg.model.train_cfg=None
        cfg.model.flickering_cnt=True
        cfg.data.test.test_mode=True
        if args.ann_file:
            cfg.data.test.ann_file=args.ann_file
        if args.scenes:
            with open(cfg.data.test.ann_file, 'rb') as stream:
                infos=pickle.load(stream)
            infos['infos']=[i for i in infos['infos'] if i['scene_name'] in set(args.scenes)]
            assert set(i['scene_name'] for i in infos['infos'])==set(args.scenes)
            subset=args.out/'subset-info.pkl'
            with subset.open('wb') as stream: pickle.dump(infos,stream)
            cfg.data.test.ann_file=str(subset)
        cfg.dump(str(args.out/'effective-config.py'))
        dataset=build_dataset(cfg.data.test)
        loader=build_dataloader(dataset, samples_per_gpu=1,workers_per_gpu=args.workers,
                                dist=False,shuffle=False,seed=0,
                                runner_type=cfg.data.test_dataloader.runner_type)
        model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
        if cfg.get('fp16') is not None: wrap_fp16_model(model)
        load_checkpoint(model,str(CHECKPOINT),map_location='cpu',strict=True,
                        revise_keys=[(r'^module\.', ''),(r'^teacher\.', '')])
        state.update(strict_load=True, dataset_frames=len(dataset),
                     dataset_scenes=len(set(i['scene_name'] for i in dataset.data_infos)),
                     parameters=sum(p.numel() for p in model.parameters()))
        history_control = None
        if args.history_mode == 'cold':
            from history_control import ColdHistory
            control_source = Path(__file__).with_name('history_control.py').read_bytes()
            (args.out/'history-control-as-run.py').write_bytes(control_source)
            state['history_control_sha256'] = hashlib.sha256(control_source).hexdigest()
            history_control = ColdHistory(model)
        model=OfficialInputParallel(model.cuda(),device_ids=[0]);model.eval()
        results=[];rows=[];seen=set();last={}
        torch.cuda.reset_peak_memory_stats()
        state['status']='running';save(args.out/'progress.json',state)
        iterator=iter(loader)
        run_start=time.perf_counter()
        while len(seen)<len(dataset) and (args.max_frames is None or len(seen)<args.max_frames):
            t0=time.perf_counter();data=next(iterator);load_ms=(time.perf_counter()-t0)*1000
            # The official flicker metric excludes moving voxels using GT speed;
            # predicted memory flow is separate and must never receive gt_flow.
            assert 'gt_flow' not in data
            if args.label_control:
                # Evaluation-only label branch; changing it must not change predictions.
                for tensor in data['gt_occupancy']:
                    tensor.fill_(1)
                for tensor in data.get('gt_speed_map',[]):
                    tensor.zero_()
            torch.cuda.synchronize();t0=time.perf_counter()
            with torch.no_grad(): output=model(return_loss=False,rescale=True,**data)
            torch.cuda.synchronize();model_ms=(time.perf_counter()-t0)*1000
            assert len(output)==1
            item=output[0];idx=int(item['index'])
            if idx in seen: raise AssertionError('Sampler repeated a frame before full coverage')
            info=dataset.data_infos[idx]
            scene=info['scene_name'];frame=int(info['global_idx'])
            assert frame==last.get(scene,-1)+1, (scene,frame,last.get(scene))
            last[scene]=frame;seen.add(idx)
            prediction=item['pred_occupancy']
            assert prediction.shape==(200,200,16)
            assert np.isfinite(prediction).all() and prediction.min()>=0 and prediction.max()<18
            prediction=prediction.astype(np.uint8)
            item['pred_occupancy']=prediction
            np.savez_compressed(args.out/'predictions'/f"{info['token']}.npz",prediction=prediction)
            results.append(item)
            row=dict(index=idx,scene=scene,token=info['token'],global_idx=frame,
                     load_ms=load_ms,model_ms=model_ms,
                     prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                     memory_nonzero_voxels=int((model.module.global_valid_masks[0]>0).sum().item()),
                     predicted_flow_shape=list(model.module.gt_flow.shape),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if history_control is not None:
                assert history_control.calls == history_control.attention_calls == len(rows)+1
                row['history_control'] = dict(history_control.last)
            rows.append(row)
            with (args.out/'frames.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            state.update(frames=len(seen),last_frame=row,elapsed_seconds=time.perf_counter()-run_start)
            save(args.out/'progress.json',state)
            if len(rows)%10==0: print('frames',len(rows),'model_ms',round(model_ms,1),'scene',scene,flush=True)
        metrics=dataset.evaluate(results,metric=['mIoU'],jsonfile_prefix=str(args.out/'official-metrics'))
        times=np.array([r['model_ms'] for r in rows])
        state.update(status='complete',finished=time.time(),metrics=metrics,
                     full_validation=len(seen)==6019 and len(last)==150,
                     elapsed_seconds=time.perf_counter()-run_start,
                     model_ms_p50=float(np.median(times)),model_ms_p95=float(np.percentile(times,95)),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(args.out/'summary.json',state);save(args.out/'progress.json',state)
        print(json.dumps(state,default=plain),flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error),updated=time.time())
        save(args.out/'progress.json',state)
        raise


if __name__=='__main__':
    main()
