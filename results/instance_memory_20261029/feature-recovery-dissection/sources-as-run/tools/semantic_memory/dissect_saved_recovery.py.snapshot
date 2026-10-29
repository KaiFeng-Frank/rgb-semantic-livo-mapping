"""Read saved lost-recovery episodes only; no inference, optimization, or selection."""
import argparse,gzip
from collections import Counter
from pathlib import Path
from tools.instance_memory.data import read,save
from tools.semantic_memory.nuscenes_pilot_data import digest


def compressed(path):
    with gzip.open(path,'rt') as f:
        import json
        return json.load(f)


def dissect(original,provenance,out):
    old=read(original/'attribution.json');traced=read(provenance/'report.json')
    fresh=compressed(provenance/'object-observations.json.gz')
    lookup={(r['scene'],r['instance'],r['frame']):r for r in fresh}
    result=[];all_cases=[];unique=set();sources={}
    for p in (original/'attribution.json',provenance/'report.json',provenance/'object-observations.json.gz'):sources[str(p)]=digest(p)
    for item in old['seeds']:
        seed=item['seed'];folder=original/'run'/f'seed-{seed}'
        op=folder/'object-observations.json.gz';sources[str(op)]=digest(op)
        rows=compressed(op);prior={(r['scene'],r['instance'],r['frame']):r for r in rows}
        episodes=item['lost_recovery_events'];original_keys={(e['scene'],e['instance'],e['start_frame'],e['end_frame']) for e in episodes}
        trace_keys={(e['scene'],e['instance'],e['start'],e['end']) for e in traced['lost_recovery_episodes'][str(seed)]}
        if original_keys!=trace_keys:raise ValueError('Provenance does not cover the same saved lost cases')
        counts=Counter();exact=0;diff=[];decisive=0;episode_results=[];lower=upper=0
        for ep in episodes:
            key=(ep['scene'],ep['instance'],ep['start_frame'],ep['end_frame']);unique.add(key)
            observed=[];case_frames=[]
            for f in ep['frames']:
                k=(ep['scene'],ep['instance'],f['frame']);o=prior[k];n=lookup[k];observed.append(n)
                if o['points']!=n['points']:raise ValueError('Object GT support changed')
                if o['correct_fraction']['balanced_temporal']>=.5:raise ValueError('Not an original lost episode')
                original_c=o['correct_fraction']['balanced_temporal_no_history'];original_t=o['correct_fraction']['balanced_temporal']
                current=n['correct_fraction'][f's{seed}_no_history'];temporal=n['correct_fraction'][f's{seed}_temporal']
                same=(original_c==current and original_t==temporal);exact+=same
                if not same:diff.append(dict(scene=k[0],instance=k[1],frame=k[2],original_current=original_c,trace_current=current,original_temporal=original_t,trace_temporal=temporal))
                if original_c<.5:continue
                if current<.5 or temporal>=.5:raise ValueError('Decisive counterfactual changed')
                v=n['provenance'][str(seed)];counts.update(v);decisive+=1
                if v['harm_no_history'] or v['harm_query_majority_mismatch']:raise ValueError('Fallback/voxel target assumptions do not hold')
                fallback=v['harm']-v['harm_any_correct'];retained_resolved=v['harm_resolved_correct']-fallback
                if not 0<=retained_resolved<=v['harm_any_correct']:raise ValueError('Oracle fallback accounting inconsistent')
                # Marginal counts only: use bounds, never invent exact intersections.
                lower+=max(0,v['harm_anchor_wrong']+v['harm_all_same']-v['harm_any_correct']-v['harm'])
                upper+=min(v['harm_anchor_wrong'],v['harm_all_same'],v['harm']-v['harm_any_correct'])
                case_frames.append(dict(frame=k[2],points=o['points'],original_current=original_c,original_temporal=original_t,
                    traced_current=current,traced_temporal=temporal,provenance=v,
                    corrected_by_empty_history_fallback=fallback,correct_candidates_retained=v['harm_any_correct'],
                    corrected_with_nonempty_correct_history=retained_resolved,
                    still_wrong_with_nonempty_correct_history=v['harm_any_correct']-retained_resolved))
            case=dict(seed=seed,scene=key[0],instance=key[1],start=key[2],end=key[3],label=ep['label'],
                anchor_never_succeeds=not any(x['correct_fraction']['anchor']>=.5 for x in observed),
                association_resolves=any(x['correct_fraction'][f's{seed}_oracle_association']>=.5 for x in observed),
                correct_prediction_filter_resolves=any(x['correct_fraction'][f's{seed}_oracle_correct']>=.5 for x in observed),
                decisive_frames=case_frames)
            episode_results.append(case);all_cases.append(case)
        epoch_rows=[]
        for epoch in range(1,6):
            path=folder/f'epoch-{epoch}.json';sources[str(path)]=digest(path)
            v=read(path)['development']['arms']['balanced_temporal']
            epoch_rows.append(dict(epoch=epoch,miou_percent=100*v['miou_fixed16'],foreground_percent=100*v['camera_foreground_accuracy']))
        sp=folder/'selected.json';sources[str(sp)]=digest(sp);selected=read(sp)['balanced_temporal']['epoch']
        actual=epoch_rows[selected-1];best_fg=max(epoch_rows,key=lambda r:r['foreground_percent'])
        if actual!=max(epoch_rows,key=lambda r:r['miou_percent']):raise ValueError('Saved selection differs from declared objective')
        fallback=counts['harm']-counts['harm_any_correct']
        result.append(dict(seed=seed,lost_episodes=len(episodes),decisive_frames=decisive,
            exact_fraction_frames=exact,numerically_different_frames=diff,
            association_resolves=sum(e['association_resolves'] for e in episode_results),
            correct_prediction_filter_resolves=sum(e['correct_prediction_filter_resolves'] for e in episode_results),
            anchor_never_succeeds=sum(e['anchor_never_succeeds'] for e in episode_results),point_counts=dict(counts),
            corrected_by_empty_history_fallback=fallback,corrected_with_nonempty_correct_history=counts['harm_resolved_correct']-fallback,
            still_wrong_with_nonempty_correct_history=counts['harm']-counts['harm_resolved_correct'],
            certified_anchor_wrong_same_identity_no_correct_candidate_intersection_bounds=[lower,upper],
            selection=dict(selected=actual,highest_saved_foreground=best_fg,
                foreground_sacrificed_pp=best_fg['foreground_percent']-actual['foreground_percent'],
                miou_selected_advantage_pp=actual['miou_percent']-best_fg['miou_percent'],
                epochs=epoch_rows,scope='Descriptive objective tradeoff only. No reselection; per-episode recovery was not saved for unselected epochs.')))
    out.mkdir(parents=True,exist_ok=False)
    report=dict(scope='Only the archived 14/18/19 lost same-weight recovery episodes. No new inference, training, thresholds or checkpoint selection.',
        episode_seed_records=len(all_cases),unique_scene_object_intervals=len(unique),seeds=result,sources_sha256=sources,
        limitations=['Three traced object-frame fractions differ numerically from the original extraction; all original lost-episode identities and decisive current-success/history-failure relations remain unchanged.',
          'Saved provenance contains masks/counts, not full candidate vectors, pooled vectors or intermediate logits. It cannot numerically separate feature aggregation from the neural readout.',
          'GT correctness filtering usually clears all candidates and invokes current fallback; this is not evidence of learnable online correctness prediction.',
          'Retention targets the old frozen anchor. This is an objective coverage gap, not an isolated causal proof that changing the loss would succeed.'])
    save(out/'report.json',report);save(out/'cases.json',all_cases)
    print('records',len(all_cases),'unique intervals',len(unique))
    for s in result:print(s['seed'],s['point_counts'],'fallback',s['corrected_by_empty_history_fallback'],'nonempty still wrong',s['still_wrong_with_nonempty_correct_history'],'selection gap',s['selection']['foreground_sacrificed_pp'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('original','provenance','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();dissect(a.original,a.provenance,a.out)
