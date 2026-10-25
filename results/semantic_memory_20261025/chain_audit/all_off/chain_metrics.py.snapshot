"""Object-frame evidence-chain counts, with explicit conditional denominators."""
from collections import Counter, defaultdict


STAGES = ('raw_identity', 'stored', 'readable', 'queried', 'local', 'shortlist', 'retrieved', 'repaired')


def stages(row):
    """Cumulative ANY-hit funnel; not sufficient coverage for object recovery."""
    tests = (row['raw_good_voxels']>0, row['stored_good_voxels']>0,
             row['readable_good_voxels']>0, row['queried_error_points']>0,
             row['local_good_error_points']>0, row['shortlist_good_error_points']>0,
             row['retrieved_good_error_points']>0, row['correct_fraction']['temporal']>=.5)
    alive, result = True, {}
    for name, value in zip(STAGES,tests):
        alive = alive and value
        result[name] = alive
    return result


def aggregate(rows):
    totals = Counter()
    funnel = Counter()
    first_loss = Counter()
    by_scene, by_class = defaultdict(list), defaultdict(list)
    for row in rows:
        c = row['correct_fraction']
        previous = row['previous_raw_correct_fraction']
        wrong = c['no_history'] < .5
        opportunity = wrong and previous is not None and previous>=.5
        totals['observations'] += 1
        totals['current_wrong'] += wrong
        totals['anchor_current_wrong'] += c['anchor']<.5
        totals['anchor_opportunities'] += c['anchor']<.5 and previous is not None and previous>=.5
        totals['all_object_repairs'] += wrong and c['temporal']>=.5
        totals['all_object_harms'] += not wrong and c['temporal']<.5
        if not opportunity:
            continue
        totals['opportunities'] += 1
        totals['opportunity_anchor_already_correct'] += c['anchor']>=.5
        totals['no_error_query_due_confidence'] += row['queried_error_points']==0 and row['confidence_error_points']==0
        totals['no_error_query_due_budget'] += row['queried_error_points']==0 and row['confidence_error_points']>0
        by_scene[row['scene']].append(row)
        by_class[str(row['label'])].append(row)
        counts = stages(row)
        for k,v in counts.items():
            funnel[k] += v
        first_loss[next((k for k,v in counts.items() if not v),'complete')] += 1
        for key in ('temporal','clean_existing_gt','supply_raw_gt'):
            totals[key+'_repairs'] += c[key]>=.5
        need = row['points_needed']
        for source in ('confidence_error_points','queried_error_points','bounded_reachable_error_points',
                       'retrieved_good_error_points','supplied_good_error_points'):
            totals[source+'_enough_for_recognition'] += row[source]>=need
        totals['retrieved_any_but_not_repaired'] += row['retrieved_good_error_points']>0 and c['temporal']<.5
        totals['retrieved_enough_but_not_repaired'] += row['retrieved_good_error_points']>=need and c['temporal']<.5
        totals['supplied_enough_but_not_repaired'] += row['supplied_good_error_points']>=need and c['supply_raw_gt']<.5
    def conditional(items):
        n = len(items)
        return dict(opportunities=n,
            repairs={key:sum(r['correct_fraction'][key]>=.5 for r in items)
                     for key in ('temporal','clean_existing_gt','supply_raw_gt')},
            funnel={stage:sum(stages(r)[stage] for r in items) for stage in STAGES},
            query_enough=sum(r['queried_error_points']>=r['points_needed'] for r in items),
            bounded_enough=sum(r['bounded_reachable_error_points']>=r['points_needed'] for r in items),
            retrieved_enough=sum(r['retrieved_good_error_points']>=r['points_needed'] for r in items),
            supplied_enough=sum(r['supplied_good_error_points']>=r['points_needed'] for r in items))
    ordered = {k:funnel[k] for k in STAGES}
    values = list(ordered.values())
    if values != sorted(values,reverse=True) or sum(first_loss.values()) != totals['opportunities']:
        raise AssertionError('Invalid funnel accounting')
    return dict(totals=dict(totals),funnel=ordered,first_loss=dict(first_loss),
        by_scene={k:conditional(v) for k,v in sorted(by_scene.items())},
        by_class={k:conditional(v) for k,v in sorted(by_class.items(),key=lambda p:int(p[0]))})
