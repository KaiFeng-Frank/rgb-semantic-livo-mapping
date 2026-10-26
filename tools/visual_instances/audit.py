"""Evaluation-only evidence for identity changes versus semantic label changes.

The GT-box vote is a proxy. Save its support and ambiguity rather than treating
every majority change as a proven tracker switch. Never used by online inference.
"""
from collections import Counter
import json

import numpy as np


class AssociationAudit:
    def __init__(self, out):
        self.out = out
        self.history = {}
        self.observations, self.pairs = [], []

    def step(self, scene, frame_index, row, src, frame, result, gt, object_ids,
             tracked_boxes):
        for slot, tid in enumerate(frame.ids):
            at = np.flatnonzero(result['ids'] == tid)
            votes = Counter(object_ids[at][object_ids[at] >= 0].tolist())
            # Sort with the original encounter-order tie rule, as in evaluate.py.
            objects = []
            for oi, count in votes.most_common():
                obj = row['objects'][oi]
                objects.append(dict(token=obj['instance'], points=count, label=obj['label'],
                                    world_center=obj['translation'], size=obj['size']))
            yy, xx = np.nonzero(frame.index == slot+1)
            obs = dict(scene=scene, frame=frame_index, stamp_ns=frame.stamp_ns,
                image_path=src['image_path'], image_sha256=src['image_sha256'],
                session=frame.session, track_id=int(tid), visual_label=int(frame.labels[slot]),
                score=float(frame.scores[slot]), detector_bbox=tracked_boxes.get(int(tid)),
                interior_mask_bbox=[int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if len(xx) else None,
                interior_mask_pixels=len(xx), associated_points=len(at),
                gt_label_counts=dict(Counter(gt[at].tolist())), objects=objects,
                box_supported_points=sum(votes.values()),
                dominant_share_of_supported=(objects[0]['points']/sum(votes.values())) if votes else None,
                dominant_share_of_associated=(objects[0]['points']/len(at)) if votes else None,
                tied_majority=len(objects)>1 and objects[0]['points']==objects[1]['points'])
            self.observations.append(obs)
            if sum(votes.values()) < 3:
                continue
            key = (scene, frame.session, int(tid))
            prev = self.history.get(key)
            if prev is not None and prev['frame'] == frame_index-1:
                same_object = prev['objects'][0]['token'] == objects[0]['token']
                self.pairs.append(dict(scene=scene, previous_frame=prev['frame'], frame=frame_index,
                    track_id=int(tid), same_gt_object=same_object,
                    visual_label_changed=prev['visual_label'] != obs['visual_label'],
                    gt_class_changed=prev['objects'][0]['label'] != objects[0]['label'],
                    previous_visual_correct=prev['visual_label'] == prev['objects'][0]['label'],
                    current_visual_correct=obs['visual_label'] == objects[0]['label'],
                    ambiguous_vote=prev['tied_majority'] or obs['tied_majority'],
                    previous=prev, current=obs))
            self.history[key] = obs

    def finish(self):
        wrong = [p for p in self.pairs if not p['same_gt_object']]
        same = [p for p in self.pairs if p['same_gt_object']]
        strong = lambda p: all(o['box_supported_points'] >= 10 and
            o['dominant_share_of_supported'] >= .8 and o['dominant_share_of_associated'] >= .8
            for o in [p['previous'], p['current']])
        counts = dict(adjacent_track_pairs=len(self.pairs), wrong_identity_pairs=len(wrong),
            wrong_identity_same_gt_class=sum(not p['gt_class_changed'] for p in wrong),
            wrong_identity_both_visual_labels_correct=sum(p['previous_visual_correct'] and p['current_visual_correct'] for p in wrong),
            wrong_identity_with_tied_vote=sum(p['ambiguous_vote'] for p in wrong),
            wrong_identity_strong_box_support=sum(strong(p) for p in wrong),
            same_identity_pairs=len(same),
            same_identity_visual_label_changes=sum(p['visual_label_changed'] for p in same),
            same_identity_both_visual_correct=sum(p['previous_visual_correct'] and p['current_visual_correct'] for p in same),
            same_identity_both_visual_wrong=sum(not p['previous_visual_correct'] and not p['current_visual_correct'] for p in same),
            same_identity_visual_correct_to_wrong=sum(p['previous_visual_correct'] and not p['current_visual_correct'] for p in same),
            same_identity_visual_wrong_to_correct=sum(not p['previous_visual_correct'] and p['current_visual_correct'] for p in same))
        (self.out/'instance-observations.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in self.observations))
        (self.out/'identity-pairs.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in self.pairs))
        (self.out/'identity-audit.json').write_text(json.dumps(dict(counts=counts,
            scope='Evaluation-only GT-box majority proxy of raw visual associations, independent of semantic fusion.',
            support_diagnostic='Both observations have >=10 box-supported points, >=80% dominant support and >=80% dominant/associated. This does not alter the original metric.'), indent=2)+'\n')
        return counts
