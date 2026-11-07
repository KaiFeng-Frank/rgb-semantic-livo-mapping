"""Check evaluation denominators, event censoring and GT box axes before scoring."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from tools.semantic_memory.pilot_metrics import confusion, scores, object_groups, events, paired_summary
from tools.semantic_memory.nuscenes_pilot_data import records, LABEL_MAP


class PilotMetricTests(unittest.TestCase):
    def test_confusion_keeps_ignored_points_out_and_counts_harm(self):
        y=np.array([0,1,1,-1])
        base=np.array([0,0,1,2]); pred=np.array([1,1,1,0])
        cm=confusion(y,pred)
        self.assertEqual(cm.sum(),3)
        self.assertEqual(scores(cm)['accuracy'],2/3)
        self.assertEqual(paired_summary(y,base,pred),dict(rescued=1,harmed=1,base_correct=2,base_wrong=1))

    def test_inverse_scatter_scores_original_points_not_voxel_majority(self):
        inv=np.array([0,0,1]); predictions=np.array([0,1])[inv]
        y=np.array([0,1,1])
        self.assertEqual(confusion(y,predictions).tolist()[:2],
                         [[1]+[0]*15,[1,1]+[0]*14])

    def test_rotated_box_width_length_axes_and_ambiguous_membership(self):
        obj=dict(instance='x',label=3,translation=[10,20,0],rotation=[np.sqrt(.5),0,0,np.sqrt(.5)],
                 size=[2,8,2],annotated_lidar_points=8)
        points=np.array([[10,20+y,0] for y in (-3,-2,-1,0,1,2,3)]+[[13,20,0]],float)
        labels=np.full(8,3)
        groups=object_groups(points,labels,[obj])
        self.assertEqual(groups[0]['indices'].tolist(),list(range(7)))
        self.assertEqual(object_groups(points,labels,[obj,obj]),[])

    def test_left_censoring_reappearance_and_never_recovered_denominators(self):
        rows=[]
        for frame,b,t in [(0,0,0),(1,1,0),(3,0,1),(4,1,1)]:
            rows.append(dict(scene='a',instance='x',frame=frame,stamp=frame*.5,
                             correct_fraction=dict(baseline=b,temporal=t)))
        result=events(rows,('baseline','temporal'))
        self.assertEqual(result['first_observed']['baseline']['episodes'],0)
        gap=result['reappearance_after_observation_gap']
        self.assertEqual(gap['baseline']['conditional_mean_delay_s'],.5)
        self.assertEqual(gap['temporal']['conditional_mean_delay_s'],0)
        recovery=result['baseline_error_recovery']
        self.assertEqual(recovery['temporal']['episodes'],2)
        self.assertEqual(recovery['temporal']['successes'],1)
        self.assertEqual(recovery['temporal']['censored'],1)

    def test_json_metadata_stream_preserves_records_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.json'
            p.write_text('[\n{"a":[1,2],"s":"} , ["},\n{"a":3}\n]')
            self.assertEqual(list(records(p)),[dict(a=[1,2],s='} , ['),dict(a=3)])
            p.write_text('[{"a":3}')
            with self.assertRaises(ValueError):
                list(records(p))

    def test_official_labels_match_native_model_classes(self):
        self.assertEqual(LABEL_MAP[[2,3,4,6,9,17,24,30]].tolist(),[6,6,6,6,0,3,10,15])
        self.assertEqual(sorted(set(LABEL_MAP.tolist())),list(range(-1,16)))


if __name__=='__main__':
    unittest.main()
