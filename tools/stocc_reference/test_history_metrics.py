"""Boundary cases for paired event scoring; no GPU or model is needed."""
import unittest
import numpy as np
from history_metrics import build_events, event_summary, point_scores


def row(frame, cold, native, instance=1, scene='scene'):
    return dict(scene=scene,log='log',instance=instance,label=4,frame=frame,
                stamp=frame*.5,correct_fraction=dict(cold_history=cold,native_history=native))


class EventContract(unittest.TestCase):
    def test_scene_start_is_not_new_arrival_and_gap_is_retained(self):
        episodes=build_events([row(0,1,1),row(1,1,1),row(3,1,0)])
        result=event_summary(episodes)
        self.assertEqual(result['first_observed']['episodes'],0)
        gap=result['reappearance_after_observation_gap']
        self.assertEqual(gap['episodes'],1)
        self.assertEqual(gap['lost'],1)
        self.assertEqual(gap['arms']['native_history']['not_observed_recovered'],1)

    def test_recovery_after_fixed_horizon_is_not_counted(self):
        rows=[row(i,0,1 if i>=5 else 0) for i in range(7)]
        recovery=event_summary(build_events(rows))['cold_history_error_onset']
        self.assertEqual(recovery['episodes'],1)
        self.assertEqual(recovery['arms']['native_history']['successes'],0)
        self.assertEqual(recovery['arms']['native_history']['mean_delay_capped_at_2s'],2.)

    def test_gains_and_harms_use_common_population(self):
        episodes=build_events([row(1,0,1,instance=1),row(1,1,0,instance=2)])
        arrivals=event_summary(episodes)['first_observed']
        self.assertEqual(arrivals['episodes'],2)
        self.assertEqual((arrivals['gained'],arrivals['lost'],arrivals['net_successes']),(1,1,0))
        for arm in ('native_history','cold_history'):
            self.assertEqual(arrivals['arms'][arm]['success_rate'],.5)

    def test_same_numeric_identity_does_not_cross_scenes(self):
        episodes=build_events([row(0,1,1,scene='a'),row(3,1,0,scene='b')])
        result=event_summary(episodes)
        self.assertEqual(result['reappearance_after_observation_gap']['episodes'],0)
        self.assertEqual(result['first_observed']['episodes'],1)

    def test_predicted_free_on_a_semantic_point_is_an_error(self):
        matrix=np.zeros((18,18),np.int64);matrix[4,4]=5;matrix[4,17]=5
        scores=point_scores(matrix)
        self.assertEqual(scores['mIoU16_present_percent'],50.)
        self.assertEqual(scores['accuracy_percent'],50.)


if __name__=='__main__':unittest.main()
