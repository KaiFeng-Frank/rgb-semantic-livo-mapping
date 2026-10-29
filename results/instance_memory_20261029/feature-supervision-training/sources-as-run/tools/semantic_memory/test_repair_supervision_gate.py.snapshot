"""Regression checks for the preregistered gate, especially history attribution."""
import copy
import unittest
from .repair_supervision_experiment import ARMS,ABLATIONS,STATIC,summarize


def result():
    names=STATIC+ARMS+ABLATIONS
    arms={k:dict(miou_fixed16=.5,camera_foreground_accuracy=.8) for k in names}
    arms['balanced_temporal']=dict(miou_fixed16=.501,camera_foreground_accuracy=.8)
    return dict(arms=arms,per_scene={str(i):copy.deepcopy(arms) for i in range(8)},
                events=dict(baseline_error_recovery={k:dict(success_rate=.7) for k in names}))


class GateTests(unittest.TestCase):
    def test_apparent_gain_cannot_hide_same_head_current_control(self):
        reports=[result() for _ in range(3)]
        self.assertTrue(summarize(reports)['gate']['passed'])
        for r in reports:r['arms']['balanced_temporal_no_history']['miou_fixed16']=.502
        self.assertFalse(summarize(reports)['gate']['passed'])

    def test_mean_improvement_cannot_hide_one_failed_seed_or_recovery(self):
        reports=[result() for _ in range(3)]
        reports[0]['arms']['balanced_temporal']['miou_fixed16']=.4999
        s=summarize(reports)
        self.assertGreater(s['gate']['mean_miou_gain_pp'],.05)
        self.assertFalse(s['gate']['passed'])
        reports=[result() for _ in range(3)]
        reports[1]['events']['baseline_error_recovery']['balanced_temporal']['success_rate']=.6
        self.assertFalse(summarize(reports)['gate']['passed'])

    def test_foreground_and_log_breadth_remain_required(self):
        reports=[result() for _ in range(3)]
        reports[0]['arms']['balanced_temporal']['camera_foreground_accuracy']=.7989
        self.assertFalse(summarize(reports)['gate']['passed'])
        reports=[result() for _ in range(3)]
        for i in range(4):reports[2]['per_scene'][str(i)]['balanced_temporal']['miou_fixed16']=.5
        self.assertFalse(summarize(reports)['gate']['passed'])


if __name__=='__main__':unittest.main()
