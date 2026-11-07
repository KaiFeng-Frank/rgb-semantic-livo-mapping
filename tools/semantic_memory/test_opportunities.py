"""Validate temporal identity, causal window and diagnostic denominators."""
import unittest

from .audit_opportunities import audit


def row(frame, current, previous_raw, temporal, instance='a', stamp=None):
    return dict(scene='one', instance=instance, frame=frame, label=3,
                stamp=frame*.5 if stamp is None else stamp,
                correct_fraction=dict(no_history=current, baseline=previous_raw, temporal=temporal))


class OpportunityTests(unittest.TestCase):
    def test_only_same_instance_immediate_past_and_valid_age_count(self):
        rows = [row(0,1,1,1), row(1,0,0,1), row(2,0,1,0,'b'),
                row(3,0,0,0), row(4,0,0,0,stamp=2.5)]
        counts = audit(rows)['totals']
        self.assertEqual(counts['object_observations'],5)
        self.assertEqual(counts['previous_correct_current_wrong'],1)
        self.assertEqual(counts['opportunity_fixed'],1)
        self.assertEqual(counts['same_object_previous_frame'],1)
        self.assertEqual(counts['current_wrong'],4)

    def test_failures_and_harm_stay_in_counts_and_input_order_does_not_matter(self):
        rows = [row(0,1,1,1),row(1,0,1,0),row(2,0,0,1),row(3,1,0,0)]
        result = audit(list(reversed(rows)))
        c = result['totals']
        self.assertEqual(c['previous_correct_current_wrong'],2)
        self.assertEqual(c['opportunity_fixed'],1)
        self.assertEqual(c['opportunity_fixed_fraction'],.5)
        self.assertEqual(c['previous_wrong_current_correct'],1)
        self.assertEqual(c['risk_harmed'],1)
        self.assertEqual(result, audit(rows))

    def test_duplicate_identity_rejected_and_future_correctness_not_used(self):
        a = row(0,0,0,0)
        with self.assertRaises(ValueError):
            audit([a,a])
        result = audit([a,row(1,0,1,0)])['totals']
        self.assertEqual(result['previous_correct_current_wrong'],0)
        self.assertEqual(result['both_wrong'],1)


if __name__ == '__main__':
    unittest.main()
