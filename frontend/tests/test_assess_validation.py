import copy
import unittest
from frontend.assess_validation import assess


class Assessment(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(version=3, diagnostic=False, arms=['geometry', 'association_B0', 'motion_B0',
                        'visual_B0', 'likelihood_B0', 'enhanced_B0'], repetitions=3, minimum_poses=220)
        self.scores = dict(runs=[dict(arm=arm, repetition=rep, poses=235,
                                     ate_rmse_m=(1.0 if arm == 'geometry' else .8) + rep * .01,
                                     rpe_1s_translation_rmse_m=.1, rpe_1s_rotation_rmse_deg=.2,
                                     total_constraints=0, semantic_updates=235, likelihood_updates=230)
                               for arm in self.cfg['arms'] for rep in range(1, 4)])

    def test_consistent_improvement_and_frozen_mechanism_pass(self):
        self.assertEqual(assess(self.cfg, self.scores)['status'], 'passed')

    def test_mean_improvement_cannot_hide_one_paired_regression(self):
        self.scores['runs'][-1]['ate_rmse_m'] = 1.04
        self.assertEqual(assess(self.cfg, self.scores)['status'], 'not_passed')

    def test_rotation_regression_or_missing_mechanism_fails(self):
        for field, value in [('rpe_1s_rotation_rmse_deg', .3), ('likelihood_updates', 0)]:
            with self.subTest(field=field):
                s = copy.deepcopy(self.scores); s['runs'][-1][field] = value
                self.assertEqual(assess(self.cfg, s)['status'], 'not_passed')

    def test_duplicate_or_missing_run_is_not_a_valid_comparison(self):
        for rows in [self.scores['runs'][:-1], self.scores['runs'] + [self.scores['runs'][0]]]:
            with self.assertRaises(ValueError):assess(self.cfg, dict(runs=rows))


if __name__ == '__main__':unittest.main()
