import unittest
from fractions import Fraction as F
from recovery import paired_summary, score_ranking, subspace_recall


class RecoveryTests(unittest.TestCase):
    def test_positive_control(self):
        self.assertEqual(score_ranking(['p','x','y'], ['p','x','y'], ['p'], 1)['recall'], 1)

    def test_high_degree_decoy_can_win_without_mechanism(self):
        self.assertEqual(score_ranking(['x','y','p'], ['p','x','y'], ['p'], 1)['recall'], 0)

    def test_matched_budget(self):
        out = score_ranking(['x','p','y'], ['p','x','y'], ['p'], 2)
        self.assertEqual((out['precision'], out['recall']), (F(1,2), 1))

    def test_invalid_rankings_and_budgets(self):
        for ranking, universe, plant, budget in [(['p','p'],['p','x'],['p'],1),
                (['p'],['p','x'],['p'],1), (['p'],['p'],['z'],1),
                (['p'],['p'],['p'],True), (['p'],['p'],['p'],0),
                (['p'],['p'],['p'],2), ([],[],[],1)]:
            with self.subTest(ranking=ranking, budget=budget), self.assertRaises(ValueError):
                score_ranking(ranking,universe,plant,budget)

    def test_paired_cases(self):
        out = paired_summary({'s0':1,'s1':0}, {'s0':0,'s1':1})
        self.assertEqual(out['mean_difference'], 0)
        self.assertEqual(out['verdict'], 'descriptive_only')

    def test_unmatched_or_invalid_cases(self):
        for a,b in [({},{}), ({'a':1},{'b':1}), ({'a':float('nan')},{'a':0}),
                    ({'a':True},{'a':0}), ({'a':2},{'a':0})]:
            with self.assertRaises(ValueError): paired_summary(a,b)

    def test_subspace_exact_and_orthogonal(self):
        p=((1,0),(0,0)); q=((0,0),(0,1))
        self.assertEqual(subspace_recall(p,p), 1)
        self.assertEqual(subspace_recall(q,p), 0)

    def test_rotated_subspace_and_rank_penalty_needed(self):
        p=((F(9,25),F(12,25)),(F(12,25),F(16,25)))
        self.assertEqual(subspace_recall(p,p), 1)
        self.assertEqual(subspace_recall(((1,0),(0,1)),p), 1)
        # Hence rank/cost MUST be matched before comparing recall.

    def test_invalid_projectors(self):
        for p in [(), ((0,),), ((2,),), ((1,1),(0,0)), ((0.5,),)]:
            with self.assertRaises(ValueError): subspace_recall(p,p)


if __name__ == '__main__': unittest.main()
