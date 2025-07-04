import unittest
import pandas as pd
import numpy as np
from py3r.behaviour.tracking import Tracking
from py3r.behaviour.features import Features
from py3r.behaviour.summary import Summary, SummaryCollection, MultipleSummaryCollection
from py3r.behaviour.exceptions import BatchProcessError

class TestSummary(unittest.TestCase):
    def setUp(self):
        data = pd.DataFrame({
            'A.x': [0, 1, 2],
            'A.y': [0, 0, 0],
            'B.x': [0, 0, 0],
            'B.y': [0, 1, 2],
            'A.likelihood': [1, 1, 1],
            'B.likelihood': [1, 1, 1],
        }, index=[0, 1, 2])
        meta = {'fps': 1, 'rescale_distance_method': 'dummy', 'smoothing': 'dummy'}
        tracking = Tracking(data, meta, handle='test_summary_tracking')
        features = Features(tracking)
        self.summary = Summary(features)

    def test_count_onset(self):
        s = pd.Series([False, True, True, False, True])
        self.assertEqual(self.summary.count_onset(s), 2)

    def test_time_true(self):
        s = pd.Series([True, True, False, True])
        self.assertEqual(self.summary.time_true(s), 3.0)

    def test_total_distance(self):
        d = self.summary.total_distance('A')
        self.assertTrue(isinstance(d, float))

    def test_store(self):
        self.summary.store(42, 'answer')
        self.assertIn('answer', self.summary.data)

    def test_make_bin(self):
        binned = self.summary.make_bin(0, 1)
        self.assertIsInstance(binned, Summary)

    def test_make_bins(self):
        bins = self.summary.make_bins(2)
        self.assertEqual(len(bins), 2)

# class TestSummaryCollection(unittest.TestCase):
#     def setUp(self):
#         data1 = pd.DataFrame({'A.x': [0, 1], 'A.y': [0, 0], 'A.likelihood': [1, 1]}, index=[0, 1])
#         data2 = pd.DataFrame({'A.x': [2, 3], 'A.y': [1, 1], 'A.likelihood': [1, 1]}, index=[0, 1])
#         meta = {'fps': 1, 'rescale_distance_method': 'dummy', 'smoothing': 'dummy'}
#         t1 = Tracking(data1.copy(), meta.copy(), handle='t1')
#         t2 = Tracking(data2.copy(), meta.copy(), handle='t2')
#         f1 = Features(t1, handle='f1')
#         f2 = Features(t2, handle='f2')
#         s1 = Summary(f1, handle='s1')
#         s2 = Summary(f2, handle='s2')
#         self.sc = SummaryCollection({'t1': s1, 't2': s2})

#     def test_batch_process(self):
#         # Test __getattr__ batch method for total_distance
#         dists = self.sc.total_distance('A')
#         self.assertEqual(set(dists.keys()), {'t1', 't2'})
#         self.assertTrue(isinstance(dists['t1'], float))
#         self.assertTrue(isinstance(dists['t2'], float))

#     def test_factory_from_list(self):
#         s1, s2 = self.sc.summary_dict['t1'], self.sc.summary_dict['t2']
#         sc2 = SummaryCollection.from_list([s1, s2])
#         self.assertEqual(set(sc2.summary_dict.keys()), {'t1', 't2'})

#     def test_error_in_batch(self):
#         # Add a method that will raise
#         def bad_method():
#             raise ValueError("fail!")
#         self.sc.summary_dict['t1'].bad_method = bad_method
#         with self.assertRaises(Exception):
#             self.sc.bad_method()

# class TestMultipleSummaryCollections(unittest.TestCase):
#     def setUp(self):
#         data1 = pd.DataFrame({'A.x': [0, 1], 'A.y': [0, 0], 'A.likelihood': [1, 1]}, index=[0, 1])
#         data2 = pd.DataFrame({'A.x': [2, 3], 'A.y': [1, 1], 'A.likelihood': [1, 1]}, index=[0, 1])
#         meta = {'fps': 1, 'rescale_distance_method': 'dummy', 'smoothing': 'dummy'}
#         t1 = Tracking(data1.copy(), meta.copy(), handle='t1')
#         t2 = Tracking(data2.copy(), meta.copy(), handle='t2')
#         f1 = Features(t1, handle='f1')
#         f2 = Features(t2, handle='f2')
#         s1 = Summary(f1, handle='s1')
#         s2 = Summary(f2, handle='s2')
#         sc1 = SummaryCollection({'t1': s1})
#         sc2 = SummaryCollection({'t2': s2})
#         self.msc = MultipleSummaryCollections({'sc1': sc1, 'sc2': sc2})

#     def test_multiple_batch(self):
#         # Test __getattr__ batch method for total_distance
#         dists = self.msc.total_distance('A')
#         self.assertEqual(set(dists.keys()), {'sc1', 'sc2'})
#         self.assertTrue(isinstance(dists['sc1']['t1'], float))
#         self.assertTrue(isinstance(dists['sc2']['t2'], float))

#     def test_error_in_multiple_batch(self):
#         # Add a method that will raise
#         def bad_method():
#             raise ValueError("fail!")
#         self.msc.dict_of_summary_collections['sc1'].summary_dict['t1'].bad_method = bad_method
#         with self.assertRaises(Exception):
#             self.msc.bad_method()

if __name__ == '__main__':
    unittest.main()
