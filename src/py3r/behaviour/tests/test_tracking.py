import unittest
import pandas as pd
import numpy as np
from py3r.behaviour.tracking import Tracking, LoadOptions, TrackingCollection, MultipleTrackingCollection
from py3r.behaviour.exceptions import BatchProcessError

class TestTracking(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A.x': [0.0, 1.2, np.nan],
            'A.y': [0.0, 0.0, 0.0],
            'A.likelihood': [1.0, 0.3, 0.8],
            'B.x': [0.0, 0.0, 0.0],
            'B.y': [0.0, 1.5, 2.2],
            'B.likelihood': [0.6, 0.4, 1.0],
        }, index=[0, 1, 2])
        self.meta = {'fps': 1}
        self.tracking = Tracking(self.data.copy(), self.meta.copy(), handle='test_tracking')

    def test_constructor(self):
        self.assertIsInstance(self.tracking, Tracking)

    def test_from_dlc(self):
        options = LoadOptions(fps=1.0)
        t = Tracking.from_dlc('util/tests/test_dlc.csv', handle='test_tracking', options=options)
        self.assertIsInstance(t, Tracking)
        self.assertEqual(t.data.shape, (3, 2))

    def test_from_dlcma(self):
        options = LoadOptions(fps=1.0)
        t = Tracking.from_dlcma('util/tests/test_dlcma.csv', handle='test_tracking', options=options)
        self.assertIsInstance(t, Tracking)
        self.assertEqual(t.data.shape, (3, 3))

    def test_from_yolo3r(self):
        options = LoadOptions(fps=1.0)
        t = Tracking.from_yolo3r('util/tests/test_yolo3r.csv', handle='test_tracking', options=options)
        self.assertIsInstance(t, Tracking)
        self.assertEqual(t.data.shape[0], 3)
        self.assertTrue(any(['nose.x' in c for c in t.data.columns]))

    def test_apply_aspectratio_correction(self):
        df = Tracking._apply_aspectratio_correction(self.data, 2.0)
        self.assertTrue(np.allclose(df['A.x'].fillna(0), [0, 2.4, 0]))

    def test_add_usermeta(self):
        self.tracking.add_usermeta({'foo': 'bar'}, overwrite=True)
        self.assertIn('usermeta', self.meta)

    def test_strip_column_names(self):
        self.tracking.strip_column_names()
        self.assertTrue(all(['.' in c for c in self.tracking.data.columns]))

    def test_time_as_expected(self):
        self.assertTrue(self.tracking.time_as_expected(0, 10))

    def test_trim(self):
        self.tracking.trim(1, 2)
        self.assertEqual(len(self.tracking.data), 2)

    def test_filter_likelihood(self):
        self.tracking.filter_likelihood(0.5)
        # A.likelihood at index 1 is 0.3, so A.x and A.y at index 1 should be np.nan
        self.assertTrue(np.isnan(self.tracking.data.loc[1, 'A.x']))
        # B.likelihood at index 1 is 0.4, so B.x and B.y at index 1 should be np.nan
        self.assertTrue(np.isnan(self.tracking.data.loc[1, 'B.x']))
        # A.likelihood at index 0 is 1.0, so A.x at index 0 should not be np.nan
        self.assertFalse(np.isnan(self.tracking.data.loc[0, 'A.x']))

    def test_find_2point_distance(self):
        dist = self.tracking.find_2point_distance('A', 'B')
        self.assertTrue(isinstance(dist, pd.Series))

    def test_get_tracked_point_names(self):
        names = self.tracking.get_tracked_point_names()
        self.assertIn('A', names)
        self.assertIn('B', names)

    def test_rescale_distance_scalar(self):
        self.tracking.rescale_distance_scalar('A', 'B', 2.0)
        self.assertIn('rescale_distance_method', self.tracking.meta)

    def test_gen_partial_smoothdict(self):
        d = self.tracking.gen_partial_smoothdict(['A'], 3, 'mean')
        self.assertIn('A', d)

    def test_gen_smoothdict(self):
        d = self.tracking.gen_smoothdict([['A']], [3], ['mean'])
        self.assertIn('A', d)

    def test_smooth_tracking(self):
        params = {'A': {'window': 2, 'type': 'mean'}, 'B': {'window': 2, 'type': 'mean'}}
        self.tracking.smooth_tracking(params)
        self.assertIn('smoothing', self.tracking.meta)

    def test_repr(self):
        r = repr(self.tracking)
        self.assertIn('Tracking', r)

# class TestTrackingCollection(unittest.TestCase):
#     def setUp(self):
#         self.data1 = pd.DataFrame({'A.x': [0, 1], 'A.y': [0, 0], 'A.likelihood': [1, 1]}, index=[0, 1])
#         self.data2 = pd.DataFrame({'A.x': [2, 3], 'A.y': [1, 1], 'A.likelihood': [1, 1]}, index=[0, 1])
#         self.meta = {'fps': 1}
#         self.t1 = Tracking(self.data1.copy(), self.meta.copy(), handle='t1')
#         self.t2 = Tracking(self.data2.copy(), self.meta.copy(), handle='t2')
#         self.tc = TrackingCollection({'t1': self.t1, 't2': self.t2})

#     def test_batch_process(self):
#         # Test __getattr__ batch method for get_tracked_point_names
#         names = self.tc.get_tracked_point_names()
#         self.assertEqual(set(names.keys()), {'t1', 't2'})
#         self.assertIn('A', names['t1'])
#         self.assertIn('A', names['t2'])

#     def test_factory_from_list(self):
#         tc2 = TrackingCollection.from_list([self.t1, self.t2])
#         self.assertEqual(set(tc2.tracking_dict.keys()), {'t1', 't2'})

#     def test_factory_from_dlc(self):
#         # Just test that it runs and returns correct keys (no file IO)
#         tc3 = TrackingCollection({'t1': self.t1})
#         self.assertIn('t1', tc3.tracking_dict)

#     def test_error_in_batch(self):
#         # Add a method that will raise
#         def bad_method():
#             raise ValueError("fail!")
#         self.t1.bad_method = bad_method
#         with self.assertRaises(Exception):
#             self.tc.bad_method()

# class TestMultipleTrackingCollection(unittest.TestCase):
#     def setUp(self):
#         self.data1 = pd.DataFrame({'A.x': [0, 1], 'A.y': [0, 0], 'A.likelihood': [1, 1]}, index=[0, 1])
#         self.data2 = pd.DataFrame({'A.x': [2, 3], 'A.y': [1, 1], 'A.likelihood': [1, 1]}, index=[0, 1])
#         self.meta = {'fps': 1}
#         t1 = Tracking(self.data1.copy(), self.meta.copy(), handle='t1')
#         t2 = Tracking(self.data2.copy(), self.meta.copy(), handle='t2')
#         tc1 = TrackingCollection({'t1': t1})
#         tc2 = TrackingCollection({'t2': t2})
#         self.mtc = MultipleTrackingCollection({'tc1': tc1, 'tc2': tc2})

#     def test_multiple_batch(self):
#         # Test __getattr__ batch method for get_tracked_point_names
#         names = self.mtc.get_tracked_point_names()
#         self.assertEqual(set(names.keys()), {'tc1', 'tc2'})
#         self.assertIn('A', names['tc1']['t1'])
#         self.assertIn('A', names['tc2']['t2'])

#     def test_error_in_multiple_batch(self):
#         # Add a method that will raise
#         def bad_method():
#             raise ValueError("fail!")
#         self.mtc.tracking_collections['tc1'].tracking_dict['t1'].bad_method = bad_method
#         with self.assertRaises(Exception):
#             self.mtc.bad_method()

if __name__ == '__main__':
    unittest.main()