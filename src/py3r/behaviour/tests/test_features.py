import unittest
import pandas as pd
import numpy as np
from py3r.behaviour.tracking import Tracking
from py3r.behaviour.features import Features, FeaturesCollection, MultipleFeaturesCollection
from py3r.behaviour.exceptions import BatchProcessError

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Create synthetic tracking data
        data = pd.DataFrame({
            'A.x': [0, 1, 2],
            'A.y': [0, 0, 0],
            'B.x': [0, 0, 0],
            'B.y': [0, 1, 2],
            'A.likelihood': [1, 1, 1],
            'B.likelihood': [1, 1, 1],
        }, index=[0, 1, 2])
        meta = {'fps': 1, 'rescale_distance_method': 'dummy', 'smoothing': 'dummy'}
        self.tracking = Tracking(data, meta, handle='test_features_tracking')
        self.features = Features(self.tracking)

    def test_distance_from(self):
        dist = self.features.distance_from('A', 'B')
        self.assertTrue(np.allclose(dist, [0, np.sqrt(2), np.sqrt(8)]))

    def test_within_distance(self):
        within = self.features.within_distance('A', 'B', 2)
        self.assertTrue(np.array_equal(within.values, [True, True, False]))

    def test_get_point_median(self):
        median = self.features.get_point_median('A')
        self.assertEqual(median, (1.0, 0.0))

    def test_define_boundary(self):
        boundary = self.features.define_boundary(['A', 'B'], 1)
        self.assertEqual(len(boundary), 2)

    def test_within_boundary(self):
        # Use static boundary
        boundary = [(0,0), (0,2), (2,2), (2,0)]
        within = self.features.within_boundary('A', boundary, median=False)
        self.assertTrue(within.iloc[0])

    def test_find_angle(self):
        angle = self.features.find_angle('A', 'B')
        self.assertTrue(np.allclose(angle, [0, np.pi/4, np.pi/4]))

    def test_find_angle_deviation(self):
        dev = self.features.find_angle_deviation('A', 'B', 'A')
        self.assertTrue(np.allclose(dev, [0, 0, 0]))

    def test_within_angle_deviation(self):
        within = self.features.within_angle_deviation('A', 'B', 'A', 0.1)
        self.assertTrue(np.all(within))

    def test_find_speed(self):
        speed = self.features.find_speed('A')
        self.assertTrue(np.allclose(speed, [np.nan, 1, 1]))

    def test_above_speed(self):
        above = self.features.above_speed('A', 0.5)
        self.assertTrue(np.all(above[1:]))

    def test_all_above_speed(self):
        all_above = self.features.all_above_speed(['A', 'B'], 0.5)
        self.assertTrue(np.all(all_above[1:]))

    def test_below_speed(self):
        below = self.features.below_speed('A', 2)
        self.assertTrue(np.all(below[1:]))

    def test_all_below_speed(self):
        all_below = self.features.all_below_speed(['A', 'B'], 2)
        self.assertTrue(np.all(all_below[1:]))

    def test_distance_change(self):
        change = self.features.distance_change('A')
        print(change)
        self.assertTrue(np.allclose(change, [np.nan, 1, 1]))

    def test_store_and_smooth_feature(self):
        s = pd.Series([1,2,3])
        self.features.store(s, 'test')
        smoothed = self.features.smooth_feature('test', 'mean', 2)
        self.assertIsInstance(smoothed, pd.Series)

class TestFeaturesCollection(unittest.TestCase):
    def setUp(self):
        data1 = pd.DataFrame({'A.x': [0, 1], 'A.y': [0, 0], 'A.likelihood': [1, 1]}, index=[0, 1])
        data2 = pd.DataFrame({'A.x': [2, 3], 'A.y': [1, 1], 'A.likelihood': [1, 1]}, index=[0, 1])
        meta = {'fps': 1, 'rescale_distance_method': 'dummy', 'smoothing': 'dummy'}
        t1 = Tracking(data1.copy(), meta.copy(), handle='t1')
        t2 = Tracking(data2.copy(), meta.copy(), handle='t2')
        f1 = Features(t1, handle='f1')
        f2 = Features(t2, handle='f2')
        self.fc = FeaturesCollection({'t1': f1, 't2': f2})

    def test_batch_process(self):
        # Test __getattr__ batch method for get_point_median
        medians = self.fc.get_point_median('A')
        self.assertEqual(set(medians.keys()), {'t1', 't2'})
        self.assertEqual(medians['t1'], (0.5, 0.0))
        self.assertEqual(medians['t2'], (2.5, 1.0))

    def test_factory_from_list(self):
        f1, f2 = self.fc.features_dict['t1'], self.fc.features_dict['t2']
        fc2 = FeaturesCollection.from_list([f1, f2])
        self.assertEqual(set(fc2.features_dict.keys()), {'t1', 't2'})

    def test_error_in_batch(self):
        # Add a method that will raise
        def bad_method():
            raise ValueError("fail!")
        self.fc.features_dict['t1'].bad_method = bad_method
        with self.assertRaises(Exception):
            self.fc.bad_method()

class TestMultipleFeaturesCollection(unittest.TestCase):
    def setUp(self):
        data1 = pd.DataFrame({'A.x': [0, 1], 'A.y': [0, 0], 'A.likelihood': [1, 1]}, index=[0, 1])
        data2 = pd.DataFrame({'A.x': [2, 3], 'A.y': [1, 1], 'A.likelihood': [1, 1]}, index=[0, 1])
        meta = {'fps': 1, 'rescale_distance_method': 'dummy', 'smoothing': 'dummy'}
        t1 = Tracking(data1.copy(), meta.copy(), handle='t1')
        t2 = Tracking(data2.copy(), meta.copy(), handle='t2')
        f1 = Features(t1, handle='f1')
        f2 = Features(t2, handle='f2')
        fc1 = FeaturesCollection({'t1': f1})
        fc2 = FeaturesCollection({'t2': f2})
        self.mfc = MultipleFeaturesCollection({'fc1': fc1, 'fc2': fc2})

    def test_multiple_batch(self):
        # Test __getattr__ batch method for get_point_median
        medians = self.mfc.get_point_median('A')
        self.assertEqual(set(medians.keys()), {'fc1', 'fc2'})
        self.assertEqual(medians['fc1']['t1'], (0.5, 0.0))
        self.assertEqual(medians['fc2']['t2'], (2.5, 1.0))

    def test_error_in_multiple_batch(self):
        # Add a method that will raise
        def bad_method():
            raise ValueError("fail!")
        self.mfc.features_collections['fc1'].features_dict['t1'].bad_method = bad_method
        with self.assertRaises(Exception):
            self.mfc.bad_method()

if __name__ == '__main__':
    unittest.main()
