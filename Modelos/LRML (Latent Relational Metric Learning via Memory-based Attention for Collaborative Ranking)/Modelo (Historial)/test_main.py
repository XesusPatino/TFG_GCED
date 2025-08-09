import unittest
import numpy as np
import tensorflow as tf
from main import calculate_rmse, calculate_rmse_combined

class TestCalculateRMSE(unittest.TestCase):
    """Test cases for the calculate_rmse function in main.py."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock model and session for testing
        self.model = MockModel()
        self.sess = MockSession()
    
    def test_basic_rmse_calculation(self):
        """Test basic RMSE calculation with known values."""
        # Setup test data with known RMSE result
        test_data = [(0, 0, 0.5), (1, 1, 0.8), (2, 2, 0.3)]
        # Configure mock to return specific predictions
        self.sess.configure_run_output(np.array([0.6, 0.7, 0.4]))
        
        # Calculate RMSE
        result = calculate_rmse(self.model, self.sess, test_data)
        
        # Expected RMSE: sqrt(mean([(0.6-0.5)^2, (0.7-0.8)^2, (0.4-0.3)^2]))
        # = sqrt(mean([0.01, 0.01, 0.01])) = sqrt(0.01) = 0.1
        expected_rmse = 0.1
        self.assertAlmostEqual(result, expected_rmse, places=6)
        
        # Verify correct values were passed to the model
        self.assertEqual(self.model.last_feed_dict[self.model.user_input], [0, 1, 2])
        self.assertEqual(self.model.last_feed_dict[self.model.item_input], [0, 1, 2])
        self.assertEqual(self.model.last_feed_dict[self.model.dropout], 1.0)

    def test_empty_data(self):
        """Test behavior with empty data list."""
        # Empty test data should raise an exception or return a default value
        test_data = []
        
        with self.assertRaises(IndexError):
            calculate_rmse(self.model, self.sess, test_data)

    def test_single_item(self):
        """Test RMSE calculation with a single data point."""
        test_data = [(0, 0, 0.5)]
        self.sess.configure_run_output(np.array([0.7]))
        
        result = calculate_rmse(self.model, self.sess, test_data)
        expected_rmse = 0.2  # sqrt((0.7-0.5)^2) = 0.2
        
        self.assertAlmostEqual(result, expected_rmse, places=6)

    def test_different_prediction_shapes(self):
        """Test handling of different shapes in prediction outputs."""
        test_data = [(0, 0, 0.5), (1, 1, 0.8), (2, 2, 0.3)]
        
        # Test with 2D array output (common in many models)
        self.sess.configure_run_output(np.array([[0.6], [0.7], [0.4]]))
        result = calculate_rmse(self.model, self.sess, test_data)
        expected_rmse = 0.1
        self.assertAlmostEqual(result, expected_rmse, places=6)
        
        # Test with multi-dimensional output
        self.sess.configure_run_output(np.array([[[0.6]], [[0.7]], [[0.4]]]))
        result = calculate_rmse(self.model, self.sess, test_data)
        expected_rmse = 0.1
        self.assertAlmostEqual(result, expected_rmse, places=6)

    def test_different_rating_ranges(self):
        """Test with different rating ranges."""
        # Test with normalized ratings (0-1)
        test_data_normalized = [(0, 0, 0.2), (1, 1, 0.6), (2, 2, 0.8)]
        self.sess.configure_run_output(np.array([0.3, 0.5, 0.9]))
        
        result = calculate_rmse(self.model, self.sess, test_data_normalized)
        # Expected: sqrt(mean([(0.3-0.2)^2, (0.5-0.6)^2, (0.9-0.8)^2]))
        # = sqrt(mean([0.01, 0.01, 0.01])) = sqrt(0.01) = 0.1
        expected_rmse = 0.1
        self.assertAlmostEqual(result, expected_rmse, places=6)
        
        # Test with raw ratings (1-5)
        test_data_raw = [(0, 0, 2), (1, 1, 3), (2, 2, 4)]
        self.sess.configure_run_output(np.array([2.5, 3.5, 3.5]))
        
        result = calculate_rmse(self.model, self.sess, test_data_raw)
        # Expected: sqrt(mean([(2.5-2)^2, (3.5-3)^2, (3.5-4)^2]))
        # = sqrt(mean([0.25, 0.25, 0.25])) = sqrt(0.25) = 0.5
        expected_rmse = 0.5
        self.assertAlmostEqual(result, expected_rmse, places=6)
        class TestCalculateRMSECombined(unittest.TestCase):
            """Test cases for the calculate_rmse_combined function in main.py."""
            
            def setUp(self):
                """Set up test environment before each test."""
                # Create mock objects
                self.model = MockModel()
                self.sess = MockSession()
                self.history_recommender = MockHistoryRecommender()
                
                # Test data mappings
                self.user_id_to_idx = {1: 0, 2: 1, 3: 2}
                self.item_id_to_idx = {101: 0, 102: 1, 103: 2}
                self.idx_to_user_id = {0: 1, 1: 2, 2: 3}
                self.idx_to_item_id = {0: 101, 1: 102, 2: 103}
                
                # Rating scale
                self.min_rating = 1.0
                self.max_rating = 5.0
            
            def test_basic_combined_rmse(self):
                """Test basic combined RMSE calculation with both model and history predictions."""
                # Setup test data with normalized ratings (0-1 scale)
                test_data = [(0, 0, 0.5), (1, 1, 0.8), (2, 2, 0.3)]
                
                # Configure mock to return specific model predictions (normalized 0-1)
                self.sess.configure_run_output(np.array([0.6, 0.7, 0.4]))
                
                # Configure history recommender to return specific predictions (raw scale 1-5)
                history_predictions = {
                    (1, 101): (4.0, 0.8),  # (prediction, confidence)
                    (2, 102): (3.5, 0.5),
                    (3, 103): (None, 0.0)  # No prediction for this pair
                }
                self.history_recommender.configure_predictions(history_predictions)
                
                # Use 0.3 as history weight
                history_weight = 0.3
                
                # Calculate combined RMSE
                result = calculate_rmse_combined(
                    self.model, self.sess, test_data, 
                    self.history_recommender, 
                    self.user_id_to_idx, self.item_id_to_idx,
                    self.idx_to_user_id, self.idx_to_item_id,
                    self.min_rating, self.max_rating, history_weight
                )
                
                # Expected calculations:
                # Item 1: model_pred = 0.6 -> 3.4, history_pred = 4.0, confidence = 0.8
                #       effective_weight = 0.3 * 0.8 = 0.24
                #       combined = (1-0.24)*3.4 + 0.24*4.0 = 2.584 + 0.96 = 3.544
                # Item 2: model_pred = 0.7 -> 3.8, history_pred = 3.5, confidence = 0.5
                #       effective_weight = 0.3 * 0.5 = 0.15
                #       combined = (1-0.15)*3.8 + 0.15*3.5 = 3.23 + 0.525 = 3.755
                # Item 3: model_pred = 0.4 -> 2.6, no history_pred
                #       combined = 2.6
                # True ratings: 0.5 -> 3.0, 0.8 -> 4.2, 0.3 -> 2.2
                # RMSE = sqrt(mean([(3.544-3.0)^2, (3.755-4.2)^2, (2.6-2.2)^2]))
                # = sqrt(mean([0.296336, 0.198025, 0.16])) = sqrt(0.218120) ≈ 0.467
                expected_rmse = 0.467
                
                self.assertAlmostEqual(result, expected_rmse, places=3)
                
                # Verify correct values were passed to the model
                self.assertEqual(self.model.last_feed_dict[self.model.user_input], [0, 1, 2])
                self.assertEqual(self.model.last_feed_dict[self.model.item_input], [0, 1, 2])
                self.assertEqual(self.model.last_feed_dict[self.model.dropout], 1.0)

            def test_zero_history_weight(self):
                """Test with history weight set to 0 - should equal basic RMSE."""
                test_data = [(0, 0, 0.5), (1, 1, 0.8)]
                
                # Configure mocks
                self.sess.configure_run_output(np.array([0.6, 0.7]))
                history_predictions = {
                    (1, 101): (4.0, 0.8),
                    (2, 102): (3.5, 0.5)
                }
                self.history_recommender.configure_predictions(history_predictions)
                
                # Calculate with zero history weight
                result = calculate_rmse_combined(
                    self.model, self.sess, test_data, 
                    self.history_recommender, 
                    self.user_id_to_idx, self.item_id_to_idx,
                    self.idx_to_user_id, self.idx_to_item_id,
                    self.min_rating, self.max_rating, 0.0
                )
                
                # Expected: Same as model predictions
                # model_preds = [0.6, 0.7] -> denormalized [3.4, 3.8]
                # true_ratings = [0.5, 0.8] -> denormalized [3.0, 4.2]
                # RMSE = sqrt(mean([(3.4-3.0)^2, (3.8-4.2)^2])) = sqrt(mean([0.16, 0.16])) = 0.4
                expected_rmse = 0.4
                self.assertAlmostEqual(result, expected_rmse, places=3)

            def test_no_history_predictions(self):
                """Test when history recommender returns no valid predictions."""
                test_data = [(0, 0, 0.5), (1, 1, 0.8)]
                
                # Configure mocks
                self.sess.configure_run_output(np.array([0.6, 0.7]))
                # No valid history predictions
                history_predictions = {
                    (1, 101): (None, 0.0),
                    (2, 102): (None, 0.0)
                }
                self.history_recommender.configure_predictions(history_predictions)
                
                # Calculate with history weight = 0.3
                result = calculate_rmse_combined(
                    self.model, self.sess, test_data, 
                    self.history_recommender, 
                    self.user_id_to_idx, self.item_id_to_idx,
                    self.idx_to_user_id, self.idx_to_item_id,
                    self.min_rating, self.max_rating, 0.3
                )
                
                # Expected: Same as model predictions since no history is available
                # model_preds = [0.6, 0.7] -> denormalized [3.4, 3.8]
                # true_ratings = [0.5, 0.8] -> denormalized [3.0, 4.2]
                # RMSE = sqrt(mean([(3.4-3.0)^2, (3.8-4.2)^2])) = sqrt(mean([0.16, 0.16])) = 0.4
                expected_rmse = 0.4
                self.assertAlmostEqual(result, expected_rmse, places=3)

            def test_full_history_weight(self):
                """Test with history weight set to 1.0 and full confidence."""
                test_data = [(0, 0, 0.5), (1, 1, 0.8)]
                
                # Configure mocks
                self.sess.configure_run_output(np.array([0.6, 0.7]))
                history_predictions = {
                    (1, 101): (4.0, 1.0),  # Full confidence
                    (2, 102): (3.5, 1.0)   # Full confidence
                }
                self.history_recommender.configure_predictions(history_predictions)
                
                # Calculate with full history weight
                result = calculate_rmse_combined(
                    self.model, self.sess, test_data, 
                    self.history_recommender, 
                    self.user_id_to_idx, self.item_id_to_idx,
                    self.idx_to_user_id, self.idx_to_item_id,
                    self.min_rating, self.max_rating, 1.0
                )
                
                # Expected: Using only history predictions
                # history_preds = [4.0, 3.5]
                # true_ratings = [0.5, 0.8] -> denormalized [3.0, 4.2]
                # RMSE = sqrt(mean([(4.0-3.0)^2, (3.5-4.2)^2])) = sqrt(mean([1.0, 0.49])) = sqrt(0.745) ≈ 0.863
                expected_rmse = 0.863
                self.assertAlmostEqual(result, expected_rmse, places=3)

            def test_empty_data(self):
                """Test behavior with empty data list."""
                # Empty test data
                test_data = []
                
                # Since we're extracting user_idxs, item_idxs, etc. from test_data first,
                # this should raise an IndexError or handle empty data appropriately
                with self.assertRaises(IndexError):
                    calculate_rmse_combined(
                        self.model, self.sess, test_data, 
                        self.history_recommender, 
                        self.user_id_to_idx, self.item_id_to_idx,
                        self.idx_to_user_id, self.idx_to_item_id,
                        self.min_rating, self.max_rating, 0.3
                    )

        # Additional mock class for history recommender
        class MockHistoryRecommender:
            """Mock history recommender class for testing."""
            def __init__(self):
                self.predictions = {}
            
            def configure_predictions(self, predictions):
                """Configure what predict_rating_from_history should return for different user-item pairs."""
                self.predictions = predictions
                
            def predict_rating_from_history(self, user_id, item_id):
                """Mock implementation returning preconfigured predictions."""
                if (user_id, item_id) in self.predictions:
                    return self.predictions[(user_id, item_id)]
                return None, 0.0
# Mock classes for testing
class MockModel:
    """Mock model class for testing."""
    def __init__(self):
        self.user_input = "user_input"
        self.item_input = "item_input" 
        self.dropout = "dropout"
        self.predict_op = "predict_op"
        self.last_feed_dict = {}

class MockSession:
    """Mock TensorFlow session for testing."""
    def __init__(self):
        self.outputs = {}
        
    def configure_run_output(self, output):
        """Configure what the run method should return."""
        self.outputs["predict_op"] = output
        
    def run(self, op, feed_dict=None):
        """Mock implementation of session.run()."""
        if feed_dict:
            # Save the feed_dict for inspection in tests
            self.last_feed_dict = feed_dict
        
        # Return configured output for the given operation
        if op in self.outputs:
            return self.outputs[op]
        return None

if __name__ == '__main__':
    unittest.main()