#!/usr/bin/env python3
"""
Test Suite for Loss Functions
Tests focal loss, weighted BCE, and metrics in loss_functions.py
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "2 - Model Training"))

from loss_functions import (
    focal_loss,
    weighted_binary_crossentropy,
    get_loss_function,
    get_metrics
)


class TestFocalLoss(unittest.TestCase):
    """Test focal loss implementation"""
    
    def setUp(self):
        """Create test data"""
        self.y_true = tf.constant([[0], [1], [0], [1]], dtype=tf.float32)
        self.y_pred = tf.constant([[0.1], [0.9], [0.2], [0.8]], dtype=tf.float32)
    
    def test_focal_loss_output_shape(self):
        """Test that focal loss returns scalar"""
        loss_fn = focal_loss(alpha=0.75, gamma=2.0)
        loss = loss_fn(self.y_true, self.y_pred)
        self.assertEqual(loss.shape, ())  # Scalar
    
    def test_focal_loss_positive(self):
        """Test that loss is always positive"""
        loss_fn = focal_loss(alpha=0.75, gamma=2.0)
        loss = loss_fn(self.y_true, self.y_pred)
        self.assertGreaterEqual(loss.numpy(), 0.0)
    
    def test_focal_loss_perfect_predictions(self):
        """Test that perfect predictions give low loss"""
        y_true = tf.constant([[0], [1], [0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.01], [0.99], [0.01], [0.99]], dtype=tf.float32)
        
        loss_fn = focal_loss(alpha=0.75, gamma=2.0)
        loss = loss_fn(y_true, y_pred)
        
        # Perfect predictions should have very low loss
        self.assertLess(loss.numpy(), 0.1)
    
    def test_focal_loss_wrong_predictions(self):
        """Test that wrong predictions give high loss"""
        y_true = tf.constant([[0], [1], [0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.99], [0.01], [0.99], [0.01]], dtype=tf.float32)
        
        loss_fn = focal_loss(alpha=0.75, gamma=2.0)
        loss = loss_fn(y_true, y_pred)
        
        # Wrong predictions should have high loss
        self.assertGreater(loss.numpy(), 1.0)
    
    def test_focal_loss_gamma_effect(self):
        """Test that gamma parameter affects loss for easy examples"""
        y_true = tf.constant([[0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.2], [0.8]], dtype=tf.float32)  # Easy examples
        
        # Low gamma (more like standard BCE)
        loss_low_gamma = focal_loss(alpha=0.75, gamma=0.5)(y_true, y_pred)
        # High gamma (strongly down-weights easy examples)
        loss_high_gamma = focal_loss(alpha=0.75, gamma=3.0)(y_true, y_pred)
        
        # High gamma should give lower loss for easy correct predictions
        self.assertLess(loss_high_gamma.numpy(), loss_low_gamma.numpy())
    
    def test_focal_loss_alpha_effect(self):
        """Test that alpha parameter balances classes"""
        # Imbalanced batch (more negatives)
        y_true = tf.constant([[0], [0], [0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.2], [0.3], [0.1], [0.7]], dtype=tf.float32)
        
        # Low alpha (less weight on positives)
        loss_low_alpha = focal_loss(alpha=0.25, gamma=2.0)(y_true, y_pred)
        # High alpha (more weight on positives)
        loss_high_alpha = focal_loss(alpha=0.90, gamma=2.0)(y_true, y_pred)
        
        # Different alphas should give different losses
        self.assertNotAlmostEqual(loss_low_alpha.numpy(), loss_high_alpha.numpy(), places=3)


class TestWeightedBCE(unittest.TestCase):
    """Test weighted binary crossentropy"""
    
    def setUp(self):
        """Create test data"""
        self.y_true = tf.constant([[0], [1], [0], [1]], dtype=tf.float32)
        self.y_pred = tf.constant([[0.1], [0.9], [0.2], [0.8]], dtype=tf.float32)
    
    def test_weighted_bce_output_shape(self):
        """Test that weighted BCE returns scalar"""
        loss_fn = weighted_binary_crossentropy(class_weight_drone=3.0)
        loss = loss_fn(self.y_true, self.y_pred)
        self.assertEqual(loss.shape, ())
    
    def test_weighted_bce_positive(self):
        """Test that loss is always positive"""
        loss_fn = weighted_binary_crossentropy(class_weight_drone=3.0)
        loss = loss_fn(self.y_true, self.y_pred)
        self.assertGreaterEqual(loss.numpy(), 0.0)
    
    def test_weighted_bce_class_weight_effect(self):
        """Test that class weight affects loss"""
        # Batch with only positive examples
        y_true = tf.constant([[1], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.7], [0.8]], dtype=tf.float32)
        
        # Low weight
        loss_low_weight = weighted_binary_crossentropy(class_weight_drone=1.0)(y_true, y_pred)
        # High weight
        loss_high_weight = weighted_binary_crossentropy(class_weight_drone=5.0)(y_true, y_pred)
        
        # Higher weight should give higher loss
        self.assertGreater(loss_high_weight.numpy(), loss_low_weight.numpy())
    
    def test_weighted_bce_imbalanced_batch(self):
        """Test weighted BCE on imbalanced batch"""
        # Heavily imbalanced (3 negatives, 1 positive)
        y_true = tf.constant([[0], [0], [0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1], [0.2], [0.3], [0.8]], dtype=tf.float32)
        
        loss_fn = weighted_binary_crossentropy(class_weight_drone=3.0)
        loss = loss_fn(y_true, y_pred)
        
        # Should compute without error
        self.assertIsNotNone(loss.numpy())
        self.assertGreater(loss.numpy(), 0.0)


class TestLossFunctionFactory(unittest.TestCase):
    """Test get_loss_function factory"""
    
    def test_get_bce_loss(self):
        """Test getting standard BCE loss"""
        loss_fn = get_loss_function('bce')
        self.assertIsNotNone(loss_fn)
        
        # Test it works
        y_true = tf.constant([[0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1], [0.9]], dtype=tf.float32)
        loss = loss_fn(y_true, y_pred)
        self.assertGreaterEqual(loss.numpy(), 0.0)
    
    def test_get_weighted_bce_loss(self):
        """Test getting weighted BCE loss"""
        loss_fn = get_loss_function('weighted_bce', class_weight=4.0)
        self.assertIsNotNone(loss_fn)
        
        # Test it works
        y_true = tf.constant([[0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1], [0.9]], dtype=tf.float32)
        loss = loss_fn(y_true, y_pred)
        self.assertGreaterEqual(loss.numpy(), 0.0)
    
    def test_get_focal_loss(self):
        """Test getting focal loss"""
        loss_fn = get_loss_function('focal', alpha=0.8, gamma=2.5)
        self.assertIsNotNone(loss_fn)
        
        # Test it works
        y_true = tf.constant([[0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1], [0.9]], dtype=tf.float32)
        loss = loss_fn(y_true, y_pred)
        self.assertGreaterEqual(loss.numpy(), 0.0)
    
    def test_get_invalid_loss(self):
        """Test that invalid loss type raises error"""
        with self.assertRaises(ValueError):
            get_loss_function('invalid_loss_type')
    
    def test_focal_loss_default_params(self):
        """Test focal loss with default parameters"""
        loss_fn = get_loss_function('focal')  # No alpha/gamma specified
        self.assertIsNotNone(loss_fn)
        
        y_true = tf.constant([[0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1], [0.9]], dtype=tf.float32)
        loss = loss_fn(y_true, y_pred)
        self.assertGreaterEqual(loss.numpy(), 0.0)


class TestMetrics(unittest.TestCase):
    """Test get_metrics function"""
    
    def test_get_metrics_returns_list(self):
        """Test that get_metrics returns a list"""
        metrics = get_metrics()
        self.assertIsInstance(metrics, list)
        self.assertGreater(len(metrics), 0)
    
    def test_metrics_types(self):
        """Test that all metrics are valid TensorFlow metrics"""
        metrics = get_metrics()
        
        # All should be callable or Metric instances
        for metric in metrics:
            self.assertTrue(
                callable(metric) or isinstance(metric, tf.keras.metrics.Metric),
                f"Invalid metric type: {type(metric)}"
            )
    
    def test_metrics_work_with_model(self):
        """Test that metrics can be used in model compilation"""
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with metrics (should not raise error)
        try:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=get_metrics()
            )
        except Exception as e:
            self.fail(f"Metrics should be compatible with model: {e}")


class TestLossComparison(unittest.TestCase):
    """Compare different loss functions"""
    
    def setUp(self):
        """Create test scenarios"""
        # Easy examples (confident correct predictions)
        self.y_true_easy = tf.constant([[0], [1], [0], [1]], dtype=tf.float32)
        self.y_pred_easy = tf.constant([[0.05], [0.95], [0.1], [0.9]], dtype=tf.float32)
        
        # Hard examples (uncertain predictions)
        self.y_true_hard = tf.constant([[0], [1], [0], [1]], dtype=tf.float32)
        self.y_pred_hard = tf.constant([[0.45], [0.55], [0.4], [0.6]], dtype=tf.float32)
    
    def test_focal_vs_bce_easy_examples(self):
        """Test that focal loss down-weights easy examples"""
        # Standard BCE
        bce_fn = tf.keras.losses.BinaryCrossentropy()
        bce_loss = bce_fn(self.y_true_easy, self.y_pred_easy)
        
        # Focal loss
        focal_fn = focal_loss(alpha=0.75, gamma=2.0)
        focal_loss_val = focal_fn(self.y_true_easy, self.y_pred_easy)
        
        # Focal should give lower loss for easy examples
        self.assertLess(focal_loss_val.numpy(), bce_loss.numpy())
    
    def test_focal_vs_bce_hard_examples(self):
        """Test focal loss behavior on hard examples"""
        # Standard BCE
        bce_fn = tf.keras.losses.BinaryCrossentropy()
        bce_loss = bce_fn(self.y_true_hard, self.y_pred_hard)
        
        # Focal loss
        focal_fn = focal_loss(alpha=0.75, gamma=2.0)
        focal_loss_val = focal_fn(self.y_true_hard, self.y_pred_hard)
        
        # Both should give relatively high loss for hard examples
        self.assertGreater(focal_loss_val.numpy(), 0.3)
        self.assertGreater(bce_loss.numpy(), 0.3)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of loss functions"""
    
    def test_focal_loss_extreme_predictions(self):
        """Test focal loss with extreme predictions"""
        y_true = tf.constant([[0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.0001], [0.9999]], dtype=tf.float32)
        
        loss_fn = focal_loss(alpha=0.75, gamma=2.0)
        loss = loss_fn(y_true, y_pred)
        
        # Should not be NaN or inf
        self.assertFalse(tf.math.is_nan(loss))
        self.assertFalse(tf.math.is_inf(loss))
    
    def test_weighted_bce_extreme_weights(self):
        """Test weighted BCE with extreme class weights"""
        y_true = tf.constant([[0], [1]], dtype=tf.float32)
        y_pred = tf.constant([[0.1], [0.9]], dtype=tf.float32)
        
        # Very high weight
        loss_fn = weighted_binary_crossentropy(class_weight_drone=100.0)
        loss = loss_fn(y_true, y_pred)
        
        # Should not be NaN or inf
        self.assertFalse(tf.math.is_nan(loss))
        self.assertFalse(tf.math.is_inf(loss))


def run_tests():
    """Run all tests with verbose output"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
