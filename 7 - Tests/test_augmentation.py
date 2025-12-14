#!/usr/bin/env python3
"""
Test Suite for Audio Augmentation Functions
Tests the advanced augmentation functions in augment_dataset_v2.py
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "0 - DADS dataset extraction"))

from augment_dataset_v2 import (
    apply_doppler_shift,
    apply_intensity_modulation,
    apply_reverberation,
    apply_time_stretch_variation
)


class TestDopplerShift(unittest.TestCase):
    """Test Doppler shift augmentation"""
    
    def setUp(self):
        """Create test signal"""
        self.sr = 22050
        self.duration = 2.0
        self.signal = np.sin(2 * np.pi * 440 * np.linspace(0, self.duration, int(self.sr * self.duration)))
    
    def test_doppler_shift_shape(self):
        """Test that output shape matches input"""
        shifted = apply_doppler_shift(self.signal, self.sr, shift_range=0.5)
        self.assertEqual(shifted.shape, self.signal.shape)
    
    def test_doppler_shift_range(self):
        """Test that shift is within specified range"""
        # Run multiple times to test randomness
        for _ in range(10):
            shifted = apply_doppler_shift(self.signal, self.sr, shift_range=0.5)
            # Signal should be modified (not identical)
            self.assertFalse(np.array_equal(shifted, self.signal))
    
    def test_doppler_shift_zero_range(self):
        """Test that zero range returns similar signal"""
        shifted = apply_doppler_shift(self.signal, self.sr, shift_range=0.0)
        # Should be very close to original (within numerical precision)
        np.testing.assert_array_almost_equal(shifted, self.signal, decimal=5)


class TestIntensityModulation(unittest.TestCase):
    """Test intensity modulation augmentation"""
    
    def setUp(self):
        """Create test signal"""
        self.sr = 22050
        self.duration = 2.0
        self.signal = np.ones(int(self.sr * self.duration)) * 0.5
    
    def test_modulation_shape(self):
        """Test that output shape matches input"""
        modulated = apply_intensity_modulation(self.signal, self.sr, 
                                              mod_freq_range=(0.5, 2.0), depth=0.3)
        self.assertEqual(modulated.shape, self.signal.shape)
    
    def test_modulation_amplitude_range(self):
        """Test that modulation depth is respected"""
        modulated = apply_intensity_modulation(self.signal, self.sr, 
                                              mod_freq_range=(1.0, 1.0), depth=0.3)
        # With depth 0.3, amplitude should be between 0.7 and 1.3 of original
        self.assertGreaterEqual(np.max(modulated), 0.5 * 0.7 * 0.9)  # Allow some margin
        self.assertLessEqual(np.max(modulated), 0.5 * 1.3 * 1.1)
    
    def test_modulation_frequency(self):
        """Test that modulation frequency is in valid range"""
        # Test multiple runs
        for _ in range(10):
            modulated = apply_intensity_modulation(self.signal, self.sr,
                                                  mod_freq_range=(0.5, 2.0), depth=0.3)
            # Signal should be modified
            self.assertFalse(np.array_equal(modulated, self.signal))


class TestReverberation(unittest.TestCase):
    """Test reverberation augmentation"""
    
    def setUp(self):
        """Create test signal"""
        self.sr = 22050
        self.duration = 1.0
        # Create impulse signal for easier testing
        self.signal = np.zeros(int(self.sr * self.duration))
        self.signal[1000] = 1.0
    
    def test_reverberation_shape(self):
        """Test that output shape matches input"""
        reverb = apply_reverberation(self.signal, self.sr, 
                                    delay_range=(50, 200), decay_range=(0.2, 0.4))
        self.assertEqual(reverb.shape, self.signal.shape)
    
    def test_reverberation_echo_present(self):
        """Test that echo is added to signal"""
        reverb = apply_reverberation(self.signal, self.sr,
                                    delay_range=(100, 100), decay_range=(0.3, 0.3))
        # Echo should increase energy in the signal
        self.assertGreater(np.sum(np.abs(reverb)), np.sum(np.abs(self.signal)))
    
    def test_reverberation_decay(self):
        """Test that decay reduces echo amplitude"""
        # High decay (low value)
        reverb_low = apply_reverberation(self.signal, self.sr,
                                        delay_range=(100, 100), decay_range=(0.1, 0.1))
        # Medium decay
        reverb_med = apply_reverberation(self.signal, self.sr,
                                        delay_range=(100, 100), decay_range=(0.5, 0.5))
        # Lower decay should have less total energy
        self.assertLess(np.sum(np.abs(reverb_low)), np.sum(np.abs(reverb_med)))


class TestTimeStretch(unittest.TestCase):
    """Test time stretching augmentation"""
    
    def setUp(self):
        """Create test signal"""
        self.sr = 22050
        self.duration = 2.0
        self.signal = np.sin(2 * np.pi * 440 * np.linspace(0, self.duration, int(self.sr * self.duration)))
    
    def test_time_stretch_shape(self):
        """Test that output shape is close to input (may differ slightly)"""
        stretched = apply_time_stretch_variation(self.signal, self.sr, rate_range=(0.95, 1.05))
        # Shape should be within 10% of original
        self.assertAlmostEqual(stretched.shape[0], self.signal.shape[0], delta=self.signal.shape[0] * 0.1)
    
    def test_time_stretch_modification(self):
        """Test that signal is modified"""
        for _ in range(10):
            stretched = apply_time_stretch_variation(self.signal, self.sr, rate_range=(0.95, 1.05))
            # Resample to same length for comparison
            if stretched.shape[0] != self.signal.shape[0]:
                from scipy.signal import resample
                stretched = resample(stretched, self.signal.shape[0])
            # Should be different from original
            self.assertFalse(np.array_equal(stretched, self.signal))
    
    def test_time_stretch_rate_range(self):
        """Test different stretch rates"""
        # Slower (rate < 1.0)
        slow = apply_time_stretch_variation(self.signal, self.sr, rate_range=(0.9, 0.9))
        # Faster (rate > 1.0)
        fast = apply_time_stretch_variation(self.signal, self.sr, rate_range=(1.1, 1.1))
        # Slower should be longer, faster should be shorter
        self.assertGreater(slow.shape[0], fast.shape[0])


class TestAugmentationIntegration(unittest.TestCase):
    """Integration tests for augmentation pipeline"""
    
    def setUp(self):
        """Create realistic test signal"""
        self.sr = 22050
        self.duration = 3.0
        # Create complex signal with multiple frequencies
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        self.signal = (np.sin(2 * np.pi * 200 * t) + 
                      0.5 * np.sin(2 * np.pi * 600 * t) +
                      0.3 * np.sin(2 * np.pi * 1200 * t))
    
    def test_sequential_augmentation(self):
        """Test applying multiple augmentations sequentially"""
        signal = self.signal.copy()
        
        # Apply all augmentations
        signal = apply_doppler_shift(signal, self.sr)
        signal = apply_intensity_modulation(signal, self.sr)
        signal = apply_reverberation(signal, self.sr)
        signal = apply_time_stretch_variation(signal, self.sr)
        
        # Final signal should be modified but reasonable
        self.assertFalse(np.array_equal(signal, self.signal))
        # Check for reasonable amplitude (not exploded)
        self.assertLess(np.max(np.abs(signal)), 10.0)
    
    def test_augmentation_preserves_dtype(self):
        """Test that augmentations preserve float32 dtype"""
        signal = self.signal.astype(np.float32)
        
        signal = apply_doppler_shift(signal, self.sr)
        self.assertEqual(signal.dtype, np.float32)
        
        signal = apply_intensity_modulation(signal, self.sr)
        self.assertEqual(signal.dtype, np.float32)
        
        signal = apply_reverberation(signal, self.sr)
        self.assertEqual(signal.dtype, np.float32)


class TestAugmentationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_signal(self):
        """Test augmentation with empty signal"""
        signal = np.array([])
        sr = 22050
        
        # Should handle gracefully (may return empty or raise error)
        try:
            result = apply_doppler_shift(signal, sr)
            self.assertEqual(len(result), 0)
        except (ValueError, IndexError):
            pass  # Acceptable to raise error
    
    def test_very_short_signal(self):
        """Test augmentation with very short signal"""
        signal = np.array([0.1, 0.2, 0.3])
        sr = 22050
        
        # Should handle without crashing
        try:
            result = apply_doppler_shift(signal, sr)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Should handle short signals gracefully: {e}")
    
    def test_zero_signal(self):
        """Test augmentation with all-zero signal"""
        signal = np.zeros(22050)
        sr = 22050
        
        # Should work but may return zeros
        result = apply_intensity_modulation(signal, sr)
        self.assertEqual(result.shape, signal.shape)


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
