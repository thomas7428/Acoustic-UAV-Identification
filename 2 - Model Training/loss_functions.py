"""
Utility functions for improved loss functions and training techniques.

This module provides:
- Focal Loss: Focus on hard-to-classify examples
- Class Weighted BCE: Balance class importance
- Distance Weighted Loss: Prioritize hard distances (500m, 350m)
- Additional metrics for better evaluation
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def recall_focused_loss(fn_penalty=50.0):
    """
    Custom loss that heavily penalizes False Negatives (missed drones).
    
    Loss = BCE + fn_penalty * FN_weight
    
    This forces the model to prioritize recall (detecting ALL drones)
    even at the cost of false positives.
    
    Args:
        fn_penalty: Multiplier for false negative penalty (default: 50.0)
                   Higher = more aggressive drone detection
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    def loss_fn(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
            y_true = tf.cast(y_true, tf.float32)
        
        # Standard BCE
        bce = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # False Negative penalty: when y_true=1 (drone) but y_pred→0 (predicted no-drone)
        # Extract drone class (index 1)
        y_true_drone = y_true[:, 1]  # True label for drone
        y_pred_drone = y_pred[:, 1]  # Predicted probability for drone
        
        # FN occurs when: true=drone (1) but predicted=no-drone (low probability)
        # Penalty = y_true_drone * (1 - y_pred_drone)
        # If drone but predicted 0 → penalty = 1.0
        # If drone but predicted 1 → penalty = 0.0
        fn_weight = y_true_drone * (1.0 - y_pred_drone)
        
        # Combined loss
        total_loss = bce + fn_penalty * fn_weight
        
        return total_loss
    
    return loss_fn


def focal_loss(alpha=0.75, gamma=2.0):
    """
    Focal Loss for addressing class imbalance and focusing on hard examples.
    
    Focal Loss = -α * (1 - p_t)^γ * log(p_t)
    
    where p_t is the model's estimated probability for the true class.
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               Higher alpha = more weight on positive class (drones)
        gamma: Focusing parameter >= 0. Higher gamma = more focus on hard examples
               gamma = 0 reduces to standard Cross-Entropy
               gamma = 2 is typical
    
    Returns:
        Loss function compatible with Keras model.compile()
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
            y_true = tf.cast(y_true, tf.float32)
        
        # Calculate categorical cross entropy
        bce = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Calculate p_t (probability of true class)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # Calculate focal loss
        alpha_factor = alpha
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        focal = alpha_factor * modulating_factor * bce
        
        return tf.reduce_mean(focal)
    
    return loss_fn


def weighted_binary_crossentropy(class_weight_drone=3.0):
    """
    Weighted Binary Cross-Entropy Loss.
    
    Applies different weights to positive (drone) and negative (no-drone) classes.
    
    Args:
        class_weight_drone: Weight for drone class (positive class)
                           class_weight_no_drone is implicitly 1.0
                           Higher values = more penalty for missing drones
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    def loss_fn(y_true, y_pred):
        # Calculate binary cross entropy
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Apply class weights
        # y_true = 1 (drone) gets weight class_weight_drone
        # y_true = 0 (no-drone) gets weight 1.0
        weights = y_true * class_weight_drone + (1 - y_true) * 1.0
        
        weighted_bce = weights * bce
        
        return tf.reduce_mean(weighted_bce)
    
    return loss_fn


def get_loss_function(loss_type='bce', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: One of 'bce', 'weighted_bce', 'focal', 'recall_focused'
        **kwargs: Additional arguments for the loss function
                  - class_weight_drone: for 'weighted_bce' (default: 3.0)
                  - alpha: for 'focal' (default: 0.75)
                  - gamma: for 'focal' (default: 2.0)
                  - fn_penalty: for 'recall_focused' (default: 50.0)
    
    Returns:
        Loss function
    
    Examples:
        >>> loss = get_loss_function('focal', alpha=0.75, gamma=2.0)
        >>> loss = get_loss_function('weighted_bce', class_weight_drone=3.0)
        >>> loss = get_loss_function('bce')  # Standard BCE
    """
    if loss_type == 'bce':
        return 'binary_crossentropy'
    
    elif loss_type == 'weighted_bce':
        class_weight = kwargs.get('class_weight_drone', 3.0)
        return weighted_binary_crossentropy(class_weight_drone=class_weight)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.75)
        gamma = kwargs.get('gamma', 2.0)
        return focal_loss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'recall_focused':
        fn_penalty = kwargs.get('fn_penalty', 50.0)
        return recall_focused_loss(fn_penalty=fn_penalty)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: 'bce', 'weighted_bce', 'focal', 'recall_focused'")


def get_metrics():
    """
    Get comprehensive metrics for binary classification.
    
    Returns:
        List of metrics to pass to model.compile()
    """
    return [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn')
    ]


# Convenience functions for common configurations
def get_focal_loss_for_drones(alpha=0.75, gamma=2.0):
    """
    Get focal loss optimized for drone detection.
    
    Args:
        alpha: 0.75 means 75% weight on drones, 25% on no-drones
        gamma: 2.0 focuses on hard examples
    
    Returns:
        Focal loss function
    """
    return focal_loss(alpha=alpha, gamma=gamma)


def get_weighted_loss_for_drones(weight=3.0):
    """
    Get weighted BCE optimized for drone detection.
    
    Args:
        weight: How much more important drones are than no-drones
                3.0 = drones are 3x more important
    
    Returns:
        Weighted BCE loss function
    """
    return weighted_binary_crossentropy(class_weight_drone=weight)


if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    print("Testing loss functions...")
    
    # Dummy data
    y_true = np.array([[1], [0], [1], [0]])  # 2 drones, 2 no-drones
    y_pred = np.array([[0.9], [0.1], [0.6], [0.4]])  # Predictions
    
    # Test BCE
    bce_loss = keras.losses.BinaryCrossentropy()
    print(f"BCE Loss: {bce_loss(y_true, y_pred):.4f}")
    
    # Test Weighted BCE
    weighted_loss = get_weighted_loss_for_drones(weight=3.0)
    print(f"Weighted BCE Loss (weight=3.0): {weighted_loss(y_true, y_pred):.4f}")
    
    # Test Focal Loss
    focal = get_focal_loss_for_drones(alpha=0.75, gamma=2.0)
    print(f"Focal Loss (α=0.75, γ=2.0): {focal(y_true, y_pred):.4f}")


def distance_weighted_loss(distance_weights=None, base_loss='bce', alpha=0.75, gamma=0.0):
    """
    Distance-Weighted Loss Function.
    
    Applies different weights to samples based on their distance to prioritize
    hard-to-detect extreme distances (500m, 350m).
    
    Args:
        distance_weights: Dict mapping distance (in meters) to weight multiplier
                         Example: {500: 10.0, 350: 8.0, 200: 5.0, 100: 2.0, 50: 1.0}
                         If None, uses default weights
        base_loss: Base loss function ('bce', 'focal')
        alpha: Focal loss alpha (if base_loss='focal')
        gamma: Focal loss gamma (if base_loss='focal')
    
    Returns:
        Loss function compatible with Keras model.compile()
        
    Usage:
        This requires passing distance labels alongside training.
        Use fit() with custom training loop or data generator that provides:
        - y_true: [batch_size, 2] where y_true[:, 0] = class label (0/1)
                                        y_true[:, 1] = distance label (500/350/200/100/50)
    """
    if distance_weights is None:
        # Default weights: prioritize extreme distances
        distance_weights = {
            500: 10.0,  # 10x weight for 500m
            350: 8.0,   # 8x weight for 350m
            200: 5.0,   # 5x weight for 200m
            100: 2.0,   # 2x weight for 100m
            50: 1.0,    # 1x weight for 50m (baseline)
            0: 1.0      # 1x weight for ambient/background (no distance)
        }
    
    def loss_fn(y_true, y_pred):
        """
        y_true: [batch_size, 2] - [:, 0] = class label, [:, 1] = distance (m)
        y_pred: [batch_size, 2] - softmax output
        """
        # Extract class labels and distances
        class_labels = y_true[:, 0]
        distance_labels = y_true[:, 1]
        
        # Calculate base loss
        if base_loss == 'focal':
            # Focal loss calculation
            epsilon = keras.backend.epsilon()
            y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Convert to one-hot
            class_labels_int = tf.cast(class_labels, tf.int32)
            class_labels_onehot = tf.one_hot(class_labels_int, depth=2)
            class_labels_onehot = tf.cast(class_labels_onehot, tf.float32)
            
            # Cross entropy
            bce = keras.losses.categorical_crossentropy(class_labels_onehot, y_pred_clipped)
            
            # Focal modulation
            p_t = tf.reduce_sum(class_labels_onehot * y_pred_clipped, axis=-1)
            focal_weight = tf.pow(1.0 - p_t, gamma)
            base_loss_value = alpha * focal_weight * bce
        else:
            # Standard BCE
            epsilon = keras.backend.epsilon()
            y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Get predicted probability for positive class
            y_pred_pos = y_pred_clipped[:, 1]
            
            # Binary cross entropy
            base_loss_value = -(class_labels * tf.math.log(y_pred_pos) + 
                               (1 - class_labels) * tf.math.log(1 - y_pred_pos))
        
        # Map distances to weights
        # Create a lookup table
        distance_values = tf.constant(list(distance_weights.keys()), dtype=tf.float32)
        weight_values = tf.constant(list(distance_weights.values()), dtype=tf.float32)
        
        # For each sample, find its weight based on distance
        # Use broadcasting to find closest distance match
        distance_labels_expanded = tf.expand_dims(distance_labels, 1)  # [batch, 1]
        distance_values_expanded = tf.expand_dims(distance_values, 0)  # [1, n_distances]
        
        # Find closest distance (for fuzzy matching)
        distances_diff = tf.abs(distance_labels_expanded - distance_values_expanded)
        closest_idx = tf.argmin(distances_diff, axis=1)
        
        # Get weights for each sample
        sample_weights = tf.gather(weight_values, closest_idx)
        
        # Apply distance weights to loss
        weighted_loss = base_loss_value * sample_weights
        
        return tf.reduce_mean(weighted_loss)
    
    return loss_fn


def get_loss_function_with_distance(loss_type='distance_weighted', **kwargs):
    """
    Get a distance-aware loss function.
    
    Args:
        loss_type: Type of loss ('distance_weighted', 'focal', 'bce')
        **kwargs: Additional arguments passed to loss function
        
    Returns:
        Loss function
    """
    if loss_type == 'distance_weighted':
        return distance_weighted_loss(**kwargs)
    elif loss_type == 'focal':
        return focal_loss(**kwargs)
    elif loss_type == 'weighted_bce':
        return weighted_binary_crossentropy(**kwargs)
    else:
        return 'sparse_categorical_crossentropy'
    
    print("\nAll tests passed!")
