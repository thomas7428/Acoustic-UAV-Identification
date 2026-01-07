import numpy as np
import tensorflow as tf
from tensorflow import keras

from loss_functions import set_dynamic_pos_weight, DYNAMIC_POS_WEIGHT


class DynamicClassWeightCallback(keras.callbacks.Callback):
    """Callback to update a global positive-class weight each epoch.

    The callback evaluates the model on a provided validation set at the end
    of each epoch, computes FP and FN counts, and updates
    `loss_functions.DYNAMIC_POS_WEIGHT` using an EMA-smoothed ratio:

        candidate = base_weight * (fn + 1) / (fp + 1)
        new = beta * current + (1-beta) * candidate
        new = clip(new, min_w, max_w)

    This steers the loss toward emphasizing recall when FN >> FP, or
    toward reducing false positives when FP >> FN.
    """

    def __init__(self, val_data, base_weight=1.0, beta=0.8, min_w=0.5, max_w=8.0,
                 batch_size=64, warmup_epochs=2, max_delta=2.0, min_pred_samples=10):
        super().__init__()
        self.x_val, self.y_val = val_data
        self.base_weight = float(base_weight)
        self.beta = float(beta)
        self.min_w = float(min_w)
        self.max_w = float(max_w)
        self.batch_size = int(batch_size)
        # defensive tuning
        self.warmup_epochs = int(warmup_epochs)
        # max multiplicative change allowed per epoch (e.g. 2.0 -> at most double)
        self.max_delta = float(max_delta)
        # minimum number of validation samples required to compute updates
        self.min_pred_samples = int(min_pred_samples)
        # initialize with current value
        try:
            self.current = float(DYNAMIC_POS_WEIGHT)
        except Exception:
            self.current = 1.0

    def on_train_begin(self, logs=None):
        try:
            # reset global state to base to avoid cross-run contamination
            set_dynamic_pos_weight(float(self.base_weight))
            self.current = float(DYNAMIC_POS_WEIGHT)
        except Exception:
            self.current = float(self.base_weight)

    def on_epoch_end(self, epoch, logs=None):
        # Predict on validation set
        preds = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)

        # Handle both softmax(2) outputs and scalar probabilities
        if len(preds.shape) > 1 and preds.shape[-1] > 1:
            preds_pos = preds[:, 1]
            pred_labels = (preds_pos >= 0.5).astype(int)
        else:
            preds_pos = preds.flatten()
            pred_labels = (preds_pos >= 0.5).astype(int)

        # Handle y_val that may be one-hot encoded or integer labels
        y_arr = np.array(self.y_val)
        if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
            # one-hot -> argmax
            y_true = np.argmax(y_arr, axis=1).astype(int)
        else:
            y_true = y_arr.reshape(-1).astype(int)

        # Compute confusion counts
        tp = int(np.sum((pred_labels == 1) & (y_true == 1)))
        tn = int(np.sum((pred_labels == 0) & (y_true == 0)))
        fp = int(np.sum((pred_labels == 1) & (y_true == 0)))
        fn = int(np.sum((pred_labels == 0) & (y_true == 1)))

        # defensive checks: ensure enough samples
        n_samples = len(y_true)
        preds_pos_mean = float(np.mean(preds_pos)) if n_samples > 0 else 0.0

        if n_samples < self.min_pred_samples:
            print(f"[DynamicClassWeight] Epoch {epoch+1}: insufficient val samples ({n_samples}), skipping update")
            return

        # detect degenerate predictions: all predicted same class or probabilities collapsed
        degenerate = False
        if (tp + fp) == 0 or (tn + fn) == 0:
            degenerate = True
        if preds_pos_mean <= 1e-6 or preds_pos_mean >= 1.0 - 1e-6:
            degenerate = True

        # candidate weight increases when FN > FP
        candidate = self.base_weight * float((fn + 1) / (fp + 1))

        if degenerate:
            # do not apply large swings when model predicts a single class; nudge towards base
            # use a conservative candidate (average with base)
            candidate = 0.5 * (candidate + self.base_weight)
            print(f"[DynamicClassWeight] Epoch {epoch+1}: degenerate preds detected (mean={preds_pos_mean:.4f}), using conservative candidate={candidate:.4f}")

        # EMA step
        new_val = self.beta * float(self.current) + (1.0 - self.beta) * float(candidate)

        # clamp multiplicative change relative to current to avoid runaway
        if self.current > 0:
            ratio = new_val / float(self.current)
            if ratio > self.max_delta:
                new_val = float(self.current) * self.max_delta
            elif ratio < 1.0 / float(self.max_delta):
                new_val = float(self.current) / self.max_delta

        # hard bounds
        new_val = max(self.min_w, min(self.max_w, new_val))

        # update global variable
        set_dynamic_pos_weight(new_val)
        self.current = new_val

        # Log changes with diagnostics
        print(f"[DynamicClassWeight] Epoch {epoch+1}: tp={tp} fp={fp} tn={tn} fn={fn} mean_pred={preds_pos_mean:.4f} -> new_pos_weight={new_val:.4f}")
