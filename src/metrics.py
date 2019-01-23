from functools import partial

from keras import backend as K
from keras.layers import Layer
from operator import truediv
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

CLASS_NUM = 28
GAP = K.constant(0.25, dtype="float32")
THRESHOLD = 0.05


def normalize(y):
    return K.cast(K.greater(K.clip(y, 0, 1), THRESHOLD), 'int32')
    # return K.cast(K.round(y), 'int32')


class MetricsLayer(Layer):
    def __init__(self, label=0, **kwargs):
        super(MetricsLayer, self).__init__(**kwargs)
        self.stateful = True
        self.epsilon = K.constant(K.epsilon(), dtype="float64")
        self.label = label


class TruePositive(MetricsLayer):
    """Create a metric for model's true positives amount calculation.

    A true positive is an outcome where the model correctly predicts the
    positive class.
    """

    def __init__(self, name="true_positive", **kwargs):
        super(TruePositive, self).__init__(name=name, **kwargs)
        self.tp = K.variable(0, dtype="int32")

    def reset_states(self):
        """Reset the state of the metric."""
        K.set_value(self.tp, 0)

    def __call__(self, y_true, y_pred):
        y_true = normalize(y_true)
        y_pred = normalize(y_pred)
        tp = K.sum(y_true[:, self.label] * y_pred[:, self.label], axis=-1)
        current_tp = self.tp * 1

        tp_update = K.update_add(self.tp, tp)
        self.add_update(tp_update, inputs=[y_true, y_pred])

        return tp + current_tp


class TrueNegative(MetricsLayer):
    """Create a metric for model's true negatives amount calculation.

    A true negative is an outcome where the model correctly predicts the
    negative class.
    """

    def __init__(self, name="true_negative", **kwargs):
        super(TrueNegative, self).__init__(name=name, **kwargs)
        self.tn = K.variable(0, dtype="int32")

    def reset_states(self):
        """Reset the state of the metric."""
        K.set_value(self.tn, 0)

    def __call__(self, y_true, y_pred):
        y_true = normalize(y_true)
        y_pred = normalize(y_pred)

        neg_y_true = 1 - y_true[:, self.label]
        neg_y_pred = 1 - y_pred[:, self.label]

        tn = K.sum(neg_y_true * neg_y_pred, axis=-1)
        current_tn = self.tn * 1

        tn_update = K.update_add(self.tn, tn)
        self.add_update(tn_update, inputs=[y_true, y_pred])

        return tn + current_tn


class FalseNegative(MetricsLayer):
    """Create a metric for model's false negatives amount calculation.

    A false negative is an outcome where the model incorrectly predicts the
    negative class.
    """

    def __init__(self, name="false_negative", **kwargs):
        super(FalseNegative, self).__init__(name=name, **kwargs)
        self.fn = K.variable(0, dtype="int32")

    def reset_states(self):
        """Reset the state of the metric."""
        K.set_value(self.fn, 0)

    def __call__(self, y_true, y_pred):
        y_true = normalize(y_true)
        y_pred = normalize(y_pred)
        neg_y_pred = 1 - y_pred[:, self.label]

        fn = K.sum(y_true[:, self.label] * neg_y_pred, axis=-1)
        current_fn = self.fn * 1

        fn_update = K.update_add(self.fn, fn)
        self.add_update(fn_update, inputs=[y_true, y_pred])

        return fn + current_fn


class FalsePositive(MetricsLayer):
    """Create a metric for model's false positive amount calculation.

    A false positive is an outcome where the model incorrectly predicts the
    positive class.
    """

    def __init__(self, name="false_positive", **kwargs):
        super(FalsePositive, self).__init__(name=name, **kwargs)
        self.fp = K.variable(0, dtype="int32")

    def reset_states(self):
        """Reset the state of the metric."""
        K.set_value(self.fp, 0)

    def __call__(self, y_true, y_pred):
        y_true = normalize(y_true)
        y_pred = normalize(y_pred)
        neg_y_true = 1 - y_true[:, self.label]

        fp = K.sum(neg_y_true * y_pred[:, self.label], axis=-1)
        current_fp = self.fp * 1

        fp_update = K.update_add(self.fp, fp)
        self.add_update(fp_update, inputs=[y_true, y_pred])

        return fp + current_fp


class MacroRecall(MetricsLayer):
    """Create a metric for model's recall calculation.

    Recall measures proportion of actual positives that was identified correctly.
    """

    def __init__(self, name="recall", **kwargs):
        super(MacroRecall, self).__init__(name=name, **kwargs)
        self.tps = [TruePositive(label=i) for i in range(CLASS_NUM)]
        self.fns = [FalseNegative(label=i) for i in range(CLASS_NUM)]

    def reset_states(self):
        """Reset the state of the metrics."""
        for i in range(CLASS_NUM):
            self.tps[i].reset_states()
            self.fns[i].reset_states()

    def __call__(self, y_true, y_pred):
        tps = []
        fns = []
        for i in range(CLASS_NUM):
            tp = self.tps[i](y_true, y_pred)
            tps.append(K.cast(tp, 'float64'))
            fn = self.fns[i](y_true, y_pred)
            fns.append(K.cast(fn, 'float64'))

        tps_updates = [self.tps[i].updates for i in range(CLASS_NUM)]
        fns_updates = [self.fns[i].updates for i in range(CLASS_NUM)]
        self.add_update(tps_updates)
        self.add_update(fns_updates)

        return truediv(K.sum(tps), K.sum(tps) + K.sum(fns) + self.epsilon)


class MacroPrecision(MetricsLayer):
    """Create  a metric for model's precision calculation.

    Precision measures proportion of positives identifications that were
    actually correct.
    """

    def __init__(self, name="precision", **kwargs):
        super(MacroPrecision, self).__init__(name=name, **kwargs)
        self.tps = [TruePositive(label=i) for i in range(CLASS_NUM)]
        self.fps = [FalsePositive(label=i) for i in range(CLASS_NUM)]

    def reset_states(self):
        """Reset the state of the metrics."""
        for i in range(CLASS_NUM):
            self.tps[i].reset_states()
            self.fps[i].reset_states()

    def __call__(self, y_true, y_pred):
        tps = []
        fps = []
        for i in range(CLASS_NUM):
            tp = self.tps[i](y_true, y_pred)
            tps.append(K.cast(tp, 'float64'))
            fp = self.fps[i](y_true, y_pred)
            fps.append(K.cast(fp, 'float64'))

        tps_updates = [self.tps[i].updates for i in range(CLASS_NUM)]
        fps_updates = [self.fps[i].updates for i in range(CLASS_NUM)]
        self.add_update(tps_updates)
        self.add_update(fps_updates)

        return truediv(K.sum(tps), K.sum(tps) + K.sum(fps) + self.epsilon)


class MacroF1Score(MetricsLayer):
    """Create a metric for the model's F1 score calculation.

    The F1 score is the harmonic mean of precision and recall.
    """

    def __init__(self, name="f1_score", **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.precision = MacroPrecision()
        self.recall = MacroRecall()

    def reset_states(self):
        """Reset the state of the metrics."""
        self.precision.reset_states()
        self.recall.reset_states()

    def __call__(self, y_true, y_pred):
        pr = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)

        self.add_update(self.precision.updates)
        self.add_update(self.recall.updates)

        return 2 * truediv(pr * rec, pr + rec + self.epsilon)


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))


def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))


class F1Callback(Callback):
    def __init__(self):
        super(F1Callback, self).__init__()
        self.f1s = []
        self.val_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        eps = np.finfo(np.float32).eps
        recall = logs["val_true_positives"] / (logs["val_possible_positives"] + eps)
        precision = logs["val_true_positives"] / (logs["val_predicted_positives"] + eps)
        self.val_f1 = 2 * precision * recall / (precision + recall + eps)
        print("f1_val (from log) =", self.val_f1)
        self.f1s.append(self.val_f1)


class F1Metrics(Callback):
    def __init__(self):
        super(F1Metrics, self).__init__()
        self.f1_macro = MacroF1Score()

    def on_train_begin(self, logs=None):
        self.f1_macro.reset_states()

    def on_epoch_end(self, epoch, logs=None):
        pass
