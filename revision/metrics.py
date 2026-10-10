"""Binary metrics and validation-only threshold selection."""
import math


def validate(labels, probabilities):
    labels, probabilities = list(labels), list(probabilities)
    if not labels or len(labels) != len(probabilities):
        raise ValueError('Labels and probabilities must be nonempty and equal-length.')
    if any(y not in (0, 1) for y in labels):
        raise ValueError('Labels must be zero or one.')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Probabilities must be finite and in [0,1].')
    return labels, probabilities


def binary_metrics(labels, probabilities, threshold):
    labels, probabilities = validate(labels, probabilities)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1].')
    predicted = [p >= threshold for p in probabilities]
    tp = sum(y == 1 and p for y, p in zip(labels, predicted))
    fn = sum(y == 1 and not p for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p for y, p in zip(labels, predicted))
    tn = sum(y == 0 and not p for y, p in zip(labels, predicted))
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, n_positive=tp+fn, n_negative=tn+fp,
                precision=tp/(tp+fp) if tp+fp else 0.0,
                recall=tp/(tp+fn) if tp+fn else 0.0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                accuracy=(tp+tn)/len(labels), fpr=fp/(fp+tn) if fp+tn else 0.0,
                threshold=threshold)


def validation_threshold(labels, probabilities):
    labels, probabilities = validate(labels, probabilities)
    if set(labels) != {0, 1}:
        raise ValueError('Validation threshold selection requires both classes.')
    thresholds = sorted(set([0.0, 1.0] + probabilities))
    # Enumerated in ascending order: exact F1 ties select the lowest threshold.
    return max(thresholds, key=lambda t: binary_metrics(labels, probabilities, t)['f1'])
