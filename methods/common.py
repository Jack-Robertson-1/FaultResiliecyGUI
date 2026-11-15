from collections import Counter

def vote(results):
    """Majority vote by detection count."""
    counts = [len(r) for r in results]
    voted_count = Counter(counts).most_common(1)[0][0]
    for r in results:
        if len(r) == voted_count:
            return r
    return results[0]

def classify(golden_boxes, result):
    """Compare detected classes (order-independent) against golden, return SDC or masked."""
    if len(result) != len(golden_boxes):
        return "SDC"
    golden_classes = sorted([g[0] for g in golden_boxes])
    result_classes = sorted([f[0] for f in result])
    if golden_classes != result_classes:
        return "SDC"
    return "masked"

def detections_agree(d1, d2):
    """Check if two results agree (same classes, order-independent)."""
    if len(d1) != len(d2):
        return False
    classes1 = sorted([a[0] for a in d1])
    classes2 = sorted([b[0] for b in d2])
    return classes1 == classes2
