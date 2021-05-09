from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchmetrics import Accuracy, Precision, Recall, F1


def compute_metrics_sklearn(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def compute_metrics_torch(pred, true, num_classes, average="macro"):
    p = Precision(average=average, num_classes=num_classes)(pred, true)
    r = Recall(average=average, num_classes=num_classes)(pred, true)
    f1 = F1(average=average, num_classes=num_classes)(pred, true)
    acc = Accuracy()(pred, true)
    return acc, p, r, f1
