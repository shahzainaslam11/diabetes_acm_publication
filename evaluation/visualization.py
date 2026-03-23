import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred):
    """
    Confusion matrix visualization (aligned with notebook style)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_metric_comparison(results_dict):
    """
    Compare metrics across models or sampling strategies

    results_dict format:
    {
        "model_name": {"accuracy": ..., "f1": ...}
    }
    """
    metrics = ["accuracy", "f1", "precision", "recall"]

    for metric in metrics:
        values = [results_dict[m][metric] for m in results_dict]

        plt.figure()
        sns.barplot(x=list(results_dict.keys()), y=values)
        plt.title(f"{metric.upper()} Comparison")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.show()
