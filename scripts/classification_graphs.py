import numpy as np
import matplotlib.pyplot as plt

def comparison_graph(model_early, model_normal, model_late):
    """
    Generate a comparative performance bar chart for classification models.

    This function:
    - Extracts accuracy and F1 score metrics from three classification models.
    - Creates a grouped bar chart comparing model performance.
    - Annotates bars with metric values.
    - Saves and displays the resulting figure.

    Parameters:
    model_early (dict):
        Results dictionary for the Early onset classifier.

    model_normal (dict):
        Results dictionary for the Normal onset classifier.

    model_late (dict):
        Results dictionary for the Late onset classifier.

    Returns:
    None

    Outputs:
    - Displays grouped bar chart.
    - Saves figure as:
        "classification_model_comparison.png"

    Notes:
    - Assumes each model dictionary contains:
        - "accuracy"
        - "f1_score"
    - Metrics are displayed on a 0–1 scale.
    """
    models = {
        "Early": model_early,
        "Normal": model_normal,
        "Late": model_late
    }
    labels = list(models.keys())
    accuracies = [m["accuracy"] for m in models.values()]
    f1_scores = [m["f1_score"] for m in models.values()]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#4A7FB5")
    bars2 = ax.bar(x + width/2, f1_scores, width, label="F1 Score", color="#2A9D6E")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Classification Model Comparison: Accuracy & F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("classification_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

def confusion_graph(model_early, model_normal, model_late):
    """
    Generate a comparative performance bar chart for classification models.

    This function:
    - Extracts accuracy and F1 score metrics from three classification models.
    - Creates a grouped bar chart comparing model performance.
    - Annotates bars with metric values.
    - Saves and displays the resulting figure.

    Parameters:
    model_early (dict):
        Results dictionary for the Early onset classifier.

    model_normal (dict):
        Results dictionary for the Normal onset classifier.

    model_late (dict):
        Results dictionary for the Late onset classifier.

    Returns:
    None

    Outputs:
    - Displays grouped bar chart.
    - Saves figure as:
        "classification_model_comparison.png"

    Notes:
    - Assumes each model dictionary contains:
        - "accuracy"
        - "f1_score"
    - Metrics are displayed on a 0–1 scale.
    """
    models = {
        "Early": model_early,
        "Normal": model_normal,
        "Late": model_late
    }
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, (name, model) in zip(axes, models.items()):
        cm = model["confusion_matrix"]
        classes = model["classes"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.colorbar(im, ax=ax)
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()