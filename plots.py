import numpy as np
import matplotlib.pyplot as plt

def plotResults(train_sizes, train_labels, train_precisions, dev_precisions, train_recalls, dev_recalls, train_f1s, dev_f1s):
    # Plot the learning curves
    plt.figure(figsize=(12, 8))

    # Plot Precision
    plt.subplot(3, 1, 1)
    plt.plot(train_sizes * len(train_labels), train_precisions, label='Train Precision', marker='o')
    plt.plot(train_sizes * len(train_labels), dev_precisions, label='Dev Precision', marker='x')
    plt.xlabel('Training Size')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.title('Learning Curves: Precision')

    # Plot Recall
    plt.subplot(3, 1, 2)
    plt.plot(train_sizes * len(train_labels), train_recalls, label='Train Recall', marker='o')
    plt.plot(train_sizes * len(train_labels), dev_recalls, label='Dev Recall', marker='x')
    plt.xlabel('Training Size')
    plt.ylabel('Recall')
    plt.legend(loc='best')
    plt.title('Learning Curves: Recall')

    # Plot F1-Score
    plt.subplot(3, 1, 3)
    plt.plot(train_sizes * len(train_labels), train_f1s, label='Train F1-Score', marker='o')
    plt.plot(train_sizes * len(train_labels), dev_f1s, label='Dev F1-Score', marker='x')
    plt.xlabel('Training Size')
    plt.ylabel('F1-Score')
    plt.legend(loc='best')
    plt.title('Learning Curves: F1-Score')

    # Show the plot
    plt.tight_layout()
    plt.show()