import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def plot_scores(scores, mean, low, high, filename):
    # Plotting
    iterations = range(1, len(scores) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, scores, label="Score per Iteration")
    plt.axhline(
        y=mean,
        color="r",
        linestyle="-",
        label=f"Mean Score = {mean:.3f}",
    )
    plt.fill_between(
        iterations, low, high, color="gray", alpha=0.2, label="95% Confidence Interval"
    )
    plt.xlabel("Bootstrap Iteration")
    plt.ylabel("Score")
    plt.title(f'Scores Over Iterations for {filename.split("_")[0].upper()}')
    plt.legend()

    plt.savefig(filename)
    plt.close()
