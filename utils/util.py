import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def plot_nmi_scores(iterations, nmi_scores, mean, low, high):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, nmi_scores, label="NMI Score per Iteration")
    plt.axhline(
        y=mean,
        color="r",
        linestyle="-",
        label=f"Mean NMI Score = {mean:.3f}",
    )
    plt.fill_between(
        iterations, low, high, color="gray", alpha=0.2, label="95% Confidence Interval"
    )
    plt.xlabel("Bootstrap Iteration")
    plt.ylabel("NMI Score")
    plt.title("NMI Scores with Confidence Interval Over Iterations")
    plt.legend()

    plt.savefig("my_plot.png")
    plt.plot()
