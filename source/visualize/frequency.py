import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich.progress import Progress
from pathlib import Path
from collections import Counter
from source.dataset.msMarco import MsMarcoDataset


def calc():
    # Count the frequency of unigrams in the dataset
    counter = Counter()
    dataset = MsMarcoDataset()
    with Progress() as p:
        t = p.add_task("Counting", total=8841823)
        for batch in dataset.docIter(4096):
            for text in batch:
                unigrams = text.split()
                counter.update(unigrams)
                p.advance(t)
    with open("saved1.bin", "wb") as f:
        pickle.dump(counter, f)

    # Count the frequency of latent features in the dataset
    version, activate, dictsize, dataset = "kld_x256_k128", 128, 768 * 256, "MsMarco"
    base = Path(f"/data/group_data/cx_group/esae/model/{version}/computed/{dataset}")
    index = np.memmap(Path(base, "docLatentIndex.bin"), dtype=np.int32, mode="r")
    value = np.memmap(Path(base, "docLatentValue.bin"), dtype=np.float32, mode="r")
    index = index.reshape(8841823, activate)
    value = value.reshape(8841823, activate)

    counter = np.zeros(dictsize, dtype=np.int32)
    with Progress() as p:
        t = p.add_task("Counting", total=8841823)
        for i in range(8841823):
            indices = np.where(value[i] > 0)
            indices = index[i][indices]
            counter[indices] += 1
            p.update(t, advance=1)
    np.save("saved2.npy", counter)


# def plot(sample_size=5000):
#     # Choose the plot style
#     sns.set_theme(style="whitegrid")
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#     legendSize, titleSize, labelSize, tickSize = 18, 24, 21, 18
#     plt.rcParams.update({"font.size": 18})

#     # Load the saved counter for bag-of-words
#     with open("saved1.bin", "rb") as f:
#         counter = pickle.load(f)
#         counter = Counter(counter.values())
#     x1, y1 = list(counter.keys()), list(counter.values())

#     # Sample a subset of the data
#     indices1 = np.random.choice(len(x1), size=min(sample_size, len(x1)), replace=False)
#     x1_sample, y1_sample = np.array(x1)[indices1], np.array(y1)[indices1]

#     # Load the saved counter for latent features
#     counter = np.load("saved2.npy")
#     counter = Counter(counter)
#     x2, y2 = list(counter.keys()), list(counter.values())

#     # Sample a subset of the latent features data
#     indices2 = np.random.choice(len(x2), size=min(sample_size, len(x2)), replace=False)
#     x2_sample, y2_sample = np.array(x2)[indices2], np.array(y2)[indices2]

#     # Scatter plot for bag-of-words on the first row
#     ax1.set_xscale("log")
#     ax1.set_yscale("log")
#     ax1.scatter(x1_sample, y1_sample, s=10, color="blue", alpha=0.6, marker="o")
#     ax1.set_xlabel("Frequency", fontsize=labelSize)
#     ax1.set_ylabel("Occurrence", fontsize=labelSize)
#     ax1.tick_params(axis="both", which="major", labelsize=tickSize)
#     ax1.set_title(
#         "Unigram Bag-of-Words",
#         fontsize=titleSize,
#         weight="bold",
#     )
#     ax1.grid(True, which="both", ls="--", lw=0.7)

#     # Scatter plot for latent features on the second row
#     ax2.set_xscale("log")
#     ax2.set_yscale("log")
#     ax2.scatter(x2_sample, y2_sample, s=10, color="green", alpha=0.6, marker="x")
#     ax2.set_xlabel("Frequency", fontsize=labelSize)
#     ax2.set_ylabel("Occurrence", fontsize=labelSize)
#     ax2.tick_params(axis="both", which="major", labelsize=tickSize)
#     ax2.set_title(
#         "Sparse Latent Features",
#         fontsize=titleSize,
#         weight="bold",
#     )
#     ax2.grid(True, which="both", ls="--", lw=0.7)

#     # Set the same y-limits for both plots
#     y_min = min(min(y1_sample), min(y2_sample))
#     y_max = max(max(y1_sample), max(y2_sample))
#     ax1.set_ylim([y_min, y_max])
#     ax2.set_ylim([y_min, y_max])

#     # Save the plot
#     plt.tight_layout()
#     plt.savefig("frequency.pdf", bbox_inches="tight", pad_inches=0)


def plot(sample_size=5000):
    # Choose the plot style
    sns.set_theme(style="whitegrid")
    legendSize, titleSize, labelSize, tickSize = 18, 24, 21, 18
    plt.rcParams.update({"font.size": 18})

    # Load the saved counter for bag-of-words
    with open("saved1.bin", "rb") as f:
        counter = pickle.load(f)
        counter = Counter(counter.values())
    x1, y1 = list(counter.keys()), list(counter.values())

    # Sample a subset of the data
    indices1 = np.random.choice(len(x1), size=min(sample_size, len(x1)), replace=False)
    x1_sample, y1_sample = np.array(x1)[indices1], np.array(y1)[indices1]

    # Load the saved counter for latent features
    counter = np.load("saved2.npy")
    counter = Counter(counter)
    x2, y2 = list(counter.keys()), list(counter.values())

    # Sample a subset of the latent features data
    indices2 = np.random.choice(len(x2), size=min(sample_size, len(x2)), replace=False)
    x2_sample, y2_sample = np.array(x2)[indices2], np.array(y2)[indices2]

    # Plot for bag-of-words
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.scatter(x1_sample, y1_sample, s=10, color="blue", alpha=0.6, marker="o")
    ax1.set_xlabel("Frequency", fontsize=labelSize)
    ax1.set_ylabel("Occurrence", fontsize=labelSize)
    ax1.tick_params(axis="both", which="major", labelsize=tickSize)
    # ax1.set_title("Unigram Bag-of-Words", fontsize=titleSize, weight="bold")
    ax1.grid(True, which="both", ls="--", lw=0.7)
    plt.tight_layout()
    fig1.savefig("unigram_bag_of_words.pdf", bbox_inches="tight", pad_inches=0)

    # Plot for latent features
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.scatter(x2_sample, y2_sample, s=10, color="green", alpha=0.6, marker="x")
    ax2.set_xlabel("Frequency", fontsize=labelSize)
    ax2.set_ylabel("Occurrence", fontsize=labelSize)
    ax2.tick_params(axis="both", which="major", labelsize=tickSize)
    # ax2.set_title("Sparse Latent Features", fontsize=titleSize, weight="bold")
    ax2.grid(True, which="both", ls="--", lw=0.7)
    plt.tight_layout()
    fig2.savefig("sparse_latent_features.pdf", bbox_inches="tight", pad_inches=0)

    # Close figures to free memory
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    plot()
