import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich.progress import Progress
from pathlib import Path
from collections import Counter
from source.dataset import MsMarcoDataset


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


def plot():
    # Set a modern aesthetic style using seaborn
    sns.set_theme(style="whitegrid")

    # Create figure and axes with larger size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Log scale settings
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Axis labels with increased font size
    ax.set_xlabel("Frequency", fontsize=14)
    ax.set_ylabel("Occurrence", fontsize=14)

    # Title with larger font size
    ax.set_title("Frequency Distribution", fontsize=16, weight="bold")

    # Load the saved counter for bag-of-words
    with open("saved1.bin", "rb") as f:
        counter = pickle.load(f)
        counter = Counter(counter.values())
    x1, y1 = list(counter.keys()), list(counter.values())

    # Scatter plot for bag-of-words
    ax.scatter(
        x1, y1, s=10, color="blue", alpha=0.6, label="Unigram Bag-of-Words", marker="o"
    )

    # Load the saved counter for latent features
    counter = np.load("saved2.npy")
    counter = Counter(counter)
    x2, y2 = list(counter.keys()), list(counter.values())

    # Scatter plot for latent features
    ax.scatter(
        x2, y2, s=10, color="green", alpha=0.6, label="Latent Features", marker="x"
    )

    # Adding gridlines for better readability
    ax.grid(True, which="both", ls="--", lw=0.5)

    # Adding a legend with a better position and larger font size
    ax.legend(loc="upper right", fontsize=12, frameon=True)

    # Save the figure with padding and tight layout
    plt.savefig("frequency.pdf", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    plot()
