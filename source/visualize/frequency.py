import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich.progress import Progress
from pathlib import Path
from collections import Counter
from source.dataset import MsMarcoDataset

sns.set_theme(style="dark", font_scale=1.4)
palette = sns.color_palette()

"""
Plot the unigram distribution with bag-of-words.
"""

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

"""
Plot the frequency with latent features.
"""

version, activate, dictsize = "240906A", 384, 768 * 256
base = Path(f"/data/group_data/cx_group/esae/model/{version}/computed/")
docLatentIndex = np.memmap(Path(base, "docLatentIndex.bin"), dtype=np.int32, mode="r")
docLatentValue = np.memmap(Path(base, "docLatentValue.bin"), dtype=np.float32, mode="r")
docLatentIndex = docLatentIndex.reshape(8841823, activate)
docLatentValue = docLatentValue.reshape(8841823, activate)

counter = np.zeros(dictsize, dtype=np.int32)
with Progress() as p:
    t = p.add_task("Counting", total=8841823)
    for i in range(8841823):
        indices = np.where(docLatentValue[i] > 0)
        indices = docLatentIndex[i][indices]
        counter[indices] += 1
        p.update(t, advance=1)
np.save("saved2.bin", counter)

with open("saved1.bin", "rb") as f:
    counter = pickle.load(f)
    counter = Counter(counter.values())
x1 = list(counter.keys())
y1 = list(counter.values())

counter = np.load("saved2.bin")
counter = Counter(counter)
x2 = list(counter.keys())
y2 = list(counter.values())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_xscale("log")
ax1.set_xlabel("Frequency")
ax1.set_yscale("log")
ax1.set_ylabel("Occurrence")
ax1.scatter(x1, y1, s=1, color=palette[0])
ax1.set_title("Unigram Bag-of-Words")
ax2.set_xscale("log")
ax2.set_xlabel("Frequency")
ax2.set_yscale("log")
ax2.set_ylabel("Occurrence")
ax2.scatter(x2, y2, s=1, color=palette[1])
ax2.set_title("Latent Features")
plt.subplots_adjust(wspace=0.3)
plt.savefig("frequency.pdf", bbox_inches="tight", pad_inches=0)
