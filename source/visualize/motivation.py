import numpy as np
import matplotlib.pyplot as plt
from source.embedding import BgeBaseEmbedding

"""
Plot the original embedding.
"""

text = "Hello, World!"
embedding = BgeBaseEmbedding()
vector = embedding.forward(text).cpu().numpy()
vector = np.clip(vector, -0.14, 0.12)

plt.axis("off")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.xlim(384 - 100, 384 + 100)
plt.plot(vector[0], color="#00bcd4")
plt.savefig("saved.svg", bbox_inches="tight", pad_inches=0, transparent=True)

"""
Plot the sparse interpretable features.
"""

vector = np.concatenate(
    [
        np.repeat(6, 12),
        np.repeat(8, 18),
        np.repeat(10, 3),
        np.repeat(12, 9),
        np.repeat(15, 5),
    ]
)

plt.axis("off")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.hist(vector, bins=20, color="#4caf50")
plt.savefig("saved.svg", bbox_inches="tight", pad_inches=0, transparent=True)

"""
Plot the reconstructed embedding.
"""

text = "Goodbye, World!"
embedding = BgeBaseEmbedding()
vector = embedding.forward(text).cpu().numpy()
vector = np.clip(vector, -0.14, 0.12)

plt.axis("off")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.xlim(384 - 100, 384 + 100)
plt.plot(vector[0], color="#ff9800")
plt.savefig("saved.svg", bbox_inches="tight", pad_inches=0, transparent=True)

"""
Plot the manipulated sparse features.
"""

vector = np.concatenate(
    [
        np.repeat(6, 12),
        np.repeat(8, 18),
        np.repeat(10, 35),
        np.repeat(12, 9),
        np.repeat(15, 50),
    ]
)

plt.axis("off")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
colors = ["#4caf50" for _ in range(20)]
colors[8], colors[19] = "#f44336", "#f44336"
_, _, patches = plt.hist(vector, bins=20)
for c, p in zip(colors, patches):
    plt.setp(p, "facecolor", c)
plt.savefig("saved.svg", bbox_inches="tight", pad_inches=0, transparent=True)
