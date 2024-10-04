import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from source.embedding import BgeBaseEmbedding

sns.set_theme()
palette = sns.color_palette()

"""
Plot the original embedding.
"""

# text = "Hello, World!"
# embedding = BgeBaseEmbedding()
# vector = embedding.forward(text).cpu().numpy()
# vector = np.clip(vector, -0.14, 0.12)

# plt.axis("off")
# plt.gca().spines["top"].set_visible(False)
# plt.gca().spines["right"].set_visible(False)
# plt.gca().spines["left"].set_visible(False)
# plt.gca().spines["bottom"].set_visible(False)
# plt.plot(vector[0], color=palette[0])
# plt.savefig("saved.pdf", bbox_inches="tight", pad_inches=0)
# plt.savefig("saved.svg", bbox_inches="tight", pad_inches=0)

"""
Plot the reconstructed embedding.
"""

# plt.axis("off")
# plt.gca().spines["top"].set_visible(False)
# plt.gca().spines["right"].set_visible(False)
# plt.gca().spines["left"].set_visible(False)
# plt.gca().spines["bottom"].set_visible(False)
# plt.plot(vector[0], color=palette[1])
# plt.savefig("saved.pdf", bbox_inches="tight", pad_inches=0)
# plt.savefig("saved.svg", bbox_inches="tight", pad_inches=0)

"""
Plot the sparse interpretable features.
"""

sparse_features = np.random.randn(100)

plt.hist(sparse_features, bins=30, color=palette[2])
plt.savefig("saved.pdf", bbox_inches="tight", pad_inches=0)
plt.savefig("saved.svg", bbox_inches="tight", pad_inches=0)
plt.show()
