import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="dark", font_scale=1.4)
palette = sns.color_palette()

"""
Plot the MRR difference for retrieval on reconstruction.
"""

x1 = [32, 64, 128, 256]
y1 = [0.2464, 0.2819, 0.3033, 0.3261]

x2 = [32, 64, 128, 256]
y2 = [0.2984, 0.3194, 0.3455, 0.3567]

"""
Plot the MRR difference for retrieval on latent features.
"""

x3 = [32, 64, 128, 256]
y3 = [0.1970, 0.2371, 0.2760, 0.2996]

x4 = [32, 64, 128, 256]
y4 = [0.2721, 0.3062, 0.3306, 0.3458]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(x1, y1, label="MSE")
ax1.plot(x2, y2, label="MSE+KLD")
ax1.set_xticks(x1)
ax1.set_xlabel("Activate")
ax1.set_ylabel("MRR")
ax1.set_title("Retrieval on Decoded")
ax1.legend()
ax2.plot(x3, y3, label="MSE")
ax2.plot(x4, y4, label="MSE+KLD")
ax2.set_xticks(x3)
ax2.set_xlabel("Activate")
ax2.set_ylabel("MRR")
ax2.set_title("Retrieval on Latent")
ax2.legend()
plt.subplots_adjust(wspace=0.4)
plt.savefig("ablation.pdf", bbox_inches="tight", pad_inches=0)
