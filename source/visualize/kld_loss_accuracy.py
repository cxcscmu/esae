import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", font_scale=1.4)
palette = sns.color_palette()

"""
Plot the MRR difference for retrieval on reconstruction.
"""

x1 = [32, 64, 128]
y1 = [0.2464, 0.2819, 0.3033]

x2 = [32, 64, 128]
y2 = [0.2984, 0.3194, 0.3455]

"""
Plot the MRR difference for retrieval on latent features.
"""

x3 = [32, 64, 128]
y3 = [0.1970, 0.2371, 0.2760]

x4 = [32, 64, 128]
y4 = [0.2721, 0.3062, 0.3306]

# Create subplots with more space and adjust aesthetics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot: Retrieval on Decoded
ax1.plot(x1, y1, label="MSE", color='blue', marker='o', linewidth=2)
ax1.plot(x2, y2, label="MSE+KLD", color='green', marker='s', linewidth=2)
ax1.set_xticks(x1)
ax1.set_xlabel("Activate")
ax1.set_ylabel("MRR")
ax1.set_title("Retrieval on Decoded", fontsize=16)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Second subplot: Retrieval on Latent
ax2.plot(x3, y3, label="MSE", color='blue', marker='o', linewidth=2)
ax2.plot(x4, y4, label="MSE+KLD", color='green', marker='s', linewidth=2)
ax2.set_xticks(x3)
ax2.set_xlabel("Activate")
ax2.set_ylabel("MRR")
ax2.set_title("Retrieval on Latent", fontsize=16)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

# Adjusting layout and saving the plot
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig("pretty_kld_loss_accuracy.pdf", bbox_inches="tight", pad_inches=0)
