# import seaborn as sns
# import matplotlib.pyplot as plt

# # MRR difference for retrieval on reconstructed embedding, MsMarco.
# x1 = [32, 64, 128]
# y1 = [0.2464, 0.2819, 0.3033]
# x2 = [32, 64, 128]
# y2 = [0.2984, 0.3194, 0.3455]

# # MRR difference for retrieval on sparse latent features, MsMarco.
# x3 = [32, 64, 128]
# y3 = [0.1970, 0.2371, 0.2760]
# x4 = [32, 64, 128]
# y4 = [0.2721, 0.3062, 0.3306]

# # MRR difference for retrieval on reconstructed embedding, BEIR.
# x5 = [32, 64, 128]
# y5 = [0.1683, 0.2365, 0.2848]
# x6 = [32, 64, 128]
# y6 = [0.2549, 0.2913, 0.3407]

# # MRR difference for retrieval on sparse latent features, BEIR.
# x7 = [32, 64, 128]
# y7 = [0.1635, 0.1964, 0.2492]
# x8 = [32, 64, 128]
# y8 = [0.2420, 0.2923, 0.3407]

# sns.set_theme(style="whitegrid")
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# legendSize, titleSize, labelSize, tickSize = 18, 24, 21, 18
# plt.rcParams.update({"font.size": 18})

# ax1.plot(x1, y1, label="MSE", alpha=0.7, color="b", marker="o", lw=2)
# ax1.plot(x2, y2, label="MSE+KLD", alpha=0.7, color="g", marker="x", lw=2)
# ax1.legend(loc="lower right", fontsize=legendSize, frameon=True)
# ax1.set_xticks(x1)
# ax1.set_xlabel("Active Latents", fontsize=labelSize)
# ax1.set_ylabel("MRR", fontsize=labelSize)
# ax1.grid(True, linestyle="--", alpha=0.7)
# ax1.tick_params(axis="both", which="major", labelsize=tickSize)
# ax1.set_title("Retrieval on MsMarco (Rec.)", fontsize=titleSize, weight="bold")

# ax2.plot(x3, y3, label="MSE", alpha=0.7, color="b", marker="o", lw=2)
# ax2.plot(x4, y4, label="MSE+KLD", alpha=0.7, color="g", marker="x", lw=2)
# ax2.legend(loc="lower right", fontsize=legendSize, frameon=True)
# ax2.set_xticks(x3)
# ax2.set_xlabel("Active Latents", fontsize=labelSize)
# ax2.set_ylabel("MRR", fontsize=labelSize)
# ax2.grid(True, linestyle="--", alpha=0.7)
# ax2.tick_params(axis="both", which="major", labelsize=tickSize)
# ax2.set_title("Retrieval on MsMarco (Spr.)", fontsize=titleSize, weight="bold")

# y_min_row1 = min(min(y1), min(y2), min(y3), min(y4))
# y_max_row1 = max(max(y1), max(y2), max(y3), max(y4))
# ax1.set_ylim(y_min_row1 - 0.01, y_max_row1 + 0.01)
# ax2.set_ylim(y_min_row1 - 0.01, y_max_row1 + 0.01)

# ax3.plot(x5, y5, label="MSE", alpha=0.7, color="b", marker="o", lw=2)
# ax3.plot(x6, y6, label="MSE+KLD", alpha=0.7, color="g", marker="x", lw=2)
# ax3.legend(loc="lower right", fontsize=legendSize, frameon=True)
# ax3.set_xticks(x5)
# ax3.set_xlabel("Active Latents", fontsize=labelSize)
# ax3.set_ylabel("MRR", fontsize=labelSize)
# ax3.grid(True, linestyle="--", alpha=0.7)
# ax3.tick_params(axis="both", which="major", labelsize=tickSize)
# ax3.set_title("Retrieval on BEIR (Rec.)", fontsize=titleSize, weight="bold")

# ax4.plot(x7, y7, label="MSE", alpha=0.7, color="b", marker="o", lw=2)
# ax4.plot(x8, y8, label="MSE+KLD", alpha=0.7, color="g", marker="x", lw=2)
# ax4.legend(loc="lower right", fontsize=legendSize, frameon=True)
# ax4.set_xticks(x7)
# ax4.set_xlabel("Active Latents", fontsize=labelSize)
# ax4.set_ylabel("MRR", fontsize=labelSize)
# ax4.grid(True, linestyle="--", alpha=0.7)
# ax4.tick_params(axis="both", which="major", labelsize=tickSize)
# ax4.set_title("Retrieval on BEIR (Spr.)", fontsize=titleSize, weight="bold")

# y_min_row2 = min(min(y5), min(y6), min(y7), min(y8))
# y_max_row2 = max(max(y5), max(y6), max(y7), max(y8))
# ax3.set_ylim(y_min_row2 - 0.01, y_max_row2 + 0.01)
# ax4.set_ylim(y_min_row2 - 0.01, y_max_row2 + 0.01)

# plt.subplots_adjust(wspace=0.25, hspace=0.3)
# plt.savefig("ablation.pdf", bbox_inches="tight", pad_inches=0)


import seaborn as sns
import matplotlib.pyplot as plt

# Data
x1 = [32, 64, 128]
y1 = [0.2464, 0.2819, 0.3033]
x2 = [32, 64, 128]
y2 = [0.2984, 0.3194, 0.3455]

x3 = [32, 64, 128]
y3 = [0.1970, 0.2371, 0.2760]
x4 = [32, 64, 128]
y4 = [0.2721, 0.3062, 0.3306]

x5 = [32, 64, 128]
y5 = [0.1683, 0.2365, 0.2848]
x6 = [32, 64, 128]
y6 = [0.2549, 0.2913, 0.3407]

x7 = [32, 64, 128]
y7 = [0.1635, 0.1964, 0.2492]
x8 = [32, 64, 128]
y8 = [0.2420, 0.2923, 0.3407]

sns.set_theme(style="whitegrid")
legendSize, labelSize, tickSize = 24, 24, 18
plt.rcParams.update({"font.size": 18})

# Plot and save the first figure
plt.figure(figsize=(8, 6))
plt.plot(x1, y1, label="MSE", alpha=0.7, color="b", marker="o", lw=2)
plt.plot(x2, y2, label="MSE+KLD", alpha=0.7, color="g", marker="x", lw=2)
plt.legend(loc="lower right", fontsize=legendSize, frameon=True)
plt.xticks(x1)
plt.xlabel("Active Latents", fontsize=labelSize)
plt.ylabel("MRR", fontsize=labelSize)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=tickSize)
plt.ylim(min(min(y1), min(y2)) - 0.01, max(max(y1), max(y2)) + 0.01)
plt.savefig("ablation1.pdf", bbox_inches="tight")
plt.close()

# Plot and save the second figure
plt.figure(figsize=(8, 6))
plt.plot(x3, y3, label="MSE", alpha=0.7, color="b", marker="o", lw=2)
plt.plot(x4, y4, label="MSE+KLD", alpha=0.7, color="g", marker="x", lw=2)
plt.legend(loc="lower right", fontsize=legendSize, frameon=True)
plt.xticks(x3)
plt.xlabel("Active Latents", fontsize=labelSize)
plt.ylabel("MRR", fontsize=labelSize)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=tickSize)
plt.ylim(min(min(y3), min(y4)) - 0.01, max(max(y3), max(y4)) + 0.01)
plt.savefig("ablation2.pdf", bbox_inches="tight")
plt.close()

# Plot and save the third figure
plt.figure(figsize=(8, 6))
plt.plot(x5, y5, label="MSE", alpha=0.7, color="b", marker="o", lw=2)
plt.plot(x6, y6, label="MSE+KLD", alpha=0.7, color="g", marker="x", lw=2)
plt.legend(loc="lower right", fontsize=legendSize, frameon=True)
plt.xticks(x5)
plt.xlabel("Active Latents", fontsize=labelSize)
plt.ylabel("MRR", fontsize=labelSize)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=tickSize)
plt.ylim(min(min(y5), min(y6)) - 0.01, max(max(y5), max(y6)) + 0.01)
plt.savefig("ablation3.pdf", bbox_inches="tight")
plt.close()

# Plot and save the fourth figure
plt.figure(figsize=(8, 6))
plt.plot(x7, y7, label="MSE", alpha=0.7, color="b", marker="o", lw=2)
plt.plot(x8, y8, label="MSE+KLD", alpha=0.7, color="g", marker="x", lw=2)
plt.legend(loc="lower right", fontsize=legendSize, frameon=True)
plt.xticks(x7)
plt.xlabel("Active Latents", fontsize=labelSize)
plt.ylabel("MRR", fontsize=labelSize)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=tickSize)
plt.ylim(min(min(y7), min(y8)) - 0.01, max(max(y7), max(y8)) + 0.01)
plt.savefig("ablation4.pdf", bbox_inches="tight")
plt.close()
