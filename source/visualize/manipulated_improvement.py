# import seaborn as sns
# import matplotlib.pyplot as plt

# # Increment with Sparse Latent
# x = [0.000, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]

# # Mean Reciprocal Rank (MRR), Precision at 10 (P@10), Recall at 10 (R@10)
# y1 = [0.3011, 0.3159, 0.3207, 0.3659, 0.3731, 0.4539, 0.5921, 0.7965, 0.9495]
# y2 = [0.0582, 0.0596, 0.0591, 0.0655, 0.0642, 0.0735, 0.0853, 0.0976, 0.1066]
# y3 = [0.5529, 0.5695, 0.5607, 0.6223, 0.6082, 0.6926, 0.8015, 0.9260, 0.9929]

# sns.set_theme(style="whitegrid")
# fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))
# plt.rcParams.update({"font.size": 18})

# ax1.set_xscale("log")
# ax1.plot(x, y1, label="MRR", marker="o", alpha=0.7, color="b", lw=2)
# # ax1.plot(x, y3, label="R@10", marker="s", alpha=0.7, color="g", lw=2)
# ax1.set_xlabel("Increment with Sparse Latent", fontsize=20)
# # ax1.set_ylabel("MRR / R@10", fontsize=20, color="black")
# ax1.set_ylabel("MRR", fontsize=20, color="black")
# ax1.grid(True, which="both", ls="--", lw=0.7)

# ax2 = ax1.twinx()
# ax2.plot(x, y2, label="P@10", marker="x", alpha=0.7, color="r", lw=2, ls="--")
# ax2.set_ylabel("P@10", fontsize=20, color="black")
# ax2.grid(False)

# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# handles, labels = lines_1 + lines_2, labels_1 + labels_2
# ax1.legend(handles, labels, loc="upper left", fontsize=18, frameon=True)

# ax1.tick_params(axis="both", which="major", labelsize=16)
# ax2.tick_params(axis="both", which="major", labelsize=16)
# ax1.set_title("Manipulated Document for Retrieval", fontsize=22, weight="bold")

# # Mean Reciprocal Rank (MRR), Precision at 10 (P@10), Recall at 10 (R@10)
# y1 = [0.3708, 0.3719, 0.3733, 0.3741, 0.3782, 0.3871, 0.3994, 0.4182, 0.4285]
# y2 = [0.0654, 0.0654, 0.0654, 0.0656, 0.0658, 0.0667, 0.0685, 0.0689, 0.0678]
# y3 = [0.6243, 0.6243, 0.6243, 0.6263, 0.6278, 0.6352, 0.6520, 0.6560, 0.6515]

# ax3.set_xscale("log")
# ax3.plot(x, y1, label="MRR", marker="o", alpha=0.7, color="b", lw=2)
# # ax3.plot(x, y3, label="R@10", marker="s", alpha=0.7, color="g", lw=2)
# ax3.set_xlabel("Increment with Sparse Latent", fontsize=20)
# # ax3.set_ylabel("MRR / R@10", fontsize=20, color="black")
# ax3.set_ylabel("MRR", fontsize=20, color="black")
# ax3.grid(True, which="both", ls="--", lw=0.7)

# ax4 = ax3.twinx()
# ax4.plot(x, y2, label="P@10", marker="x", alpha=0.7, color="r", lw=2, ls="--")
# ax4.set_ylabel("P@10", fontsize=20, color="black")
# ax4.grid(False)

# lines_1, labels_1 = ax3.get_legend_handles_labels()
# lines_2, labels_2 = ax4.get_legend_handles_labels()
# handles, labels = lines_1 + lines_2, labels_1 + labels_2
# ax3.legend(handles, labels, loc="upper left", fontsize=18, frameon=True)

# ax3.tick_params(axis="both", which="major", labelsize=16)
# ax4.tick_params(axis="both", which="major", labelsize=16)
# ax3.set_title("Manipulated Query for Retrieval", fontsize=22, weight="bold")

# plt.subplots_adjust(wspace=0.4)
# plt.savefig("manipulated_improvement.pdf", bbox_inches="tight", pad_inches=0.1)

import seaborn as sns
import matplotlib.pyplot as plt

# Data
x = [0.000, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]
y1_set1 = [0.3011, 0.3159, 0.3207, 0.3659, 0.3731, 0.4539, 0.5921, 0.7965, 0.9495]
y2_set1 = [0.0582, 0.0596, 0.0591, 0.0655, 0.0642, 0.0735, 0.0853, 0.0976, 0.1066]

y1_set2 = [0.3708, 0.3719, 0.3733, 0.3741, 0.3782, 0.3871, 0.3994, 0.4182, 0.4285]
y2_set2 = [0.0654, 0.0654, 0.0654, 0.0656, 0.0658, 0.0667, 0.0685, 0.0689, 0.0678]

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 18})

# Plot for Manipulated Document for Retrieval
def plot_manipulated_document():
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xscale("log")
    ax1.plot(x, y1_set1, label="MRR", marker="o", alpha=0.7, color="b", lw=2)
    ax1.set_xlabel("Increment with Sparse Latent", fontsize=20)
    ax1.set_ylabel("MRR", fontsize=20, color="black")
    ax1.grid(True, which="both", ls="--", lw=0.7)

    ax2 = ax1.twinx()
    ax2.plot(x, y2_set1, label="P@10", marker="x", alpha=0.7, color="r", lw=2, ls="--")
    ax2.set_ylabel("P@10", fontsize=20, color="black")
    ax2.grid(False)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    handles, labels = lines_1 + lines_2, labels_1 + labels_2
    ax1.legend(handles, labels, loc="upper left", fontsize=18, frameon=True)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)
    # ax1.set_title("Manipulated Document for Retrieval", fontsize=22, weight="bold")

    plt.tight_layout()
    fig.savefig("manipulated_document_retrieval.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# Plot for Manipulated Query for Retrieval
def plot_manipulated_query():
    fig, ax3 = plt.subplots(figsize=(8, 6))
    ax3.set_xscale("log")
    ax3.plot(x, y1_set2, label="MRR", marker="o", alpha=0.7, color="b", lw=2)
    ax3.set_xlabel("Increment with Sparse Latent", fontsize=20)
    ax3.set_ylabel("MRR", fontsize=20, color="black")
    ax3.grid(True, which="both", ls="--", lw=0.7)

    ax4 = ax3.twinx()
    ax4.plot(x, y2_set2, label="P@10", marker="x", alpha=0.7, color="r", lw=2, ls="--")
    ax4.set_ylabel("P@10", fontsize=20, color="black")
    ax4.grid(False)

    # Combine legends from both axes
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax4.get_legend_handles_labels()
    handles, labels = lines_1 + lines_2, labels_1 + labels_2
    ax3.legend(handles, labels, loc="upper left", fontsize=18, frameon=True)

    ax3.tick_params(axis="both", which="major", labelsize=16)
    ax4.tick_params(axis="both", which="major", labelsize=16)
    # ax3.set_title("Manipulated Query for Retrieval", fontsize=22, weight="bold")

    plt.tight_layout()
    fig.savefig("manipulated_query_retrieval.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# Run the plot functions
plot_manipulated_document()
plot_manipulated_query()
