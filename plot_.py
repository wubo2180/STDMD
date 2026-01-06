# Reorganized version of the final plotting script
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# Use fallback font if Times New Roman is not available
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 26,
    'axes.titlesize': 26,
    'axes.labelsize': 26,
    'legend.fontsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.unicode_minus': False
})

# Data for the plots
data_dict = {
    "meta_lr": ['0.001', '0.005', '0.01', '0.015', '0.02'],
    "update_lr": ['0.001', '0.005', '0.01', '0.015', '0.02'],
    "k_spt": ['50', '100', '150', '200', '250'],
    "k_qry": ['50', '100', '150', '200', '250'],
    "hidden_dim": ['16', '32', '64', '128', '256'],
    "coefficient": ['0.1', '0.3', '0.5', '0.7', '0.9'],
    "mse_meta_lr": [0.7791, 0.7712, 0.7650, 0.7862, 0.7991],
    "mse_update_lr": [0.7762, 0.7693, 0.7650, 0.7817, 0.8035],
    "mse_k_spt": [0.7911, 0.7650, 0.7762, 0.7813, 0.7853],
    "mse_k_qry": [0.7921, 0.7650, 0.7782, 0.7752, 0.7824],
    "mse_hidden_dim": [0.7682, 0.7650, 0.7648, 0.7635, 0.7632],
    "mse_coefficient": [0.7881, 0.7701, 0.7650, 0.7832, 0.7931]
}

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# Subplot (a)
axes[0].plot(data_dict["meta_lr"], data_dict["mse_meta_lr"], marker='o', markersize=10, linewidth=3, linestyle='--', label=r'$\alpha$')
axes[0].plot(data_dict["update_lr"], data_dict["mse_update_lr"], marker='o', markersize=10, linewidth=3, linestyle='--', label=r'$\gamma$')
axes[0].set_title("(a) Learning Rate Sensitivity")
axes[0].set_xlabel(r"Meta_lr $\alpha$ and Update_lr $\gamma$")
axes[0].set_ylabel("MSE")
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
axes[0].set_yticks(np.arange(0.7650, 0.8001, 0.005))
axes[0].legend()
axes[0].grid(True)

# Subplot (b)
axes[1].plot(data_dict["k_spt"], data_dict["mse_k_spt"], marker='o', markersize=10, linewidth=3, linestyle='--', label='k_spt')
axes[1].plot(data_dict["k_qry"], data_dict["mse_k_qry"], marker='o', markersize=10, linewidth=3, linestyle='--', label='k_qry')
axes[1].set_title("(b) Support/Query Sample Size")
axes[1].set_xlabel("k_spt and k_qry Samples")
axes[1].set_ylabel("MSE")
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
axes[1].set_yticks(np.arange(0.7650, 0.8001, 0.005))
axes[1].legend()
axes[1].grid(True)

# Subplot (c)
axes[2].plot(data_dict["hidden_dim"], data_dict["mse_hidden_dim"], marker='o', markersize=10, linewidth=3, linestyle='--', label=r'$d$')
axes[2].set_title("(c) Hidden Dimension Size")
axes[2].set_xlabel("Hidden Dimension")
axes[2].set_ylabel("MSE")
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
axes[2].set_yticks(np.arange(0.7621, 0.7701, 0.002))
axes[2].legend()
axes[2].grid(True)

# Subplot (d)
axes[3].plot(data_dict["coefficient"], data_dict["mse_coefficient"], marker='o', markersize=10, linewidth=3, linestyle='--', label=r'$\beta$')
axes[3].set_title("(d) Loss Balance Coefficient")
axes[3].set_xlabel("Balance Coefficient")
axes[3].set_ylabel("MSE")
axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
axes[3].set_yticks(np.arange(0.7621, 0.8001, 0.005))
axes[3].legend()
axes[3].grid(True)

# Final layout
plt.tight_layout()
output_path = "figure/Updated_Hyperparameter_Study_WikiMaths_MSE_bigfont_Cleaned.pdf"
plt.savefig(output_path, format="pdf")
plt.show()