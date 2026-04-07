import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the first CSV file into a DataFrame
file_path1 = './svr-pfi-11feature.csv'  # Replace 'file1.csv' with the path to your first file
data1 = pd.read_csv(file_path1, header=None)

# Calculate mean and standard deviation across rows for the first file
data1_mean = data1.mean()
data1_std = data1.std()

head_list = [r'$E_\mathrm{g}^\mathrm{PBE}$', r'$E_\mathrm{coh}$', r'$\overline{|n|}$', r'$\overline{Z}$', r'$\overline{r}$', r'$\overline{\chi}$', r'$\sigma(|n|)$', r'$\sigma(p)$', r'$\sigma(m)$', '$\sigma(r)$', r'$\sigma(\chi)$']


# Assume data1_mean, data1_std, and head_list are defined
# Sorting indices in descending order based on data1_mean
sorted_indices = np.argsort(data1_mean)[::-1]
sorted_data1_mean = data1_mean[sorted_indices]
sorted_data1_std = data1_std[sorted_indices]
sorted_head_list = [head_list[i] for i in sorted_indices]

# Define the width of each bar
bar_width = 0.35

# Set positions for bars for the first file
bar_positions1 = np.arange(len(sorted_data1_mean))

# Set up the figure with two subplots and shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 3]})

# Plot the lower values in ax1
ax1.bar(bar_positions1, sorted_data1_mean, yerr=sorted_data1_std, color='green', width=bar_width, label='svr-pfi', capsize=5)
ax1.set_ylim(2.5, 3.5)  # Adjust this range to your needs
ax1.tick_params(axis='y', labelsize=15)

# Plot the higher values in ax2
ax2.bar(bar_positions1, sorted_data1_mean, yerr=sorted_data1_std, color='green', width=bar_width, label='svr-pfi', capsize=5)
ax2.set_ylim(0, 0.5)  # Adjust this range to your needs
ax2.tick_params(axis='y', labelsize=15)

# Add numerical values on each bar for ax1 and ax2
for i, (mean, std) in enumerate(zip(sorted_data1_mean, sorted_data1_std)):
    ax1.text(i, mean + std + 0.02, f'{mean:.2f}', ha='center', fontsize=18)
    ax2.text(i, mean + std + 0.02, f'{mean:.2f}', ha='center', fontsize=18)

# Add the broken axis effect
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

# Add diagonal lines to indicate the break
d = .015  # Size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Customize x-ticks
plt.xticks(bar_positions1, sorted_head_list, fontsize=18)
plt.tight_layout()

# Save the figure as a PNG file with DPI=125
plt.savefig('figure-pfi.png', dpi=125)
plt.show()

