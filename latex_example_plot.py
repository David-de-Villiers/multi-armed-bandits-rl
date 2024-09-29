import matplotlib.pyplot as plt
import numpy as np

# Activate LaTeX-style rendering for text
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use serif fonts by default
    "font.serif": ["Computer Modern"],  # Match LaTeX's default font
    "axes.labelsize": 12,  # Set axis label font size
    "font.size": 12,  # Set general font size
    "legend.fontsize": 10,  # Set legend font size
    "xtick.labelsize": 10,  # Set x-axis tick label font size
    "ytick.labelsize": 10,  # Set y-axis tick label font size
})

# Example plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label=r'$\sin(x)$')
plt.title(r'Plot of $\sin(x)$', fontsize=14)  # Title with LaTeX math
plt.xlabel(r'$x$')
plt.ylabel(r'$\sin(x)$')
plt.legend()
plt.grid(True)

# Save figure in high-quality format for LaTeX inclusion
plt.savefig('latex_styled_plot.pdf', format='pdf', bbox_inches='tight')

# Show plot
plt.show()
