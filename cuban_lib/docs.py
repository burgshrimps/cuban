import matplotlib.pyplot as plt

colors = ['white', 'lightgrey', '#b7954b', '#5066a2', '#f0b6a0', '#6ac0b7', '#df624c', 'lightgrey', 'lightgrey']
# Create a figure and an axes, making the background transparent
fig, ax = plt.subplots(figsize=(2.5, 0.04), dpi=300)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot a lightgray rectangle with height 1 and width 10
rectangle = plt.Rectangle((0, 0), 1, 0.05, color='lightgrey')
ax.add_patch(rectangle)

# Plot a small blue portion on the rectangle
blue_portion = plt.Rectangle((0.6, 0), 0.4, 0.05, color='#df624c')  # Adjust the width as needed for the "small portion"
ax.add_patch(blue_portion)

# Remove axes
ax.axis('off')
ax.set_ylim(0, 0.05)

# Display the plot
plt.savefig('match.png', dpi=300, bbox_inches='tight', transparent=True)