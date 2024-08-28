import numpy as np
import matplotlib.pyplot as plt

# Initial data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create the figure and axis
fig, ax = plt.subplots()

# Plot the initial line plot (dynamic part)
line, = ax.plot(x, y, color='blue', label="Sine Wave")

# Plot initial fill_between (dynamic part)
# We initialize it as an empty plot since it will be updated in the loop.
fill = ax.fill_between(x, y1=0, y2=0, color='lightgray', alpha=0.5)

# Set labels and legend
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.legend()

# Draw the canvas once to enable blitting
fig.canvas.draw()

# Extract the background of the canvas (everything except the dynamic parts)
background = fig.canvas.copy_from_bbox(ax.bbox)

# Update loop
for phase in np.linspace(0, 2 * np.pi, 100):
    # Restore the background (without the dynamic parts)
    fig.canvas.restore_region(background)
    
    # Update fill_between with new y-values
    new_y1 = 0.5 * np.sin(x + phase)
    new_y2 = 0.5 * np.cos(x + phase)

    for collection in ax.collections:
        collection.remove()
    
    # # Remove the previous fill_between collection
    # if fill:
    #     fill.remove()
    
    # Create new fill_between
    fill = ax.fill_between(x, new_y1, new_y2, color='lightgray', alpha=0.5)
    
    # Update the y-data of the line plot
    line.set_ydata(np.sin(x + phase))
    
    # Redraw the updated fill_between and line
    ax.draw_artist(fill)
    ax.draw_artist(line)
    
    # Blit only the updated area
    fig.canvas.blit(ax.bbox)
    
    # Update the background with the new fill
    background = fig.canvas.copy_from_bbox(ax.bbox)
    
    # Pause briefly to create an animation effect
    plt.pause(0.01)

# Display the plot
plt.show()
