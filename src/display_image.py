import matplotlib.pyplot as plt




#Helper function to display image using plt. Expand on this to make a nice looking window. Use in all outputs. 


def display_image(image, title="Image"):
    """Display an image using matplotlib."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()