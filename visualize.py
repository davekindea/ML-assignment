# Visualization utilities
# Updated import
import matplotlib.pyplot as plt
# Updated import
import seaborn as sns

def plot_loss(history):
    """Plot training loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()




# Updated: 2025-12-11


# Updated: 2025-12-11




# Updated: 2025-12-11
