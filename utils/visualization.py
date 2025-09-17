# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_maze(maze, path=None):
    """
    maze: 2D numpy array (H,W) 0=open, 1=wall
    path: list of (x,y) tuples
    """
    plt.imshow(maze, cmap='gray_r')
    if path:
        path = np.array(path)
        plt.plot(path[:,1], path[:,0], color='red', linewidth=2)
    plt.show()
