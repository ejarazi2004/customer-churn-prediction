import os 
import matplotlib.pyplot as plt

def save_fig(filename, subfolder="eda", fmt="png", dpi=300):
    folder = os.path.join("results", subfolder)
    
    os.makedirs(folder, exist_ok=True)
    
    save_path = os.path.join(folder, f"{filename}.{fmt}")
    
    plt.savefig(save_path, format=fmt, dpi=dpi, bbox_inches="tight")