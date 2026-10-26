"""Matplotlib plotting utilities for square packing layouts."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from geometry import get_square_vertices, create_square_polygon

def plot_packing(state: np.ndarray, n: int, side: float, save_path: str = "packing.png"):
    """Renders the outer bounding box and individual colored rotated squares."""
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab20")
    
    polygons = []
    for i in range(n):
        x, y, theta = state[3 * i : 3 * i + 3]
        poly = create_square_polygon(x, y, theta, side)
        polygons.append(poly)
        
        verts = get_square_vertices(x, y, theta, side)
        patch = patches.Polygon(verts, closed=True, facecolor=cmap(i % 20), edgecolor="black", alpha=0.85, lw=1.5)
        ax.add_patch(patch)
        ax.text(x, y, str(i + 1), ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Compute outer boundary
    all_bounds = [p.bounds for p in polygons]
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)
    L = max(max_x - min_x, max_y - min_y)
    
    # Draw container square
    container = patches.Rectangle((min_x, min_y), L, L, fill=False, edgecolor="crimson", linestyle="--", lw=2.5, label=f"Container L={L:.4f}")
    ax.add_patch(container)

    margin = 0.5 * side
    ax.set_xlim(min_x - margin, min_x + L + margin)
    ax.set_ylim(min_y - margin, min_y + L + margin)
    ax.set_aspect('equal')
    ax.set_title(f"Squares in Square (N={n}, Side L={L:.5f})", fontsize=14, fontweight='bold')
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Visual saved to {save_path}")
