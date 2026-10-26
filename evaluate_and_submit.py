"""Normalizer and Kaggle submission formatter."""
import pandas as pd
import numpy as np
from geometry import create_square_polygon, compute_total_overlap, compute_bounding_square_size

def export_submission(state: np.ndarray, n: int, side: float, filename: str = "submission.csv"):
    """
    Shifts coordinates to origin (min_x=0, min_y=0) and saves x, y, theta per square.
    """
    polygons = [create_square_polygon(state[3*i], state[3*i+1], state[3*i+2], side) for i in range(n)]
    min_x = min(p.bounds[0] for p in polygons)
    min_y = min(p.bounds[1] for p in polygons)
    L = compute_bounding_square_size(polygons)
    overlap = compute_total_overlap(polygons)

    rows = []
    for i in range(n):
        x = state[3 * i] - min_x
        y = state[3 * i + 1] - min_y
        theta = state[3 * i + 2] % (np.pi / 2)
        rows.append({
            "square_id": i,
            "x": x,
            "y": y,
            "theta_rad": theta,
            "theta_deg": np.degrees(theta)
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    
    print("=" * 45)
    print("SUBMISSION SUMMARY")
    print(f"Squares Packed (N):   {n}")
    print(f"Bounding Side (L):    {L:.6f}")
    print(f"Total Overlap Area:   {overlap:.8f} (Must be 0.0)")
    print(f"Exported To:          {filename}")
    print("=" * 45)
    return df
