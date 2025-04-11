import numpy as np

def normalize_vector(vec):
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec