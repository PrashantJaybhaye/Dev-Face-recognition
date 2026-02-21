import numpy as np
from deepface import DeepFace

def cosine_distance(source_rep, test_rep):
    if isinstance(source_rep, list):
        source_rep = np.array(source_rep)
    if isinstance(test_rep, list):
        test_rep = np.array(test_rep)
    a = np.dot(source_rep, test_rep)
    norm_s = np.linalg.norm(source_rep)
    norm_t = np.linalg.norm(test_rep)
    denom = norm_s * norm_t
    if denom == 0:
        return 1.0
    return 1 - (a / denom)

# test using mock embeddings
emb1 = np.array([0.1, 0.2, 0.3])
emb2 = np.array([0.1, 0.2, 0.3])
emb3 = np.array([0.3, 0.2, 0.1])
print("Same:", cosine_distance(emb1, emb2))
print("Diff:", cosine_distance(emb1, emb3))
