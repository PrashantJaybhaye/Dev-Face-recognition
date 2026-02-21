import numpy as np
from deepface import DeepFace

def cosine_distance(source_rep, test_rep):
    if isinstance(source_rep, list):
        source_rep = np.array(source_rep)
    if isinstance(test_rep, list):
        test_rep = np.array(test_rep)
    a = np.matmul(np.transpose(source_rep), test_rep)
    b = np.sum(np.multiply(source_rep, source_rep))
    c = np.sum(np.multiply(test_rep, test_rep))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# test using mock embeddings
emb1 = np.array([0.1, 0.2, 0.3])
emb2 = np.array([0.1, 0.2, 0.3])
emb3 = np.array([0.3, 0.2, 0.1])
print("Same:", cosine_distance(emb1, emb2))
print("Diff:", cosine_distance(emb1, emb3))
