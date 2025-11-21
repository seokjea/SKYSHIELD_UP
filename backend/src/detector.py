import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SkyShield:
    def __init__(self, attack_vectors, threshold_block=0.4):
        self.attack_vectors = attack_vectors
        self.threshold_block = threshold_block

    def set_threshold(self, new_value):
        self.threshold_block = float(new_value)

    def predict(self, user_vec):
        sims = cosine_similarity([user_vec], self.attack_vectors)[0]
        score = float(np.max(sims))

        if score >= self.threshold_block:
            return "BLOCK", score
        elif score >= self.threshold_block * 0.8:
            return "REVIEW", score
        return "ALLOW", score
