# L2-normalized cosine similarity

# z = z / ||z||
# p = p / ||p||
# score = z · p


import torch


def cosine_similarity(z, p):
    z = torch.nn.functional.normalize(z, p=2, dim=-1)
    p = torch.nn.functional.normalize(p, p=2, dim=-1)
    score = torch.matmul(z, p.t())
    return score