from sklearn.cluster import KMeans
import torch

feats = torch.load("dicts/resnet34x4_f2pre_dict.pt")  # shape [N, C]
feats_np = feats.numpy()

n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feats_np)

dictionary = torch.tensor(kmeans.cluster_centers_.T).float()  # shape [C, K]
print("Dictionary shape:", dictionary.shape)

torch.save(dictionary, "dicts/resnet34x4_f2pre_dictionary.pt")