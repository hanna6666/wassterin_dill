import argparse
import torch
import os
import torch.nn.functional as F
import time
from tqdm import tqdm

def eig_sqrt(item):
    # res = []
    # for i,item in enumerate(x):
    # print(i)
    item = item.cpu()# .to(dtype=torch.complex128)
    evals, evecs = torch.linalg.eig(item)              # get eigendecomposition
    # U,S,V = torch.linalg.svd(item)
    # print(S)
    # S = S**(1/2)
    #  print(S)
    # mpow = torch.matmul (U, torch.matmul (torch.diag (S), V))
    # print(evals)
    evals = evals**(1/2)                                # raise eigenvalues to power 1/2
    # print(evals)
    mpow = torch.matmul (evecs, torch.matmul (torch.diag (evals), torch.inverse (evecs)))
    return mpow.cuda()

def centering(K):
    n = K.size(0)
    unit = torch.ones(n,n, dtype=K.dtype).to(device=K.device)
    I = torch.eye(n, dtype=K.dtype).to(device=K.device)
    H = I - unit/n
    # return K @ H
    return torch.mm(torch.mm(H, K), H)

def linear_HSIC(X, Y):
    """
    Compute linear HSIC between X and Y.
    Args:
        X: (n, d) or (d,n)
        Y: (n, d) or (d,n)
    """
    L_X = torch.mm(X, X.transpose(0,1)) # (n, n) or (d, d)
    L_Y = torch.mm(Y, Y.transpose(0,1)) # (n, n) of (d, d)
    # L_X = torch.bmm(X, X.transpose(-1,-2)) 
    # L_Y = torch.bmm(Y, Y.transpose(-1,-2))
    return torch.sum((centering(L_X) * centering(L_Y)))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X).clamp(min=1e-12))
    var2 = torch.sqrt(linear_HSIC(Y, Y).clamp(min=1e-12))
    return hsic / (var1 * var2)

def linear_CKA_precenter(X,Y):
    """
    """
    X = X - X.mean(0,keepdim=True)
    Y = Y - Y.mean(0,keepdim=True)
    
    YTX = torch.mm(Y.transpose(0,1),X)
    hsic = torch.norm(YTX, p='fro') ** 2
    
    XTX = torch.mm(X.transpose(0,1),X)
    var1 = torch.norm(XTX, p='fro') 

    YTY = torch.mm(Y.transpose(0,1),Y)
    var2 = torch.norm(YTY, p='fro')

    return hsic / (var1 * var2)

def linear_CKA_precenter_matrix(X,Y):
    """
    Args:
        X: (N,D)
        Y: (B,N,D)
    """
    B,N,D = Y.shape
    X = X - X.mean(0,keepdim=True)
    Y = Y - Y.mean(1,keepdim=True)
    
    X = X.unsqueeze(0).repeat(B,1,1)
    YTX = torch.bmm(Y.transpose(1,2),X)
    # print("YTX",YTX.shape)
    hsic = torch.norm(YTX, p='fro', dim=(1,2)) ** 2

    XTX = torch.bmm(X.transpose(1,2),X)
    # print("XTX",XTX.shape)
    var1 = torch.norm(XTX, p='fro', dim=(1,2))

    YTY = torch.bmm(Y.transpose(1,2),Y)
    # print("YTY",YTY.shape)
    var2 = torch.norm(YTY, p='fro', dim=(1,2))

    return hsic / (var1 * var2)

def rbf(X, sigma=None):
    GX = torch.mm(X, X.transpose(0,1)) 
    KX = -2  * GX + + torch.diag(GX)[: , None] + torch.diag(GX)[None, :]
    mdist = torch.median(KX)
    KX = torch.exp(-KX / (2 * sigma ** 2 * mdist))
    return KX

def kernel_CKA(X, Y, sigma=None):
    
    rbf_X = rbf(X, sigma)
    rbf_Y = rbf(Y, sigma)

    hsic = torch.sum(centering(rbf_X) * centering(rbf_Y))
    var1 = torch.sqrt(torch.sum(centering(rbf_X) * centering(rbf_X)).clamp(min=1e-12))
    var2 = torch.sqrt(torch.sum(centering(rbf_Y) * centering(rbf_Y)).clamp(min=1e-12))
    
    return hsic / (var1 * var2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="ResNet34")
    parser.add_argument("-f", "--filename", type=str, default="features_train.pth")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="imagenet",
        choices=["cifar100", "imagenet"],
    )
    parser.add_argument("-method", "--method", type=str, default="linear_CKA")
    parser.add_argument("-savename", "--savename", type=str, default="linear_cka.pth")
    args = parser.parse_args()
    
    prefix = "wkd_cost_matrix"
    path_feat = os.path.join(prefix, args.dataset, args.model, args.filename)
    class_feat = torch.load(path_feat).float().cuda()

    num_class = class_feat.shape[0]
    
    if args.method == "linear_CKA":
        print("Calculating linear CKA distance matrix...")
    
        class_feat = class_feat 
        dist = torch.zeros(num_class, num_class).cuda()
        for i in tqdm(range(num_class)):
            for j in range(i, num_class):
                dist[i,j] = 1 - linear_CKA_precenter(class_feat[i], class_feat[j])
                dist[j,i] = dist[i,j]
                
        print(dist.shape)
        print(dist)
        print("Saving linear CKA distance matrix...")
        os.makedirs(os.path.join("./wkd_cost_matrix", args.dataset, args.model), exist_ok=True)
        torch.save(dist, os.path.join("./wkd_cost_matrix", args.dataset, args.model, args.savename)) 
    