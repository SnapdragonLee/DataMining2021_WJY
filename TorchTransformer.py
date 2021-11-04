import torch

def main():
    path = 'D:/ML/transformer/bert-base-chinese/'
    model = torch.hub.load('huggingface/pytorch-transformers','model',path)