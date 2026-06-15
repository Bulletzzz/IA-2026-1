from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import random

CAMINHO = Path(__file__).resolve().parent.parent / "dados" / "imagens"
SEMENTE = 42
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]


def transformacoes(tamanho, treino):
    if treino:
        return transforms.Compose([
            transforms.RandomResizedCrop(tamanho, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA, DESVIO),
        ])
    return transforms.Compose([
        transforms.Resize(int(tamanho * 1.15)),
        transforms.CenterCrop(tamanho),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIO),
    ])


def limitar_por_classe(conjunto, teto):
    indices_por_classe = defaultdict(list)
    for idx, (_, rotulo) in enumerate(conjunto.samples):
        indices_por_classe[rotulo].append(idx)
    rng = random.Random(SEMENTE)
    indices = []
    for lista in indices_por_classe.values():
        indices.extend(rng.sample(lista, min(teto, len(lista))))
    return indices


def carregar(tamanho=224, teto=500, lote=64):
    gerador = torch.Generator().manual_seed(SEMENTE)
    base_treino = datasets.ImageFolder(CAMINHO, transform=transformacoes(tamanho, True))
    base_teste = datasets.ImageFolder(CAMINHO, transform=transformacoes(tamanho, False))

    indices = limitar_por_classe(base_treino, teto)
    embaralhados = torch.randperm(len(indices), generator=gerador).tolist()
    indices = [indices[i] for i in embaralhados]
    corte = int(len(indices) * 0.8)

    treino = Subset(base_treino, indices[:corte])
    teste = Subset(base_teste, indices[corte:])

    carregador_treino = DataLoader(treino, batch_size=lote, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    carregador_teste = DataLoader(teste, batch_size=lote, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    return carregador_treino, carregador_teste, base_treino.classes
