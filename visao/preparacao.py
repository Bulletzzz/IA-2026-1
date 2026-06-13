from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CAMINHO = Path(__file__).resolve().parent.parent / "dados" / "imagens"
SEMENTE = 42
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]


def transformacoes(tamanho, treino):
    etapas = [transforms.Resize((tamanho, tamanho))]
    if treino:
        etapas += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
    etapas += [transforms.ToTensor(), transforms.Normalize(MEDIA, DESVIO)]
    return transforms.Compose(etapas)


def limitar_por_classe(conjunto, teto):
    contagem = {}
    indices = []
    for indice, (_, rotulo) in enumerate(conjunto.samples):
        if contagem.get(rotulo, 0) < teto:
            indices.append(indice)
            contagem[rotulo] = contagem.get(rotulo, 0) + 1
    return indices


def carregar(tamanho=128, teto=200, lote=32):
    gerador = torch.Generator().manual_seed(SEMENTE)
    base = datasets.ImageFolder(CAMINHO)
    indices = limitar_por_classe(base, teto)
    corte = int(len(indices) * 0.8)
    embaralhados = torch.randperm(len(indices), generator=gerador).tolist()
    indices = [indices[i] for i in embaralhados]

    treino_base = datasets.ImageFolder(CAMINHO, transform=transformacoes(tamanho, True))
    teste_base = datasets.ImageFolder(CAMINHO, transform=transformacoes(tamanho, False))
    treino = Subset(treino_base, indices[:corte])
    teste = Subset(teste_base, indices[corte:])

    carregador_treino = DataLoader(treino, batch_size=lote, shuffle=True)
    carregador_teste = DataLoader(teste, batch_size=lote, shuffle=False)
    return carregador_treino, carregador_teste, base.classes
