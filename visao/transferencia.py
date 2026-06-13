import torch
from torch import nn
from torchvision import models
from sklearn.metrics import accuracy_score

from preparacao import carregar, SEMENTE
from cnn import avaliar

EPOCAS = 5
TAXA = 0.001
TAMANHO = 224


def montar(classes):
    rede = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for parametro in rede.features.parameters():
        parametro.requires_grad = False
    entradas = rede.classifier[1].in_features
    rede.classifier[1] = nn.Linear(entradas, classes)
    return rede


def treinar():
    torch.manual_seed(SEMENTE)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    treino, teste, classes = carregar(tamanho=TAMANHO)

    rede = montar(len(classes)).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    treinaveis = [p for p in rede.parameters() if p.requires_grad]
    otimizador = torch.optim.Adam(treinaveis, lr=TAXA)

    for epoca in range(1, EPOCAS + 1):
        rede.train()
        perda_total = 0.0
        for imagens, rotulos in treino:
            imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)
            otimizador.zero_grad()
            perda = criterio(rede(imagens), rotulos)
            perda.backward()
            otimizador.step()
            perda_total += perda.item()
        print(f"epoca {epoca:2d}  perda {perda_total / len(treino):.4f}")

    print("acuracia:", round(avaliar(rede, teste, dispositivo), 4))
    return rede, classes


if __name__ == "__main__":
    treinar()
