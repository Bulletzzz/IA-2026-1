import torch
from torch import nn
from sklearn.metrics import accuracy_score

from preparacao import carregar, SEMENTE

EPOCAS = 10
TAXA = 0.001
TAMANHO = 128


class CnnDoZero(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.extrator = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        return self.classificador(self.extrator(x))


def avaliar(rede, carregador, dispositivo):
    rede.eval()
    reais, previstos = [], []
    with torch.no_grad():
        for imagens, rotulos in carregador:
            saida = rede(imagens.to(dispositivo)).argmax(dim=1).cpu()
            reais += rotulos.tolist()
            previstos += saida.tolist()
    return accuracy_score(reais, previstos)


def treinar():
    torch.manual_seed(SEMENTE)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    treino, teste, classes = carregar(tamanho=TAMANHO)

    rede = CnnDoZero(len(classes)).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(rede.parameters(), lr=TAXA)

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
