import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report

from preparacao import preparar, SEMENTE

EPOCAS = 200
TAXA = 0.01


class RedeDensa(nn.Module):
    def __init__(self, entradas, classes):
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Linear(entradas, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, classes),
        )

    def forward(self, x):
        return self.camadas(x)


def treinar():
    torch.manual_seed(SEMENTE)
    treino_x, teste_x, treino_y, teste_y, colunas = preparar()

    treino_x = torch.from_numpy(treino_x.astype(np.float32))
    teste_x = torch.from_numpy(teste_x.astype(np.float32))
    treino_y = torch.from_numpy(treino_y.astype(np.int64))
    teste_y = torch.from_numpy(teste_y.astype(np.int64))

    classes = int(treino_y.max().item()) + 1
    rede = RedeDensa(treino_x.shape[1], classes)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(rede.parameters(), lr=TAXA, weight_decay=1e-4)

    for epoca in range(1, EPOCAS + 1):
        rede.train()
        otimizador.zero_grad()
        perda = criterio(rede(treino_x), treino_y)
        perda.backward()
        otimizador.step()
        if epoca % 20 == 0:
            print(f"epoca {epoca:3d}  perda {perda.item():.4f}")

    rede.eval()
    with torch.no_grad():
        previsoes = rede(teste_x).argmax(dim=1).numpy()

    real = teste_y.numpy()
    print("acuracia:", round(accuracy_score(real, previsoes), 4))
    print(classification_report(real, previsoes, target_names=["iniciante", "intermediario", "avancado"]))
    return rede, colunas


if __name__ == "__main__":
    treinar()
