import torch
from torch import nn
from torchvision import models
from sklearn.metrics import accuracy_score

from preparacao_pixels import carregar

SEMENTE = 42
EPOCAS = 20
TAXA = 0.0001
TAMANHO = 224


def criar_resnet(num_classes):
    rede = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    rede.fc = nn.Linear(512, num_classes)
    return rede


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
    print(f"Dispositivo: {dispositivo}")
    treino, teste, classes = carregar(tamanho=TAMANHO)
    print(f"Classes: {list(classes)}\n")

    rede = criar_resnet(len(classes)).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(rede.parameters(), lr=TAXA, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="max", factor=0.5, patience=3)

    melhor_acuracia = 0.0
    paciencia_maxima = 6
    epocas_sem_melhorar = 0

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

        perda_media = perda_total / len(treino)
        acuracia_atual = avaliar(rede, teste, dispositivo)
        lr_atual = otimizador.param_groups[0]["lr"]
        scheduler.step(acuracia_atual)

        print(f"epoca {epoca:2d}  perda {perda_media:.4f}  acurácia {acuracia_atual:.4f}  lr {lr_atual:.6f}", end="")

        if acuracia_atual > melhor_acuracia:
            melhor_acuracia = acuracia_atual
            epocas_sem_melhorar = 0
            torch.save({"tipo": "resnet", "estado_da_rede": rede.state_dict(), "classes": classes, "tamanho_imagem": TAMANHO}, "modelo_resnet.pth")
            print("  -> Melhor modelo salvo!")
        else:
            epocas_sem_melhorar += 1
            print(f"  (sem melhoria há {epocas_sem_melhorar} épocas)")
            if epocas_sem_melhorar >= paciencia_maxima:
                print(f"\n[!] Early stopping na época {epoca}.")
                break

    print(f"\nMelhor acurácia: {melhor_acuracia:.4f}")


if __name__ == "__main__":
    treinar()
