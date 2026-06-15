import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from preparacao import carregar, separar_alvo, SEMENTE
from rede_densa import RedeDensa, EPOCAS, TAXA

NIVEIS = ["iniciante", "intermediario", "avancado"]

ALUNOS = {
    "Alec": {
        "Age": 23, "Gender": "Female", "Weight (kg)": 62, "Height (m)": 1.66,
        "Max_BPM": 188, "Avg_BPM": 150, "Resting_BPM": 74,
        "Session_Duration (hours)": 0.5, "Calories_Burned": 280, "Workout_Type": "Cardio",
        "Fat_Percentage": 29, "Water_Intake (liters)": 1.4,
        "Workout_Frequency (days/week)": 2, "BMI": 22.5,
    },
    "Barbosa": {
        "Age": 45, "Gender": "Male", "Weight (kg)": 90, "Height (m)": 1.78,
        "Max_BPM": 182, "Avg_BPM": 160, "Resting_BPM": 55,
        "Session_Duration (hours)": 1.8, "Calories_Burned": 1450, "Workout_Type": "Strength",
        "Fat_Percentage": 13, "Water_Intake (liters)": 3.6,
        "Workout_Frequency (days/week)": 5, "BMI": 28.4,
    },
    "Moaca": {
        "Age": 30, "Gender": "Male", "Weight (kg)": 75, "Height (m)": 1.75,
        "Max_BPM": 185, "Avg_BPM": 152, "Resting_BPM": 65,
        "Session_Duration (hours)": 1.0, "Calories_Burned": 750, "Workout_Type": "HIIT",
        "Fat_Percentage": 20, "Water_Intake (liters)": 2.5,
        "Workout_Frequency (days/week)": 3, "BMI": 24.5,
    },
}


def treinar_rede(matriz, rotulos):
    torch.manual_seed(SEMENTE)
    x = torch.from_numpy(matriz.astype(np.float32))
    y = torch.from_numpy(rotulos.astype(np.int64))
    rede = RedeDensa(x.shape[1], int(y.max().item()) + 1)
    criterio = torch.nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(rede.parameters(), lr=TAXA, weight_decay=1e-4)
    for _ in range(EPOCAS):
        rede.train()
        otimizador.zero_grad()
        perda = criterio(rede(x), y)
        perda.backward()
        otimizador.step()
    rede.eval()
    return rede


def montar_modelos():
    atributos, alvo = separar_alvo(carregar())
    escalador = StandardScaler()
    matriz = escalador.fit_transform(atributos)
    floresta = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEMENTE)
    floresta.fit(matriz, alvo)
    rede = treinar_rede(matriz, alvo.to_numpy())
    return floresta, rede, escalador, list(atributos.columns)


def vetorizar(aluno, escalador, colunas):
    linha = pd.get_dummies(pd.DataFrame([aluno]), columns=["Gender", "Workout_Type"], drop_first=True)
    return escalador.transform(linha.reindex(columns=colunas, fill_value=0))


def prever_floresta(vetor, floresta):
    return NIVEIS[floresta.predict(vetor)[0]]


def prever_rede(vetor, rede):
    with torch.no_grad():
        saida = rede(torch.from_numpy(vetor.astype(np.float32)))
    return NIVEIS[int(saida.argmax(dim=1)[0])]


if __name__ == "__main__":
    floresta, rede, escalador, colunas = montar_modelos()
    for nome, aluno in ALUNOS.items():
        entrada = vetorizar(aluno, escalador, colunas)
        print(f"{nome:8s} random forest: {prever_floresta(entrada, floresta):14s} rede densa: {prever_rede(entrada, rede)}")
