from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CAMINHO = Path(__file__).resolve().parent.parent / "dados" / "tabular" / "gym_members_exercise_tracking.csv"
SEMENTE = 42


def carregar():
    dados = pd.read_csv(CAMINHO)
    return dados.drop_duplicates().reset_index(drop=True)


def separar_alvo(dados):
    alvo = dados["Experience_Level"] - 1
    atributos = dados.drop(columns=["Experience_Level"])
    atributos = pd.get_dummies(atributos, columns=["Gender", "Workout_Type"], drop_first=True)
    return atributos, alvo


def preparar():
    atributos, alvo = separar_alvo(carregar())
    treino_x, teste_x, treino_y, teste_y = train_test_split(
        atributos, alvo, test_size=0.2, stratify=alvo, random_state=SEMENTE
    )
    escalador = StandardScaler()
    treino_x = escalador.fit_transform(treino_x)
    teste_x = escalador.transform(teste_x)
    return treino_x, teste_x, treino_y.to_numpy(), teste_y.to_numpy(), list(atributos.columns)
