from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preparacao import preparar, SEMENTE


def treinar():
    treino_x, teste_x, treino_y, teste_y, colunas = preparar()
    floresta = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEMENTE)
    floresta.fit(treino_x, treino_y)
    previsoes = floresta.predict(teste_x)
    print("acuracia:", round(accuracy_score(teste_y, previsoes), 4))
    print(classification_report(teste_y, previsoes, target_names=["iniciante", "intermediario", "avancado"]))
    return floresta, colunas


if __name__ == "__main__":
    treinar()
