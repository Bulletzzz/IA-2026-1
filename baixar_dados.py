from pathlib import Path
import subprocess

DESTINO = Path(__file__).resolve().parent / "dados"
DATASETS = {
    "tabular": "valakhorasani/gym-members-exercise-dataset",
    "imagens": "hasyimabdillah/workoutexercises-images",
}


def baixar(nome, identificador):
    pasta = DESTINO / nome
    pasta.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", identificador, "-p", str(pasta), "--unzip"],
        check=True,
    )


def main():
    for nome, identificador in DATASETS.items():
        baixar(nome, identificador)


if __name__ == "__main__":
    main()
