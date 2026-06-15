import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import urllib.request
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets
from PIL import Image
from collections import defaultdict
import random
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

CAMINHO = Path(__file__).resolve().parent.parent / "dados" / "imagens"
CACHE = Path(__file__).resolve().parent.parent / "dados" / "cache_pose.pt"
MODELO_POSE = Path(__file__).resolve().parent.parent / "dados" / "pose_landmarker_lite.task"
SEMENTE = 42

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def garantir_modelo_pose():
    if not MODELO_POSE.exists():
        print("Baixando modelo de pose MediaPipe (~5MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODELO_POSE)
        print("Modelo baixado.")


def criar_detector():
    garantir_modelo_pose()
    base_options = mp_python.BaseOptions(model_asset_path=str(MODELO_POSE))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extrair_pose(img_pil, detector):
    img_rgb = np.array(img_pil.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    resultado = detector.detect(mp_image)
    if not resultado.pose_landmarks:
        return None
    lm = resultado.pose_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    quadril = coords[[23, 24]].mean(axis=0)
    coords -= quadril
    dist = np.linalg.norm(coords[11] - coords[12]) + 1e-6
    coords /= dist
    return coords.flatten()


def construir_cache(teto=500):
    base = datasets.ImageFolder(CAMINHO)
    indices_por_classe = defaultdict(list)
    for idx, (_, rotulo) in enumerate(base.samples):
        indices_por_classe[rotulo].append(idx)

    rng = random.Random(SEMENTE)
    indices_selecionados = []
    for rotulo in sorted(indices_por_classe):
        indices = indices_por_classe[rotulo]
        escolhidos = rng.sample(indices, min(teto, len(indices)))
        indices_selecionados.extend(escolhidos)

    print(f"Extraindo pose de {len(indices_selecionados)} imagens (pode levar alguns minutos)...")
    X, y = [], []
    sem_pose = 0

    detector = criar_detector()
    try:
        for i, idx in enumerate(indices_selecionados):
            if i % 200 == 0:
                print(f"  {i}/{len(indices_selecionados)}")
            path, rotulo = base.samples[idx]
            vetor = extrair_pose(Image.open(path), detector)
            if vetor is not None:
                X.append(vetor)
                y.append(rotulo)
            else:
                sem_pose += 1
    finally:
        detector.close()

    print(f"Com pose: {len(X)} | Sem pose: {sem_pose}")
    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    torch.save({"X": X_t, "y": y_t, "classes": base.classes}, CACHE)
    print(f"Cache salvo em: {CACHE}")
    return X_t, y_t, base.classes


def carregar(lote=64):
    if CACHE.exists():
        print("Carregando cache de poses...")
        dados = torch.load(CACHE, weights_only=False)
        X, y, classes = dados["X"], dados["y"], dados["classes"]
    else:
        X, y, classes = construir_cache()

    gerador = torch.Generator().manual_seed(SEMENTE)
    dataset = TensorDataset(X, y)
    n_treino = int(len(dataset) * 0.8)
    n_teste = len(dataset) - n_treino
    treino, teste = random_split(dataset, [n_treino, n_teste], generator=gerador)

    carregador_treino = DataLoader(treino, batch_size=lote, shuffle=True)
    carregador_teste = DataLoader(teste, batch_size=lote, shuffle=False)

    print(f"Treino: {n_treino} | Teste: {n_teste} | Classes: {classes}")
    return carregador_treino, carregador_teste, classes
