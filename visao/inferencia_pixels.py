import sys
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import cv2
from pathlib import Path
from collections import Counter

from preparacao_pixels import MEDIA, DESVIO
from cnn_pixels import CnnDoZero
from resnet_finetune import criar_resnet

PASTA_VIDEO = Path(__file__).resolve().parent.parent / "video_inferencia"
FRAMES_EXTRAIR = 10
TAMANHO = 224

transform = transforms.Compose([
    transforms.Resize(int(TAMANHO * 1.15)),
    transforms.CenterCrop(TAMANHO),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIO),
])


def carregar_modelo(caminho_pth, dispositivo):
    checkpoint = torch.load(caminho_pth, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    tipo = checkpoint.get("tipo", "resnet" if "resnet" in caminho_pth.stem else "cnn")

    if tipo == "cnn":
        rede = CnnDoZero(len(classes))
    else:
        rede = criar_resnet(len(classes))

    rede.load_state_dict(checkpoint["estado_da_rede"])
    rede.to(dispositivo).eval()
    return rede, classes


def extrair_frames(caminho_video):
    cap = cv2.VideoCapture(str(caminho_video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / FRAMES_EXTRAIR) for i in range(FRAMES_EXTRAIR)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(transform(img))
    cap.release()
    return frames


def classificar_video(caminho_video, rede, classes, dispositivo):
    frames = extrair_frames(caminho_video)
    if not frames:
        print(f"Erro: nenhum frame extraído de {caminho_video.name}")
        return

    batch = torch.stack(frames).to(dispositivo)
    with torch.no_grad():
        saidas = rede(batch).argmax(dim=1).cpu().tolist()

    contagem = Counter(saidas)
    classe_final = contagem.most_common(1)[0][0]

    print(f"\nVídeo: {caminho_video.name}")
    print("Votos por classe:")
    for idx, votos in sorted(contagem.items(), key=lambda x: -x[1]):
        print(f"  {classes[idx]:<20} {votos} voto(s)")
    print(f"\n>>> Resultado: {classes[classe_final].upper()} <<<\n")


def main():
    if len(sys.argv) < 2:
        print("Uso: python visao/inferencia_pixels.py <modelo.pth>")
        print("Exemplos:")
        print("  python visao/inferencia_pixels.py modelo_cnn_pixels.pth")
        print("  python visao/inferencia_pixels.py modelo_resnet.pth")
        return

    modelo_path = Path(sys.argv[1])
    if not modelo_path.exists():
        print(f"Modelo não encontrado: {modelo_path}")
        return

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rede, classes = carregar_modelo(modelo_path, dispositivo)

    print(f"Modelo: {modelo_path.name}")
    print(f"Classes: {list(classes)}")
    print(f"Lendo vídeos de: {PASTA_VIDEO}\n")

    extensoes = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [v for v in PASTA_VIDEO.iterdir() if v.suffix.lower() in extensoes]

    if not videos:
        print("Nenhum vídeo encontrado em 'video_inferencia/'.")
        return

    for video in sorted(videos):
        classificar_video(video, rede, classes, dispositivo)


if __name__ == "__main__":
    main()
