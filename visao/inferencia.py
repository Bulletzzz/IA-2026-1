import torch
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from preparacao import extrair_pose, criar_detector
from cnn import RedePostura

PASTA_VIDEO = Path(__file__).resolve().parent.parent / "video_inferencia"
MODELO_PATH = Path(__file__).resolve().parent.parent / "modelo_academia.pth"
FRAMES_EXTRAIR = 10




def extrair_poses_video(caminho_video):
    cap = cv2.VideoCapture(str(caminho_video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / FRAMES_EXTRAIR) for i in range(FRAMES_EXTRAIR)]
    poses = []
    detector = criar_detector()
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                vetor = extrair_pose(img_pil, detector)
                if vetor is not None:
                    poses.append(torch.tensor(vetor, dtype=torch.float32))
    finally:
        detector.close()
    cap.release()
    return poses


def classificar_video(caminho_video, rede, classes, dispositivo):
    poses = extrair_poses_video(caminho_video)
    if not poses:
        print(f"Erro: nenhuma pose detectada em {caminho_video.name}")
        return

    batch = torch.stack(poses).to(dispositivo)
    with torch.no_grad():
        saidas = rede(batch).argmax(dim=1).cpu().tolist()

    contagem = Counter(saidas)
    classe_final = contagem.most_common(1)[0][0]

    print(f"\nVídeo: {caminho_video.name}")
    print(f"Frames com pose detectada: {len(poses)}/{FRAMES_EXTRAIR}")
    print("Votos por classe:")
    for idx, votos in sorted(contagem.items(), key=lambda x: -x[1]):
        print(f"  {classes[idx]:<20} {votos} voto(s)")
    print(f"\n>>> Resultado: {classes[classe_final].upper()} <<<\n")


def main():
    if not MODELO_PATH.exists():
        print("Modelo não encontrado. Treine primeiro com: python visao/cnn.py")
        return

    checkpoint = torch.load(MODELO_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rede = RedePostura(len(classes)).to(dispositivo)
    rede.load_state_dict(checkpoint["estado_da_rede"])
    rede.eval()

    print(f"Modelo carregado | Classes: {list(classes)}")
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
