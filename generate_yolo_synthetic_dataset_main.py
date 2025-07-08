import os
import random
from PIL import Image, ImageEnhance
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Parâmetros do dataset
IMG_SIZE = 640
COMPONENTES_POR_IMAGEM = 6
IMAGENS_POR_COMPONENTE = 20
TRAIN_RATIO = 0.8

# Lista dos principais componentes AWS (ajuste conforme disponibilidade dos ícones)
COMPONENTES_PRINCIPAIS = [
    "EC2", "Lambda", "ECS", "EKS", "ElasticBeanstalk", "S3", "EBS", "EFS", "FSx",
    "RDS", "Dynamodb", "Aurora", "ElastiCache", "Redshift", "VPC", "ELB", "ALB", "NLB",
    "CloudFront", "APIGateway", "DirectConnect", "Route53", "IAM", "Cognito", "KMS",
    "WAF", "Shield", "Cloudwatch", "Cloudtrail", "Config", "SystemsManager", "SNS",
    "SQS", "Eventbridge", "StepFunctions", "Codecommit", "Codebuild", "Codedeploy",
    "Codepipeline", "Cloudformation"
]

# Pastas
ICONS_DIR = os.path.join("dataset", "aws")
OUTPUT_DIR = "yolo_dataset_main"
IMAGES_TRAIN = os.path.join(OUTPUT_DIR, "images", "train")
IMAGES_VAL = os.path.join(OUTPUT_DIR, "images", "val")
LABELS_TRAIN = os.path.join(OUTPUT_DIR, "labels", "train")
LABELS_VAL = os.path.join(OUTPUT_DIR, "labels", "val")

for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    os.makedirs(d, exist_ok=True)

# Filtra apenas os componentes que possuem ícone disponível
componentes_disponiveis = [c for c in COMPONENTES_PRINCIPAIS if os.path.exists(os.path.join(ICONS_DIR, f"{c}.png"))]
class2idx = {c: i for i, c in enumerate(componentes_disponiveis)}
icon_files = {c: os.path.join(ICONS_DIR, f"{c}.png") for c in componentes_disponiveis}

# Calcula o total de imagens
TOTAL_IMAGENS = IMAGENS_POR_COMPONENTE * len(componentes_disponiveis)

# Garante distribuição balanceada dos componentes nas imagens
componente_count = defaultdict(int)
imagens = []

# Gera combinações de componentes para cada imagem
while len(imagens) < TOTAL_IMAGENS:
    # Seleciona componentes menos usados até agora
    disponiveis = sorted(componentes_disponiveis, key=lambda c: componente_count[c])
    selecionados = random.sample(disponiveis[:max(10, len(disponiveis))], COMPONENTES_POR_IMAGEM)
    for c in selecionados:
        componente_count[c] += 1
    imagens.append(selecionados)

# Embaralha as imagens
random.shuffle(imagens)

# Divide em treino e validação
all_indices = list(range(TOTAL_IMAGENS))
train_indices, val_indices = train_test_split(all_indices, train_size=TRAIN_RATIO, random_state=42)

# Função para verificar sobreposição significativa
def tem_sobreposicao(nova_box, boxes, limiar=0.15):
    x1, y1, x2, y2 = nova_box
    for bx1, by1, bx2, by2 in boxes:
        ix1, iy1 = max(x1, bx1), max(y1, by1)
        ix2, iy2 = min(x2, bx2), min(y2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_nova = (x2 - x1) * (y2 - y1)
        area_existente = (bx2 - bx1) * (by2 - by1)
        if inter / min(area_nova, area_existente) > limiar:
            return True
    return False

# Gera as imagens sintéticas
for split, indices, img_dir, lbl_dir in [
    ("train", train_indices, IMAGES_TRAIN, LABELS_TRAIN),
    ("val", val_indices, IMAGES_VAL, LABELS_VAL)
]:
    for idx in indices:
        componentes = imagens[idx]
        # Fundo com cor levemente variada
        bg_color = tuple([random.randint(220, 255) for _ in range(3)] + [255])
        img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), bg_color)
        boxes = []
        labels = []
        for comp in componentes:
            icon = Image.open(icon_files[comp]).convert("RGBA")
            # Variação de escala
            icon_size = random.randint(96, 180)
            icon = icon.resize((icon_size, icon_size), Image.LANCZOS)
            # Rotação aleatória
            angle = random.randint(0, 359)
            icon = icon.rotate(angle, expand=True)
            # Pequena variação de cor
            enhancer = ImageEnhance.Color(icon)
            icon = enhancer.enhance(random.uniform(0.8, 1.2))
            # Tenta posicionar sem sobreposição significativa
            for tentativa in range(30):
                x = random.randint(0, IMG_SIZE - icon.size[0])
                y = random.randint(0, IMG_SIZE - icon.size[1])
                box = (x, y, x + icon.size[0], y + icon.size[1])
                if not tem_sobreposicao(box, boxes):
                    break
            else:
                pass
            img.alpha_composite(icon, (x, y))
            boxes.append(box)
            labels.append(comp)
        # Salva imagem
        img = img.convert("RGB")
        img_name = f"synt_{idx:04d}.jpg"
        img.save(os.path.join(img_dir, img_name), quality=95)
        # Salva anotação YOLO
        label_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for (x1, y1, x2, y2), comp in zip(boxes, labels):
                xc = (x1 + x2) / 2 / IMG_SIZE
                yc = (y1 + y2) / 2 / IMG_SIZE
                w = (x2 - x1) / IMG_SIZE
                h = (y2 - y1) / IMG_SIZE
                f.write(f"{class2idx[comp]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        print(f"{split}: {img_name} gerada com {COMPONENTES_POR_IMAGEM} componentes.")

# Salva as classes usadas
with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
    for c in componentes_disponiveis:
        f.write(c + "\n")

print("\nDataset sintético principal para YOLO gerado com sucesso!")
print(f"Imagens de treino: {IMAGES_TRAIN}")
print(f"Imagens de validação: {IMAGES_VAL}")
print(f"Classes usadas: {os.path.join(OUTPUT_DIR, 'labels.txt')}") 