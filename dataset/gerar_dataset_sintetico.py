import os
import random
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np

# Parâmetros
NUM_IMAGENS = 1000
IMG_SIZE = 640
COMP_MIN = 3
COMP_MAX = 7

# Caminhos
base_dir = os.path.dirname(__file__)
aws_dir = os.path.join(base_dir, 'aws')
dataset_dir = os.path.join(base_dir, '../synthetic_dataset')
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Lista de componentes (nomes dos arquivos PNG)
componentes = [f for f in os.listdir(aws_dir) if f.endswith('.png')]
classes = [os.path.splitext(f)[0] for f in componentes]

# Função para gerar fundo variado
def gerar_fundo():
    fundo = Image.new('RGB', (IMG_SIZE, IMG_SIZE), random.choice(['white', 'lightgray', 'gray']))
    draw = ImageDraw.Draw(fundo)
    tipo = random.choice(['linhas', 'caixas', 'setas', 'nada'])
    if tipo == 'linhas':
        for _ in range(random.randint(3, 8)):
            x1, y1 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
            x2, y2 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
            cor = random.choice(['black', 'gray', 'darkgray'])
            draw.line((x1, y1, x2, y2), fill=cor, width=random.randint(2, 5))
    elif tipo == 'caixas':
        for _ in range(random.randint(2, 5)):
            x1, y1 = random.randint(0, IMG_SIZE-100), random.randint(0, IMG_SIZE-100)
            x2, y2 = x1+random.randint(60, 200), y1+random.randint(40, 150)
            cor = random.choice(['#e0e0e0', '#c0c0c0', '#a0a0a0'])
            draw.rectangle([x1, y1, x2, y2], outline='black', fill=cor)
    elif tipo == 'setas':
        for _ in range(random.randint(2, 5)):
            x1, y1 = random.randint(0, IMG_SIZE-50), random.randint(0, IMG_SIZE-50)
            x2, y2 = x1+random.randint(40, 200), y1+random.randint(20, 100)
            cor = random.choice(['black', 'gray'])
            draw.line((x1, y1, x2, y2), fill=cor, width=4)
            # Desenha a ponta da seta
            if x2 > x1:
                draw.polygon([(x2, y2), (x2-10, y2-7), (x2-10, y2+7)], fill=cor)
            else:
                draw.polygon([(x2, y2), (x2+10, y2-7), (x2+10, y2+7)], fill=cor)
    # tipo 'nada' não faz nada
    return fundo

# Função para aplicar aumentação de cor
def augmentar_cor(img):
    # Brilho
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    # Contraste
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    # Saturação
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.6, 1.4))
    # Hue (convertendo para numpy)
    arr = np.array(img.convert('HSV'))
    arr[..., 0] = (arr[..., 0].astype(int) + random.randint(-20, 20)) % 180
    img = Image.fromarray(arr, mode='HSV').convert('RGB')
    return img

# Função para verificar sobreposição
def checa_sobreposicao(box, boxes):
    for b in boxes:
        # [x1, y1, x2, y2]
        if not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]):
            return True
    return False

# Geração das imagens sintéticas
for idx_img in range(NUM_IMAGENS):
    fundo = gerar_fundo()
    n_comp = random.randint(COMP_MIN, COMP_MAX)
    comps_escolhidos = random.sample(componentes, n_comp)
    boxes = []
    labels = []
    tentativas = 0
    for comp in comps_escolhidos:
        tentativas += 1
        if tentativas > 50:  # Evita loop infinito
            break
        img_comp = Image.open(os.path.join(aws_dir, comp)).convert('RGBA')
        img_comp = augmentar_cor(img_comp)
        # Escala aleatória
        escala = random.uniform(0.7, 1.3)
        w, h = img_comp.size
        new_w = int(w * escala)
        new_h = int(h * escala)
        img_comp = img_comp.resize((new_w, new_h), Image.LANCZOS)
        # Posição aleatória sem sobreposição
        for _ in range(30):
            x1 = random.randint(0, IMG_SIZE - new_w)
            y1 = random.randint(0, IMG_SIZE - new_h)
            x2 = x1 + new_w
            y2 = y1 + new_h
            if not checa_sobreposicao([x1, y1, x2, y2], boxes):
                fundo.paste(img_comp, (x1, y1), img_comp)
                boxes.append([x1, y1, x2, y2])
                classe_idx = classes.index(os.path.splitext(comp)[0])
                # YOLO: classe x_centro y_centro largura altura (normalizado)
                x_c = (x1 + x2) / 2 / IMG_SIZE
                y_c = (y1 + y2) / 2 / IMG_SIZE
                w_norm = (x2 - x1) / IMG_SIZE
                h_norm = (y2 - y1) / IMG_SIZE
                labels.append(f"{classe_idx} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
                break
    # Salva imagem
    nome_img = f'synt_{idx_img:04d}.jpg'
    fundo.save(os.path.join(images_dir, nome_img), quality=95)
    # Salva label
    with open(os.path.join(labels_dir, nome_img.replace('.jpg', '.txt')), 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels) + '\n')
    if (idx_img+1) % 100 == 0:
        print(f'{idx_img+1} imagens geradas...')

print('Dataset sintético gerado com sucesso!') 