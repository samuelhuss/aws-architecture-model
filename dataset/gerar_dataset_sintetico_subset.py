import os
import random
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import shutil

# Parâmetros
NUM_COMPONENTES = 25
NUM_IMAGENS = 200
IMG_SIZE = 640
COMP_MIN = 3
COMP_MAX = 7
TRAIN_SPLIT = 160
VAL_SPLIT = 40

# Caminhos
base_dir = os.path.dirname(__file__)
aws_dir = os.path.join(base_dir, 'aws')
dataset_dir = os.path.join(base_dir, '../synthetic_dataset_subset')
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
train_img_dir = os.path.join(dataset_dir, 'images/train')
val_img_dir = os.path.join(dataset_dir, 'images/val')
train_label_dir = os.path.join(dataset_dir, 'labels/train')
val_label_dir = os.path.join(dataset_dir, 'labels/val')

for d in [images_dir, labels_dir, train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(d, exist_ok=True)

# Seleciona 25 componentes aleatórios
todos_componentes = [f for f in os.listdir(aws_dir) if f.endswith('.png')]
componentes_escolhidos = random.sample(todos_componentes, NUM_COMPONENTES)
classes = [os.path.splitext(f)[0] for f in componentes_escolhidos]

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
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.6, 1.4))
    arr = np.array(img.convert('HSV'))
    arr[..., 0] = (arr[..., 0].astype(int) + random.randint(-20, 20)) % 180
    img = Image.fromarray(arr, mode='HSV').convert('RGB')
    return img

def checa_sobreposicao(box, boxes):
    for b in boxes:
        if not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]):
            return True
    return False

# Geração das imagens sintéticas
imagens_geradas = []
labels_gerados = []
for idx_img in range(NUM_IMAGENS):
    fundo = gerar_fundo()
    n_comp = random.randint(COMP_MIN, COMP_MAX)
    comps_escolhidos = random.sample(componentes_escolhidos, n_comp)
    boxes = []
    labels = []
    for comp in comps_escolhidos:
        img_comp = Image.open(os.path.join(aws_dir, comp))
        # Garante que a imagem tem canal alfa (RGBA)
        if img_comp.mode != 'RGBA':
            img_comp = img_comp.convert('RGBA')
        # Cria uma máscara de alfa explícita (L mode)
        alpha_mask = img_comp.split()[-1]
        if alpha_mask.getextrema() == (255, 255) or alpha_mask.getextrema() == (0, 0):
            # Se a máscara for toda opaca ou transparente, força opaca
            alpha_mask = Image.new('L', img_comp.size, 255)
        img_comp = augmentar_cor(img_comp)
        escala = random.uniform(0.7, 1.3)
        w, h = img_comp.size
        new_w = int(w * escala)
        new_h = int(h * escala)
        img_comp = img_comp.resize((new_w, new_h), Image.LANCZOS)
        alpha_mask = alpha_mask.resize((new_w, new_h), Image.LANCZOS)
        for _ in range(30):
            x1 = random.randint(0, IMG_SIZE - new_w)
            y1 = random.randint(0, IMG_SIZE - new_h)
            x2 = x1 + new_w
            y2 = y1 + new_h
            if not checa_sobreposicao([x1, y1, x2, y2], boxes):
                # Cola usando a máscara de alfa explícita
                fundo.paste(img_comp, (x1, y1), alpha_mask)
                boxes.append([x1, y1, x2, y2])
                classe_idx = classes.index(os.path.splitext(comp)[0])
                x_c = (x1 + x2) / 2 / IMG_SIZE
                y_c = (y1 + y2) / 2 / IMG_SIZE
                w_norm = (x2 - x1) / IMG_SIZE
                h_norm = (y2 - y1) / IMG_SIZE
                labels.append(f"{classe_idx} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
                break
    nome_img = f'subset_{idx_img:04d}.jpg'
    fundo.save(os.path.join(images_dir, nome_img), quality=95)
    with open(os.path.join(labels_dir, nome_img.replace('.jpg', '.txt')), 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels) + '\n')
    imagens_geradas.append(nome_img)
    labels_gerados.append(labels)
    if (idx_img+1) % 50 == 0:
        print(f'{idx_img+1} imagens geradas...')

# Split garantindo todas as classes em ambos os conjuntos
random.shuffle(imagens_geradas)
classe_em_treino = set()
classe_em_val = set()
train_imgs = []
val_imgs = []
for img in imagens_geradas:
    label_path = os.path.join(labels_dir, img.replace('.jpg', '.txt'))
    with open(label_path, 'r', encoding='utf-8') as f:
        classes_na_img = set(int(l.split()[0]) for l in f if l.strip())
    # Prioriza adicionar ao treino até todas as classes estarem lá
    if len(train_imgs) < TRAIN_SPLIT and not classe_em_treino.issuperset(classes_na_img):
        train_imgs.append(img)
        classe_em_treino.update(classes_na_img)
    elif len(val_imgs) < VAL_SPLIT and not classe_em_val.issuperset(classes_na_img):
        val_imgs.append(img)
        classe_em_val.update(classes_na_img)
    elif len(train_imgs) < TRAIN_SPLIT:
        train_imgs.append(img)
    elif len(val_imgs) < VAL_SPLIT:
        val_imgs.append(img)
    if len(train_imgs) >= TRAIN_SPLIT and len(val_imgs) >= VAL_SPLIT:
        break

# Copia imagens e labels para as pastas finais
def copiar_imgs_labels(imgs, img_dir, label_dir):
    for img in imgs:
        shutil.copy2(os.path.join(images_dir, img), os.path.join(img_dir, img))
        shutil.copy2(os.path.join(labels_dir, img.replace('.jpg', '.txt')), os.path.join(label_dir, img.replace('.jpg', '.txt')))

copiar_imgs_labels(train_imgs, train_img_dir, train_label_dir)
copiar_imgs_labels(val_imgs, val_img_dir, val_label_dir)

# Gera o data.yaml
yaml_content = f"""
train: images/train
val: images/val

nc: {NUM_COMPONENTES}
names:
""" + ''.join([f"  - {c}\n" for c in classes])
with open(os.path.join(dataset_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print('Dataset sintético subset criado com sucesso!') 