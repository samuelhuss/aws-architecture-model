import os
import importlib
from diagrams import Diagram
from diagrams.aws import *
from PIL import Image
import inspect

# Lista dos submódulos AWS a serem processados
aws_modules = [
    "analytics", "ar", "blockchain", "business", "compute", "cost", "database", "devtools", "enablement", "enduser", "engagement", "game", "general", "integration", "iot", "management", "media", "migration", "ml", "mobile", "network", "quantum", "robotics", "satellite", "security", "storage"
]

# Pasta onde as imagens serão salvas
output_dir = os.path.join("dataset", "aws")
os.makedirs(output_dir, exist_ok=True)

# Lista para armazenar os nomes das classes
labels = []
# Lista para armazenar os nomes dos componentes que deram erro
erros = []

# Tamanho da imagem de saída
IMG_SIZE = (256, 256)

for module in aws_modules:
    try:
        # Importa dinamicamente o submódulo
        mod = importlib.import_module(f"diagrams.aws.{module}")
        # Percorre todos os membros do módulo
        for name, obj in inspect.getmembers(mod):
            # Verifica se é uma classe de componente real (não começa com '_') e herda de Node
            if inspect.isclass(obj) and hasattr(obj, "_load_icon") and not name.startswith("_"):
                try:
                    img_name = f"{name}.png"
                    img_path = os.path.join(output_dir, img_name)
                    # Se a imagem já existe, pula para o próximo
                    if os.path.exists(img_path):
                        print(f"Já existe: {img_name}, pulando...")
                        labels.append(name)
                        continue
                    # Cria um contexto de diagrama temporário para evitar erro de contexto global
                    with Diagram("Temp", show=False, outformat="png"):
                        node = obj("Exemplo")
                        icon_path = node._load_icon()
                    # Abre a imagem
                    img = Image.open(icon_path).convert("RGBA")
                    # Redimensiona para IMG_SIZE
                    img = img.resize(IMG_SIZE, Image.LANCZOS)
                    # Salva a imagem com o nome da classe
                    img.save(img_path)
                    labels.append(name)
                    print(f"Salvo: {img_name}")
                except Exception as e:
                    print(f"Erro ao processar {name}: {e}")
                    erros.append(name)
    except Exception as e:
        print(f"Erro ao importar diagrams.aws.{module}: {e}")

# Salva o arquivo de erros
with open(os.path.join(output_dir, "erros.txt"), "w", encoding="utf-8") as f:
    for erro in erros:
        f.write(erro + "\n")

# Gera o labels.txt a partir dos arquivos PNG existentes
labels_from_files = [
    os.path.splitext(f)[0]
    for f in os.listdir(output_dir)
    if f.endswith('.png')
]
labels_from_files.sort()
with open(os.path.join(output_dir, "labels.txt"), "w", encoding="utf-8") as f:
    for label in labels_from_files:
        f.write(label + "\n")
print(f"Arquivo de labels atualizado com base nas imagens salvas.")

print(f"\nTotal de componentes salvos: {len(labels_from_files)}")
print(f"Imagens salvas em: {output_dir}")
print(f"Arquivo de labels: {os.path.join(output_dir, 'labels.txt')}")
print(f"Arquivo de erros: {os.path.join(output_dir, 'erros.txt')}") 