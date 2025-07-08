from ultralytics import YOLO
import os

# Caminho do modelo treinado
modelo_treinado = '../runs/detect/subset_test/weights/best.pt'

# Pasta com as imagens de arquitetura para testar
pasta_arquiteturas = '../arquiteturas_reais/'

# Cria a pasta de saída se não existir
os.makedirs('runs/detect/predict/', exist_ok=True)

# Carrega o modelo YOLO treinado
model = YOLO(modelo_treinado)

# Faz a predição em todas as imagens da pasta
results = model.predict(
    source=pasta_arquiteturas,  # Pasta com imagens
    imgsz=640,                  # Tamanho da imagem (igual ao treino)
    conf=0.25,                  # Confiança mínima para detecção
    save=True                   # Salva as imagens com as detecções desenhadas
)

print('Predição concluída! Veja as imagens em runs/detect/predict/') 

