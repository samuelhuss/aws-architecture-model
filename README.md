# Plataforma de Reconhecimento de Componentes de Arquitetura em Diagramas (AWS)

Este projeto tem como objetivo criar um pipeline completo para reconhecimento automático de componentes de arquitetura (AWS) em diagramas, utilizando visão computacional e modelos de detecção de objetos (YOLO).

## Sumário
- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Geração dos Ícones dos Componentes](#geração-dos-ícones-dos-componentes)
- [Geração do Dataset Sintético para YOLO](#geração-do-dataset-sintético-para-yolo)
- [Treinamento do Modelo YOLO](#treinamento-do-modelo-yolo)
- [Validação e Inferência](#validação-e-inferência)
- [Dicas e Próximos Passos](#dicas-e-próximos-passos)
- [Aprendizados e Experimentos](#aprendizados-e-experimentos)

---

## Visão Geral

A plataforma automatiza o processo de:
1. **Coletar ícones de componentes AWS** usando a biblioteca diagrams.
2. **Gerar imagens sintéticas** de diagramas com múltiplos componentes.
3. **Criar anotações no formato YOLO** para detecção de objetos.
4. **Treinar um modelo YOLO** para reconhecer e localizar componentes em diagramas.
5. **Preparar o pipeline para uso prático e expansão para outras clouds (GCP, Azure, etc).**

---

## Estrutura do Projeto

```
├── dataset/
│   └── aws/
│       ├── <classe>.png         # Ícones individuais dos componentes AWS
│       └── labels.txt           # Lista de classes
├── yolo_dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── generate_aws_components_dataset.py   # Script para gerar ícones
├── generate_yolo_synthetic_dataset.py   # Script para gerar dataset sintético para YOLO
├── data.yaml                            # Configuração do dataset para YOLO
└── README.md
```

---

## Geração dos Ícones dos Componentes

1. **Objetivo:** Baixar e salvar os ícones de todos os componentes AWS disponíveis na biblioteca diagrams.
2. **Como executar:**
   - Ative o ambiente virtual:
     ```bash
     .venv\Scripts\Activate  # Windows
     # ou
     source .venv/bin/activate  # Linux/Mac
     ```
   - Instale as dependências:
     ```bash
     pip install diagrams pillow
     ```
   - Execute o script:
     ```bash
     python generate_aws_components_dataset.py
     ```
   - Os ícones serão salvos em `dataset/aws/` e a lista de classes em `labels.txt`.

---

## Geração do Dataset Sintético para YOLO

1. **Objetivo:** Criar imagens compostas (diagramas sintéticos) com múltiplos componentes e gerar as anotações no formato YOLO.
2. **Como executar:**
   - Instale as dependências:
     ```bash
     pip install pillow scikit-learn numpy
     ```
   - Execute o script:
     ```bash
     python generate_yolo_synthetic_dataset.py
     ```
   - O dataset será criado em `yolo_dataset/` com a estrutura esperada pelo YOLO.

---

## Treinamento do Modelo YOLO

1. **Objetivo:** Treinar um modelo YOLO (ex: YOLOv8) para detectar e classificar componentes em diagramas.
2. **Como executar:**
   - Instale o Ultralytics YOLO:
     ```bash
     pip install ultralytics
     ```
   - Certifique-se de que o arquivo `data.yaml` está correto e sincronizado com as classes.
   - Treine o modelo:
     ```bash
     yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
     ```
   - Os resultados e pesos serão salvos em `runs/detect/`.

---

## Validação e Inferência

- **Validação:**
  ```bash
  yolo detect val model=runs/detect/treinamento/weights/best.pt data=data.yaml
  ```
- **Inferência em imagens reais:**
  ```bash
  yolo detect predict model=runs/detect/treinamento/weights/best.pt source=CAMINHO/DA/IMAGEM_OU_PASTA
  ```
- As imagens com as detecções serão salvas em `runs/detect/val` ou `runs/detect/predict`.

---

## Dicas e Próximos Passos

- **Aumente o dataset sintético**: Gere milhares de imagens, varie tamanho, rotação, fundo, sobreposição.
- **Inclua imagens reais**: Ajuda o modelo a generalizar para casos do mundo real.
- **Reduza o número de classes** para experimentos rápidos.
- **Visualize as predições** para garantir que as labels estão corretas.
- **Expanda para GCP/Azure**: Basta adaptar os scripts para os componentes dessas clouds.
- **Implemente um pipeline de inferência**: Automatize o upload, detecção e geração de relatórios (ex: STRIDE).

---

## Aprendizados e Experimentos

Durante o desenvolvimento, realizei dois experimentos principais:

1. **Dataset grande, pouca variação:**
   - Treinei o YOLO com centenas de classes e imagens sintéticas com pouca variação (fundo branco, tamanhos e posições fixas, sem rotação ou ruído).
   - **Resultado:** O modelo não conseguiu aprender bem, apresentando mAP, precisão e recall muito baixos, mesmo após várias épocas. As predições não batiam com as labels, nem mesmo nas imagens sintéticas de validação.

2. **Dataset pequeno, com variação:**
   - Reduzi para 20 classes aleatórias e gerei 200 imagens sintéticas, aplicando variações de rotação, escala, cor de fundo e pequenas mudanças de cor nos ícones.
   - **Resultado:** O modelo aprendeu muito melhor, atingindo mAP50 de 86% e mAP50-95 de quase 80%, com precisão e recall acima de 70%. As predições passaram a bater com as labels nas imagens de validação.

**Conclusão:**
- A variação nas imagens sintéticas (augmentation) é fundamental para o modelo aprender.
- É melhor começar com poucas classes e garantir que o modelo aprende, para depois escalar.
- O balanceamento entre número de classes e exemplos por classe é essencial para bons resultados.

---

## Contato
Dúvidas ou sugestões? Fique à vontade para abrir uma issue ou contribuir! 