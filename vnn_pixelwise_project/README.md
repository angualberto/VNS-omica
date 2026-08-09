# VNN Pixel-wise Mamografia

Projeto experimental em PyTorch para treinar uma VNN pixel-wise em patches de mamografia.

A entrada do modelo tem shape:

```text
[B, 8, 256, 256]
```

A saída tem shape:

```text
[B, 1, 256, 256]
```

A saída é um mapa de probabilidade pixel-wise. Este projeto é pesquisa experimental e **não substitui anotação médica real nem diagnóstico médico**.

## Canais de entrada

1. imagem em escala de cinza
2. magnitude do gradiente
3. Sobel X
4. Sobel Y
5. entropia local
6. FFT baixa frequência local
7. FFT alta frequência local
8. pseudocor/família cromática simplificada tipo AlizaMS/RainbowB

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Teste rápido do modelo

```bash
python vnn_pixelwise_mammo.py
```

Em uma RTX 3060 12GB, a configuração segura é:

```text
PATCH_SIZE=256
BATCH_SIZE=2 ou 4
IN_CH=8
AMP=True
```

## Leitura DICOM

Para imagens `.dcm`, o projeto usa `pydicom`, corrige `MONOCHROME1`, aplica `WindowCenter/WindowWidth` quando disponível e normaliza para `[0,1]`. Os codecs `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg` e `gdcm` estão listados para suportar DICOM comprimido.

## CSV esperado

O CSV deve conter pelo menos:

```text
image_path,cancer
/path/para/imagem1.dcm,0
/path/para/imagem2.dcm,1
```

Opcionalmente pode conter:

```text
mask_path
```

Se `mask_path` existir, a máscara real será usada. Se não existir, o dataset cria uma pseudo-máscara fraca.

## Pseudo-máscara

Quando não há máscara real:

- `cancer=0`: máscara zerada
- `cancer=1`: top regiões por gradiente + entropia

Essa pseudo-máscara é apenas um alvo fraco experimental. Ela não substitui bounding box, segmentação médica ou anotação radiológica.

## Treino

```bash
python train_vnn_pixelwise.py \
  --csv /caminho/para/dataset.csv \
  --out-dir ./saida_vnn_pixelwise \
  --patch-size 256 \
  --stride 128 \
  --batch-size 4 \
  --epochs 10 \
  --amp \
  --cache-images
```

Se der erro de memória, reduza:

```bash
--batch-size 2
```

O treino salva:

- `best_vnn_pixelwise.pt`
- `last_vnn_pixelwise.pt`
- `history.csv`
- `metrics_final.json`
- CSVs de predição por patch

## Predição

```bash
python predict_vnn_pixelwise.py \
  --image /caminho/imagem.dcm \
  --model ./saida_vnn_pixelwise/best_vnn_pixelwise.pt \
  --out-dir ./predicao
```

Saídas:

- `prob_map.png`
- `overlay.png`
- `painel_predicao.png`

## Testes

```bash
pytest -q
```

Os testes verificam:

- shape em CPU
- shape em GPU se CUDA existir
- backward
- entrada `[4,8,256,256]` e saída `[4,1,256,256]`

## Observação médica

Este código é para pesquisa computacional. Mamografias são exames médicos e qualquer interpretação clínica exige validação, dados anotados por especialistas e avaliação regulatória adequada.

Para conjuntos pequenos ou maquinas com RAM disponivel, `--cache-images` evita decodificar o mesmo DICOM novamente a cada patch e acelera muito o treino. Para bases completas, monitore o consumo de RAM.

## Pipeline rapido com cache offline

Para evitar que a GPU espere leitura DICOM, FFT e entropia em cada epoch, use o pipeline em duas etapas.

### 1. Precomputar patches `.pt`

```bash
python 1_precompute_features.py \
  --csv /caminho/dataset.csv \
  --out-dir ./cache_patches \
  --patch-size 128 \
  --stride 128 \
  --max-patches-per-image 16
```

Esse passo e o unico que abre DICOM e calcula os 8 canais. Imagens DICOM invalidas sao puladas e registradas em `dicom_read_errors.csv`.

### 2. Treinar somente lendo tensores cacheados

```bash
python 3_train_vnn_cached.py \
  --manifest ./cache_patches/manifest.csv \
  --out-dir ./saida_cached \
  --batch-size 8 \
  --epochs 10 \
  --num-workers 4
```

Sem mascara real, o mapa pixel-wise e agregado por `top-k mean` e treinado com o rotulo global da imagem (supervisao fraca/MIL). O modelo nao e uma segmentacao medica validada.

### 3. Medir velocidade

```bash
python 4_test_speed.py \
  --manifest ./cache_patches/manifest.csv \
  --batch-size 8 \
  --num-workers 4
```

O Dataset cacheado (`2_dataset_cached.py`) apenas carrega `.pt`; nao importa nem executa `pydicom`, Sobel, FFT ou entropia.

## Pipeline sharded VNN-MIL para base RSNA completa

O cache por patch acima e adequado para smoke tests, mas cria muitos arquivos pequenos. Para a
base completa, use shards: cada DICOM e aberto uma unica vez durante o preprocessamento offline,
e o treino abre poucos arquivos grandes contendo bolsas MIL.

### 1. Preprocessamento offline em shards

```bash
python precompute_shards.py \
  --csv /media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/rsna_kaggle_oficial/train.csv \
  --images-dir /media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/rsna_kaggle_oficial/train_images \
  --out-dir ../saida_vnn_mil_sharded/cache_128 \
  --patch-size 128 \
  --top-k-patches 8 \
  --shard-size 1024 \
  --workers 4
```

O script:

- faz o split por `patient_id` antes de abrir imagens e aborta se detectar vazamento;
- le somente DICOM, corrige `MONOCHROME1`, aplica windowing e remove fundo preto;
- calcula offline os oito canais (`gray`, gradiente, Sobel, entropia, FFT e pseudocor);
- salva `shards/shard_XXXX.pt`, `manifest.csv`, `dicom_read_errors.csv` e `preprocess_params.json`;
- preserva `view`, `density` e `difficult_negative_case`;
- usa apenas o rotulo global `cancer`; atencao MIL nao e mascara de segmentacao.

Por padrao os shards armazenam features em `float16` para reduzir I/O e RAM. Use
`--stored-dtype float32` quando a precisao de armazenamento for prioridade. O valor
`--shard-size 1024` ocupa aproximadamente 256 MiB por shard em `128x128`, evitando que
workers mantenham shards de 1 GiB simultaneamente na RAM.
O preprocessamento usa ate `--workers 4` processos com apenas poucos resultados em voo;
reduza para `2` se a RAM ficar pressionada durante a leitura dos DICOM grandes.

### 2. Treino CUDA com attention MIL

```bash
python train_mil_vnn_sharded.py \
  --manifest ../saida_vnn_mil_sharded/cache_128/manifest.csv \
  --out-dir ../saida_vnn_mil_sharded/treino \
  --epochs 10 \
  --num-workers 2 \
  --prefetch-factor 4
```

O `batch-size` padrao e autoajustado testando `16,32,64` na GPU. Para forcar um valor:

```bash
python train_mil_vnn_sharded.py \
  --manifest ../saida_vnn_mil_sharded/cache_128/manifest.csv \
  --out-dir ../saida_vnn_mil_sharded/treino_b32 \
  --batch-size 32 --epochs 10 --num-workers 4
```

O treino usa CUDA, AMP, `cudnn.benchmark`, memoria pinada, workers persistentes,
`prefetch_factor=4` e leitura ordenada por shard. As saidas incluem checkpoints, historico
de throughput e VRAM, metricas globais/por `view`/por `density`/casos dificeis e
`top_attention_patches.csv`.

### 3. Benchmark do carregamento

```bash
python benchmark_pipeline.py \
  --sharded-manifest ../saida_vnn_mil_sharded/cache_128/manifest.csv \
  --old-manifest ../sistema_integrado_rsna_csv_treino/vnn_cache_full_256/manifest.csv \
  --batch-size 32 --num-workers 2 --batches 50 \
  --out-json ../saida_vnn_mil_sharded/benchmark.json
```

`--old-manifest` e opcional. O resultado informa segundos por batch, patches/s, pico de
VRAM e utilizacao media da GPU consultada durante o benchmark.

## Comparacao CNN, VNN e CNN+VNN com MIL

Os shards criados acima tambem alimentam uma comparacao controlada entre os encoders. Nenhum
dos scripts abaixo abre DICOM ou recalcula features. O `train.csv` e usado apenas para restaurar
metadados como `laterality` que nao estavam no primeiro manifesto sharded.

### Modelos

- `cnn_encoder.py`: CNN leve com embedding de 128 dimensoes por patch.
- `vnn_encoder.py`: Volterra low-rank com termo linear `1x1` e termo quadratico `U(x)*V(x)`.
- `hybrid_cnn_vnn_mil.py`: modos `cnn`, `vnn` e `hybrid`, com attention MIL por imagem.

### Treinar os tres modelos

Execute depois que qualquer treino usando a mesma GPU/disco for encerrado:

```bash
python train_hybrid_mil.py \
  --manifest ../saida_vnn_mil_sharded/cache_128/manifest.csv \
  --metadata-csv /media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/rsna_kaggle_oficial/train.csv \
  --out-dir ../saida_vnn_mil_sharded/comparacao_hibrida \
  --mode all --epochs 10 --batch-size 64 --num-workers 4
```

Para testar `batch_size=128` na RTX 3060, rode um modelo isolado primeiro:

```bash
python train_hybrid_mil.py \
  --manifest ../saida_vnn_mil_sharded/cache_128/manifest.csv \
  --metadata-csv /media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/rsna_kaggle_oficial/train.csv \
  --out-dir ../saida_vnn_mil_sharded/comparacao_hibrida_b128 \
  --mode hybrid --epochs 1 --batch-size 128 --num-workers 4
```

Cada modo salva `metrics_epoch.csv`, `best_model.pt`, `predictions.csv`,
`threshold_analysis.csv`, `roc.png`, `pr.png`, `metrics.json` e
`top_attention_patches.csv`. O threshold com melhor F1 e uma analise exploratoria sobre o
conjunto de teste; para estimativa final sem vies, fixe o threshold em um conjunto de validacao
independente antes de avaliar em teste.

### Comparar resultados

```bash
python compare_models.py \
  --results-dir ../saida_vnn_mil_sharded/comparacao_hibrida
```

### Visualizar attention MIL

```bash
python interpret_attention.py \
  --manifest ../saida_vnn_mil_sharded/cache_128/manifest.csv \
  --metadata-csv /media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/rsna_kaggle_oficial/train.csv \
  --checkpoint ../saida_vnn_mil_sharded/comparacao_hibrida/hybrid/best_model.pt \
  --predictions ../saida_vnn_mil_sharded/comparacao_hibrida/hybrid/predictions.csv \
  --out-dir ../saida_vnn_mil_sharded/comparacao_hibrida/attention_hybrid
```

## Cores AlizaMS em DICOM com Fortran/OpenMP

`analisar_alizams_cores_dicom_csv.py` aplica a LUT `black_rainbow_lut` usada na analise
AlizaMS/RainbowB aos DICOM indicados por `train.csv`. O backend hibrido mantem leitura,
windowing e graficos em Python e desloca a acumulacao cromatica para Fortran com OpenMP.

```bash
bash build_alizams_fortran_openmp.sh
OMP_NUM_THREADS=4 python analisar_alizams_cores_dicom_csv.py \
  --backend fortran_openmp \
  --workers 4 \
  --out-dir ../saida_analise_cores_alizams_dicom_rsna_fortran
```

Nesta maquina existe `gfortran` e `nvcc`, mas nao existe `nvfortran`. CUDA Fortran requer
NVIDIA HPC SDK (`nvfortran`); `nvcc` compila CUDA C/C++, nao codigo Fortran. A versao
OpenMP e executavel imediatamente e mantem a comparacao entre cancer e sem cancer.

As figuras de attention usam somente os patches cacheados e devem ser interpretadas como
evidencia exploratoria do modelo, nao como segmentacao ou localizacao clinica validada.
