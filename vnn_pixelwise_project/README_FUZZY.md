# Camada Fuzzy para Tendencia Estatistica de Cancer - RSNA

## Objetivo

Esta camada combina scores MIL, atencao por patch, pseudocores AlizaMS/RainbowB e,
quando disponiveis, features estruturais precomputadas. A saida
`tendencia_cancer` esta no intervalo `[0, 1]` e possui tres faixas:

- `baixa`: valor menor que `0.34`;
- `intermediaria`: valor entre `0.34` e `0.67`;
- `alta`: valor a partir de `0.67`.

A saida e uma tendencia estatistica experimental. Ela nao e diagnostico medico,
nao substitui avaliacao radiologica e nao representa mascara de lesao.

## Arquivos

- `fuzzy_cancer_rules.py`: pertinencias, dez regras Mamdani e centroide.
- `fuzzy_dataset_features.py`: une CSVs precomputados por `patient_id,image_id`.
- `run_fuzzy_inference.py`: gera predicoes, thresholds e metricas estratificadas.
- `compare_fuzzy_vs_models.py`: compara scores de modelos e fuzzy.
- `explain_fuzzy_case.py`: relatorio das regras ativadas para uma imagem.
- `plot_fuzzy_results.py`: graficos de distribuicao, ROC/PR e confusao.
- `fuzzy_patch_selector.py`: reordena e seleciona patches cacheados por regras cromaticas fuzzy.
- `train_mil_vnn_fuzzy_selected.py`: treina VNN e CNN+VNN usando os novos shards e compara com a VNN anterior.

## Entradas

Obrigatorias para VNN fuzzy:

- `predictions.csv` do VNN, CNN ou Hybrid, contendo score e label por imagem;
- `cores_alizams_por_imagem.csv`, gerado previamente a partir dos DICOM.

Recomendadas:

- `top_attention_patches.csv` ou `attention_examples.csv`;
- CSV estrutural precomputado contendo gradiente, entropia, wavelet e FFT.

O arquivo de atencao VNN atual grava apenas os top patches. Nessa situacao,
`attention_entropy` e estimada distribuindo o peso restante entre patches nao
salvos e a coluna `attention_is_approximated_from_top_patches` registra isso.

## Execucao Disponivel Neste Projeto

Preparar features fuzzy a partir do VNN existente e das cores completas:

```bash
python fuzzy_dataset_features.py \
  --predictions-csv ../saida_vnn_mil_sharded/treino/predictions.csv \
  --colors-csv ../saida_analise_cores_alizams_dicom_rsna/cores_alizams_por_imagem.csv \
  --attention-csv ../saida_vnn_mil_sharded/treino/top_attention_patches.csv \
  --model-mode vnn \
  --output-csv ../resultados_fuzzy_vnn/fuzzy_features_predictions.csv
```

Quando os tres modelos estiverem treinados, forneca tambem
`--vnn-predictions-csv`, `--cnn-predictions-csv` e
`--hybrid-predictions-csv`. Assim a mesma inferencia pode ativar regras que
dependem simultaneamente dos tres escores.

As features estruturais antigas cobrem somente parte de outra execucao. Elas podem
ser fornecidas opcionalmente, e regras que dependem de features ausentes ficam
inativas, sem imputacao artificial:

```bash
python fuzzy_dataset_features.py \
  --predictions-csv ../saida_vnn_mil_sharded/treino/predictions.csv \
  --colors-csv ../saida_analise_cores_alizams_dicom_rsna/cores_alizams_por_imagem.csv \
  --attention-csv ../saida_vnn_mil_sharded/treino/top_attention_patches.csv \
  --structural-csv ../sistema_integrado_rsna_csv_treino/classificador_patches_xgboost_wavelet_gradiente_cor_fft/features_imagem_patches.csv \
  --model-mode vnn \
  --output-csv ../resultados_fuzzy_vnn/fuzzy_features_predictions.csv
```

Executar inferencia e metricas:

```bash
python run_fuzzy_inference.py \
  --features_csv ../resultados_fuzzy_vnn/fuzzy_features_predictions.csv \
  --predictions_csv ../saida_vnn_mil_sharded/treino/predictions.csv \
  --output_dir ../resultados_fuzzy_vnn
```

Gerar visualizacoes:

```bash
python plot_fuzzy_results.py \
  --fuzzy_csv ../resultados_fuzzy_vnn/fuzzy_predictions.csv \
  --output_dir ../resultados_fuzzy_vnn/plots
```

Explicar um caso:

```bash
python explain_fuzzy_case.py \
  --image_id 541722628 --patient_id 10011 \
  --fuzzy_csv ../resultados_fuzzy_vnn/fuzzy_predictions.csv \
  --attention_dir ../saida_vnn_mil_sharded/treino/top_attention_patches.csv \
  --output_dir ../resultados_fuzzy_vnn/explicacoes
```

Comparar modelos quando CNN e Hybrid tiverem sido treinados:

```bash
python compare_fuzzy_vs_models.py \
  --models_dirs ../resultados_cnn ../saida_vnn_mil_sharded/treino ../resultados_hybrid ../resultados_fuzzy_vnn \
  --output_dir ../comparacao_final
```

## CUDA Fortran

O calculo cromatico CUDA Fortran foi validado separadamente com a LUT AlizaMS e
produziu as mesmas cores dominantes da referencia Python. A camada fuzzy nao reabre
DICOM: ela carrega os CSVs ja gerados, independentemente de terem sido produzidos
por CUDA Fortran ou pelo fallback Python. Se o caminho do CSV nao identificar CUDA
Fortran, o montador informa que esta usando features precomputadas existentes.

## Reprodutibilidade e Limitacoes

- `fuzzy_parameters.json` salva pertinencias, regras e operadores usados.

## Selecao fuzzy de patches

O seletor opera somente sobre patches ja armazenados nos shards. Ele aplica a LUT
AlizaMS/RainbowB ao canal cinza cacheado, calcula `fuzzy_color_score` sem usar a
coluna `cancer` e combina:

```text
0.40 * fuzzy_color_score + 0.25 * gradiente_score
+ 0.20 * entropia_score + 0.15 * wavelet_high_score
```

`wavelet_high_score` e um detalhe Haar calculado offline sobre o patch cacheado,
pois os shards originais possuem FFT, mas nao possuem um canal wavelet. A selecao
e restrita aos oito patches candidatos ja salvos por imagem; ela nao procura novos
locais no DICOM.

Para a comparacao de selecao, o modo `vnn` reutiliza `VNNMILAttention` de
`train_mil_vnn_sharded.py`, a mesma arquitetura da baseline. O modelo hibrido e
uma comparacao adicional e nao deve ser interpretado como isolando apenas o efeito
da selecao de patches.

```bash
python fuzzy_patch_selector.py \
  --manifest ../saida_vnn_mil_sharded/cache_128/manifest.csv \
  --metadata-csv ../../rsna_kaggle_oficial/train.csv \
  --out-dir ../cache_fuzzy_selected \
  --top-k 4 --device cuda

python train_mil_vnn_fuzzy_selected.py \
  --manifest ../cache_fuzzy_selected/manifest.csv \
  --metadata-csv ../../rsna_kaggle_oficial/train.csv \
  --baseline-vnn-dir ../saida_vnn_mil_sharded/treino \
  --out-dir ../treino_fuzzy_selected \
  --mode all --epochs 10 --batch-size 64
```

Saidas principais: `selected_patches_manifest.csv`, `selection_parameters.json`,
montagens `patch_comparisons/*_before_after.png`, preditores treinados, patches de
maior atencao e `comparacao_selecao_fuzzy.csv`.

### Modo espectral completo

Com `--ranking-mode spectral` (padrao), o seletor tambem usa o mapa de entropia
local, os mapas FFT baixa/alta e sua razao, e energia wavelet Haar de tres niveis
calculada offline sobre cada patch cacheado. A formula passa a priorizar evidencia
espectral e cromatica sem usar o label:

```text
0.25 * fuzzy_color_score + 0.25 * fuzzy_spectral_score
+ 0.15 * gradiente_score + 0.15 * entropia_score
+ 0.10 * wavelet_high_score + 0.10 * fft_high_score
```
- As regras fornecidas sao hipoteses interpretaveis predefinidas, nao calibradas no teste.
- O threshold sweep sobre o split avaliado e exploratorio; para resultado independente,
  fixe regras e threshold usando validacao separada antes de avaliar o teste.
- O split original por `patient_id` deve ser preservado nos CSVs de modelo.
- Pseudocor RainbowB deriva de intensidade em escala de cinza; nao e cor biologica.
- Attention MIL indica peso do modelo e nao e ground truth de segmentacao.
