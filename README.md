# VNS-omica

Conjunto de pipelines experimentais em Python/PyTorch para análise computacional de imagens médicas (mamografia RSNA e DICOM), com foco em **VNN (Volterra Neural Networks)** pixel-wise e por patch, atenção MIL, segmentação com U-Net e análise de componentes cromáticas via LUTs AlizaMS/RainbowB.

> Pesquisa experimental. Nenhum resultado aqui substitui anotação radiológica, segmentação médica validada ou diagnóstico clínico.

## Módulos principais

| Área | Arquivos |
|------|----------|
| Sistema integrado LIDC (DICOM → U-Net → Fuzzy/Fourier/textura → NODULO) | `lidc_sistema_integrado.py`, `lidc_unet_segmentacao.py`, `lidc_fourier_segmentacao.py` |
| VNN pixel-wise em mamografia | `train_vnn_mamografia.py`, `vnn_pixelwise_mammo.py` |
| Pipeline VNN-MIL completo (8 canais, shards, treino CUDA) | `vnn_pixelwise_project/` (ver [README](./vnn_pixelwise_project/README.md)) |
| Camada fuzzy de probabilidade + análise de cores AlizaMS | `vnn_pixelwise_project/fuzzy_cancer_rules.py`, `analisar_*_rsna.py` |
| XGBoost com patches wavelet/gradiente/cor/FFT | `treinar_xgboost_patches_wavelet_gradiente_cor_fft.py` |
| Utilitários de download | `baixar_rsna_kaggle.sh`, `baixar_rsna_hd_maior.sh` |

Documentação detalhada do submódulo de mamografia VNN: [`vnn_pixelwise_project/README.md`](./vnn_pixelwise_project/README.md).

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requisitos principais: `torch>=2.6`, `torchvision`, `pydicom`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`.

## Teste rápido

```bash
bash run_test.sh
```

Executa um sanity check de `torch`/CUDA e o pipeline de segmentação por Fourier. Para o pipeline LIDC integrado:

```bash
bash run_integrado.sh
```

## Uso

Cada pipeline tem entrada própria (CSVs com `image_path,cancer` para mamografia; DICOM + anotações XML para LIDC). Exemplos completos de treino, predição, cache offline e sharding estão no [`README.md`](./vnn_pixelwise_project/README.md) do módulo de mamografia.

## Aviso médico

Este projeto é **experimental**. Imagens médicas exigem validação por especialistas e avaliação regulatória para qualquer uso clínico; este código não substitui diagnóstico.

## Licença

Uso acadêmico e de pesquisa.