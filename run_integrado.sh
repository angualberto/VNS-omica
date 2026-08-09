#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem"
SCRIPT="$ROOT/lidc_sistema_integrado.py"

# Detecta python no venv local ou usa python3 do sistema
if [ -x "$ROOT/.venv/bin/python" ]; then
	PY="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
	PY=python3
else
	echo "Python 3 nao encontrado. Ative o .venv ou instale python3." >&2
	exit 1
fi

# Caminho padrão: RSNA baixado no HD maior, com train.csv + train_images
DEFAULT_RSNA_ROOT="/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/rsna_breast_cancer_detection"
DEFAULT_DICOM_ROOT="$DEFAULT_RSNA_ROOT"
DEFAULT_RETRAIN_DICOM_ROOT="/media/angualberto/C8C814BEC814AD26/TCIA/LIDC-IDRI/extracted"
DEFAULT_MODEL_DIR="$ROOT/sistema_integrado"
DEFAULT_OUT_DIR="$ROOT/sistema_integrado"

if [ ! -d "$DEFAULT_DICOM_ROOT" ]; then
	DEFAULT_DICOM_ROOT="/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/TCIA-PT-all/manifest-PT-all/ACRIN-FLT-Breast/ACRIN-FLT-Breast_001"
fi
if [ ! -d "$DEFAULT_DICOM_ROOT" ] && [ -d "$DEFAULT_RETRAIN_DICOM_ROOT" ]; then
	DEFAULT_DICOM_ROOT="$DEFAULT_RETRAIN_DICOM_ROOT"
fi

has_arg() {
	local needle="$1"
	shift
	for arg in "$@"; do
		if [ "$arg" = "$needle" ]; then
			return 0
		fi
	done
	return 1
}

# Argumentos padrao (podem ser sobrescritos passando parametros ao script)
DEFAULT_ARGS=(--mode features --dataset-type mammography --dicom-root "$DEFAULT_DICOM_ROOT" --model-dir "$DEFAULT_MODEL_DIR" --out-dir "$DEFAULT_OUT_DIR" --max-series 300 --patch-size 128 --epochs 50 --batch-size 8 --save-overlays none)
if [ $# -eq 0 ]; then
	ARGS=("${DEFAULT_ARGS[@]}")
else
	ARGS=("$@")
	if ! has_arg "--dicom-root" "${ARGS[@]}"; then
		ARGS+=("--dicom-root" "$DEFAULT_DICOM_ROOT")
	fi
	if ! has_arg "--model-dir" "${ARGS[@]}"; then
		ARGS+=("--model-dir" "$DEFAULT_MODEL_DIR")
	fi
	if ! has_arg "--out-dir" "${ARGS[@]}"; then
		ARGS+=("--out-dir" "$DEFAULT_OUT_DIR")
	fi
	if ! has_arg "--dataset-type" "${ARGS[@]}"; then
		ARGS+=("--dataset-type" "mammography")
	fi
	if ! has_arg "--patch-size" "${ARGS[@]}"; then
		ARGS+=("--patch-size" "128")
	fi
	if ! has_arg "--save-overlays" "${ARGS[@]}"; then
		ARGS+=("--save-overlays" "none")
	fi
fi

echo "ROOT=$ROOT"
echo "Using Python: $PY"
echo "Running: $PY $SCRIPT ${ARGS[*]}"

# Executa com log e faz fallback se houver erro por poucas anotacoes
LOG=$(mktemp /tmp/run_integrado.XXXX.log)
set +e
"$PY" "$SCRIPT" "${ARGS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

if [ $RC -eq 0 ]; then
	echo "Pipeline concluído com sucesso. Log: $LOG"
	exit 0
fi

if grep -q "Poucas fatias positivas para treinar U-Net" "$LOG" || grep -q "RuntimeError: Poucas fatias positivas" "$LOG"; then
	echo "Detectado erro de poucas anotações LIDC. Tentando fallback -> modo infer usando modelos existentes."

	# procura --model-dir nos args
	MODEL_DIR=""
	for ((i=0;i<${#ARGS[@]};i++)); do
		if [ "${ARGS[$i]}" = "--model-dir" ]; then
			MODEL_DIR="${ARGS[$i+1]}"
			break
		fi
	done
	if [ -z "$MODEL_DIR" ]; then
		# default model dir = data_root/sistema_integrado ; tentar infer a partir do DEFAULT_DATA_ROOT
		MODEL_DIR="/media/angualberto/C8C814BEC814AD26/TCIA/LIDC-IDRI/sistema_integrado"
	fi

	if [ -f "$MODEL_DIR/modelos_integrados_sem_randomforest.joblib" ] && [ -f "$MODEL_DIR/unet_lidc_best.pt" ]; then
		echo "Modelos encontrados em: $MODEL_DIR -> rodando inferência"
		# constrói argumentos de infer substituindo/indicando --mode infer
		NEW_ARGS=()
		SKIP_NEXT=0
		for ((i=0;i<${#ARGS[@]};i++)); do
			if [ $SKIP_NEXT -eq 1 ]; then SKIP_NEXT=0; continue; fi
			a="${ARGS[$i]}"
			if [ "$a" = "--mode" ]; then
				NEW_ARGS+=("--mode" "infer")
				SKIP_NEXT=1
				continue
			fi
			NEW_ARGS+=("$a")
		done
		# se nao tinha --mode, adiciona
		has_mode=0
		for a in "${NEW_ARGS[@]}"; do
			if [ "$a" = "--mode" ]; then has_mode=1; break; fi
		done
		if [ $has_mode -eq 0 ]; then
			NEW_ARGS+=("--mode" "infer")
		fi

		echo "Running infer: $PY $SCRIPT ${NEW_ARGS[*]}"
		"$PY" "$SCRIPT" "${NEW_ARGS[@]}"
		exit $?
	else
		echo "Modelos não encontrados em $MODEL_DIR. Verifique --model-dir ou treine em uma base anotada." >&2
		echo "Log completo: $LOG"
		exit $RC
	fi
else
	echo "Processo terminou com erro (não relacionado a poucas anotações). Veja log: $LOG" >&2
	exit $RC
fi
