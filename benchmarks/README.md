# Frozen benchmark artifacts

KWS and VWW are equal-priority primary workloads; anomaly detection is secondary.
MLCommons source is pinned at `4addd0fa08d216e20637637874e084895f289da4`.
Each `manifests/{kws,vww,ad}.json` pins artifact/source/runner/evaluator/index
SHA256 values. `*.inventory.json` was generated from the downloaded official INT8
TFLite artifact, not from a presumed architecture or a hand-written layer list.

| Artifact | MACs per inference | Constant tensor bytes | Required operator families |
|---|---:|---:|---|
| KWS `kws_ref_model.tflite` | 2,656,768 | 24,376 | ordinary/depthwise Conv, average pool, FC, reshape, softmax |
| VWW `vww_96_int8.tflite` | 7,489,664 | 219,072 | ordinary/depthwise Conv, average pool, FC, reshape, softmax |
| AD `ad01_int8.tflite` | 264,192 | 270,880 | FC |

These are **derived inventory counts**, not measured accelerator throughput or
memory footprints. MAC counts exclude pooling/activation work. Constant bytes
include tensor buffers such as INT32 bias; they are not padded physical BSRAM
consumption. Inventories retain tensor types, shapes, quantization axes/scales,
operator versions and scalar builtin attributes, including fused activation.
No unknown/custom operator was found in these three frozen INT8 artifacts.

## Results and scope — September 9, 2026

| Complete declared split | Float source | Static INT8 v2 | Target | Calibration |
|---|---:|---:|---:|---:|
| KWS canonical Speech Commands v2 test, 4,890 clips | 4,507 / 4,890 (92.17%) | 4,514 / 4,890 (92.31%) | 90% | 96 training clips/windows |
| VWW training-recipe validation, 10,961 images | 9,261 / 10,961 (84.49%) | 9,239 / 10,961 (84.29%) | 80% | 11 upstream calibration images |

The VWW split is the complete first 10% of filenames **within each class** of the
Silicon Labs 96×96 archive, following the upstream training partition without
random augmentation. It is **not certified as the official MLPerf accuracy set**.
These are research validation results, not an MLPerf submission. The separate
1,000-file KWS performance-stimulus manifest is not used as the full accuracy set.
Calibration and accuracy have no matching IDs or preprocessed input contents.
No quality-set tuning, retraining, or weight replacement was performed.

Source-framework conversion uses 16 deterministic float probes, `atol=1e-5`,
`rtol=1e-4`: KWS original SavedModel, VWW float TFLite plus original H5 verification,
and AD original H5. The nominal KWS float TFLite was hybrid-quantized and failed
the strict conversion check; the AD TFLite-derived route failed original-H5 parity.
Those rejected routes are preserved under `docs/research/evidence/`. They were
replaced by source-framework conversion, not by loosening tolerances.
`*.canonical-inventory.json` describes these actual ONNX graphs separately from
the published INT8 TFLite inventories; MAC counts agree for all three workloads.
AD conversion/canonicalization passes, but its dataset/calibration and ROC-AUC
remain unverified. That portion of R02 remains open.

The accelerator boundary is **integer logits**, followed by host argmax (first
index on ties); no probability output is promised. Explicit boundary records are
in `*.boundary.json`. Software v2 supports the complete KWS/VWW logits graphs.
Independent centered-input integer arithmetic matches every intermediate layer
on three selected real inputs per primary workload. This is a directed all-layer
check, not a claim of independent-oracle evaluation of every accuracy sample.
The current RTL cannot execute v2 images and requesting that target rejects.

## Evidence map

* `{kws,vww,ad}.json`: immutable source files, evaluator and model hashes.
* `dataset-archives.json`: exact archive URLs, sizes and SHA256 before extraction.
* `{kws,vww}.data.json`: every calibration/accuracy sample ID, raw and feature
  SHA256, label and preprocessing recipe; NPZ payload hashes.
* `*.calibration.json`: every intermediate float range and calibration provenance.
* `*.build.json`: source/logits/canonical graph, image, numerical code and
  environment hashes. Rebuilding from saved calibration reproduced both images
  byte for byte after the final compiler change.
* `*.parity.json`, `*.integer-parity.json`, `*.float-quality.json`,
  `*.static-quality.json`: separate conversion, implementation and quality checks.

Payloads remain outside Git; manifests and recipes are committed. Exact image
hashes require the pinned conversion/compiler environments. Different ONNX
export versions or CPU math may produce different bytes; revalidate instead of
silently replacing reference hashes.

## Reproduce source conversion

Use separate environments: Python 3.11 with
`tools/research/conversion-environment.lock.txt` for TensorFlow conversion and
preprocessing; Python 3.13 with `docs/research/evidence/compiler-environment.lock.txt`
for compilation. The full locks describe the tested macOS environment; the
smaller `requirements-conversion.txt` is the direct dependency list. The published
TFLite inventory uses `requirements-inventory.txt` in the compiler environment.
Commands below assume repository root, writable `work/`, compiler Python `python`,
and conversion Python stored in `CONVERT_PY`. Set it to your isolated interpreter.

```sh
mkdir -p work/upstream work/quality
python tools/research/fetch_artifacts.py benchmarks/manifests/kws.json work/upstream
python tools/research/fetch_artifacts.py benchmarks/manifests/vww.json work/upstream
python tools/research/fetch_artifacts.py benchmarks/manifests/ad.json work/upstream
"$CONVERT_PY" tools/research/convert_savedmodel.py work/upstream/benchmark/training/keyword_spotting/trained_models/kws_ref_model work/kws.onnx work/kws.parity.json
"$CONVERT_PY" tools/research/convert_benchmark.py work/upstream/benchmark/training/visual_wake_words/trained_models/vww_96_float.tflite work/vww.onnx work/vww.parity.json
"$CONVERT_PY" tools/research/validate_keras_parity.py work/upstream/benchmark/training/visual_wake_words/trained_models/vww_96.h5 work/vww.onnx work/vww.parity.json
"$CONVERT_PY" tools/research/convert_keras.py work/upstream/benchmark/training/anomaly_detection/trained_models/ad01.h5 work/ad.onnx work/ad.parity.json
```

Download the three archives in `dataset-archives.json`, verify each SHA256, and
extract into separate directories. Here `work/speech-train` and `work/speech-test`
contain the WAV class directories, while `work/vww/vw_coco2014_96` contains the
`person` and `non_person` directories. Keep the upstream split/recipe files.

```sh
"$CONVERT_PY" tools/research/prepare_quality_data.py kws --upstream work/upstream --data work/speech-train --test work/speech-test --input-name serving_default_input_1:0 --output-dir work/quality
"$CONVERT_PY" tools/research/prepare_quality_data.py vww --upstream work/upstream --data work/vww/vw_coco2014_96 --input-name input_1 --output-dir work/quality
```

## Compile and validate

Apply the explicit classifier boundary with the compiler Python:

```python
import json
from pathlib import Path
import sys
import onnx
sys.path.insert(0, 'compiler')
from classifier_boundary import logits_model
for name in ('kws', 'vww'):
    graph, boundary = logits_model(onnx.load(f'work/{name}.onnx'))
    onnx.save(graph, f'work/{name}-logits.onnx')
    Path(f'work/{name}.boundary.json').write_text(json.dumps(boundary, indent=2))
```

```sh
python compiler/static_cli.py work/kws-logits.onnx work/quality/kws.calibration.npz work/quality/kws.uq2
python compiler/static_cli.py work/vww-logits.onnx work/quality/vww.calibration.npz work/quality/vww.uq2
python tools/research/verify_integer_images.py work/quality/kws.uq2 work/quality/kws.accuracy.npz --report work/kws.integer-parity.json
python tools/research/verify_integer_images.py work/quality/vww.uq2 work/quality/vww.accuracy.npz --report work/vww.integer-parity.json
python tools/research/evaluate_static.py work/quality/kws.uq2 work/quality/kws.accuracy.npz --sha256 66f6b5cd75410e7b7b5d483dab8e48c363256822a066689fe895041c7941b8ac --report work/kws.static-quality.json
python tools/research/evaluate_static.py work/quality/vww.uq2 work/quality/vww.accuracy.npz --sha256 fc57a65876dae96143e22296c334e4a82cd3dba2b99b8299b28fc337d41fb2b0 --report work/vww.static-quality.json
"$CONVERT_PY" tools/research/evaluate_float_source.py work/upstream/benchmark/training/keyword_spotting/trained_models/kws_ref_model work/quality/kws.accuracy.npz --input-name serving_default_input_1:0 --report work/kws.float-quality.json
"$CONVERT_PY" tools/research/evaluate_float_source.py work/upstream/benchmark/training/visual_wake_words/trained_models/vww_96.h5 work/quality/vww.accuracy.npz --input-name input_1 --report work/vww.float-quality.json
make ci PYTHON=python
```

For model inventories, run `inventory_tflite.py INPUT REPORT` on the published
INT8 model and `inventory_onnx.py INPUT REPORT --classifier` on KWS/VWW converted
ONNX (omit `--classifier` for AD). Every output tensor and required operator must
remain accounted for. Do not use full-dataset accuracy as a substitute for exact
integer implementation checks or physical board validation.
