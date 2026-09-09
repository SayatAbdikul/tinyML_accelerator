# Ushqyn: research potential and Tang Nano 20K development assessment

Assessment date: 2026-09-08. Source revision inspected: `3aa5fe8`, plus the current working tree. The pre-existing change in `.claude/skills/project-analysis/SKILL.md` was left untouched. This review adds this document; it does not change the accelerator implementation.

**September 9 implementation follow-up:** the initial findings below are a dated
audit. Fresh Gowin results now supersede the absence of reports: the minimal UART
design routes, while the current accelerator fails synthesis at 273,847 DFF
against a 15,750 limit. Recovered historical reports distinguish 89.201 MHz
**synthesis** from 37.502 MHz **post-route**; neither proves current board execution.
The corrected software compiler and primary quality evidence are documented in
[phase 0–1 status](research/PHASE_0_1_STATUS.md). The broader novelty framing below
is narrowed by the [prior-work decision](research/novelty_matrix.md).

**Verdict: the project is a useful foundation for a strong FPGA research paper, but the current implementation and evidence do not support a top-tier paper yet.** A programmable INT8 accelerator running on the Tang Nano 20K is a worthwhile engineering objective. The research contribution needs to be a new, generalizable method for overcoming resource constraints, supported by convincing experiments. The choice of board, custom ISA, and successful MNIST inference are not sufficient novelty on their own.

The strongest direction is a compiler and shared compute architecture that jointly optimize **physical block-memory allocation, tensor lifetimes, tiling, and quantization boundaries**. Keep the Tang Nano 20K as the primary implementation target and use its constraints to motivate a broader architectural result.

## What was inspected and verified

The review covered the compiler and integer golden model, ONNX handling, ISA/configuration generation, simulation Conv2D/MaxPool paths, synthesizable FPGA hierarchy, memory IP and UART interface, timing constraints, test harnesses, CI, and documentation. It included a targeted literature comparison, rather than an exhaustive systematic literature review.

| Evidence | Result and interpretation |
|---|---|
| Existing fast compiler test selection | 58 tests passed. The README's count of 44 is stale. Passing these tests does not establish general ONNX semantic correctness. |
| ISA drift check | Passed during the compiler audit. It covers the generated simulation decoder, not all independent FPGA copies. |
| Conv2D cocotb regression, rerun during this review | 2/2 passed, with ReLU disabled/enabled on the existing small geometry. |
| MaxPool cocotb regression, rerun during this review | 2/2 passed, including the existing multi-channel geometries. |
| Targeted valid ONNX examples | Reproduced incorrect bias semantics, skipped BatchNormalization, skipped large standalone ReLU, missing terminal Add/STORE, and a bias-free Conv failure in the compiler/golden flow. |
| Physical FPGA validation | No new synthesis, place-and-route, power measurement, or board programming was performed. |
| Historical performance claims | The repository documents them, but no corresponding synthesis project/report, bitstream, or raw measurement package was found in the reviewed tree. Treat them as historical reported results pending reproduction. |

The unit-test runs used Verilator 5.044 and cocotb 1.9.2. Temporary build outputs and logs are under `/private/tmp/tinyml-review-*`. The compiler probes are in `/private/tmp/tinyml-compiler-audit.dEA2iQ/probe.py`; these temporary files are not a permanent reproducibility archive.

The existing strengths are substantial: a complete model-to-instruction workflow, modular execution units, a bit-level reference implementation, useful regressions, an ISA specification, liveness-based buffer-ID allocation, and practical work on BSRAM latency and tile prefetching. These are assets to preserve. The key distinction is that agreement between two implementations of the same arithmetic establishes consistency; it does not establish that the arithmetic correctly represents the imported neural network.

## Findings that should be addressed first

### 1. Quantization can change the meaning of the imported model

In [dram.py](/Users/sayat/Documents/GitHub/tinyML_accelerator/compiler/dram.py:97), weights and biases are independently normalized to INT8. In [golden_model.py](/Users/sayat/Documents/GitHub/tinyML_accelerator/compiler/golden_model.py:211), the INT8 bias is then added directly to an INT8 × INT8 accumulation. Its scale is not aligned to the product's scale.

A valid two-output Gemm makes the problem observable:

- Input: `x = [1, 0]`.
- Weight matrix: `W = diag(0.1, 0.1)`.
- Bias: `b = [0, 1]`.
- ONNX reference output: `[0.1, 1]`, selecting class 1.
- Current compiler/golden output: `[127, 1]`, selecting class 0.
- Increasing the second bias to 100 leaves the compiler/golden output unchanged.

This is not simply an ordinary small quantization error. Independent normalization discards the bias magnitude relative to the dot product. RTL agreement with that golden model would reproduce the same semantic error.

Start with calibrated static activation scales, symmetric per-channel weight scales, and INT32 biases in product units. For output channel `c`, the arithmetic should have the form:

```
acc[c] = sum((qx - zx) * qw[c]) + round(b_float[c] / (sx * sw[c]))
qy[c]  = clamp(round(acc[c] * sx * sw[c] / sy) + zy)
```

Specify the exact rounding, saturation, zero-point handling, and overflow behavior. Compile the scale ratios into integer multipliers/shifts. This follows the scale relationship documented in the [LiteRT quantization specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec). If dynamic activation scales remain available, bias conversion must account for the corresponding runtime input scale; fixed independently quantized INT8 biases cannot supply that relationship.

### 2. Unsupported ONNX operations can silently alter execution

[compile.py](/Users/sayat/Documents/GitHub/tinyML_accelerator/compiler/compile.py:174) treats BatchNormalization as a passthrough without implementing or folding its transform. [The Add branch](/Users/sayat/Documents/GitHub/tinyML_accelerator/compiler/compile.py:246) skips Add, and [large standalone ReLU handling](/Users/sayat/Documents/GitHub/tinyML_accelerator/compiler/compile.py:250) can emit no ReLU at all.

The targeted probes confirmed:

- A 1024-element ReLU followed by MaxPool compiles without the ReLU.
- A BatchNormalization with a negative channel scale followed by Gemm produces the wrong sign/class in the compiler/golden execution.
- A Gemm followed by terminal Add compiles without Add and without a final STORE.
- A bias-free Conv compiles but fails in golden execution with `KeyError: 0`.

Add an explicit supported-operator and supported-attribute validation pass. Reject unsupported operations and geometries before emitting an executable. Fold inference BatchNormalization into weights/biases, synthesize a zero bias for bias-free operators, and implement residual Add when expanding to residual networks. Validate groups, dilation, padding, stride, Gemm transpose/scaling, and pooling attributes rather than assuming the exported example's defaults.

### 3. The default FPGA configuration has drifted from the documented board build

[generate_config.py](/Users/sayat/Documents/GitHub/tinyML_accelerator/generate_config.py:19) sources `SimProfile`, while [the FPGA top](/Users/sayat/Documents/GitHub/tinyML_accelerator/src/tinyml_accelerator_top_fpga.sv:1) imports that generated package. This selects 32-element outer tiles, 16-bit addresses, and 16 vector buffers of 4 KiB each. The documented FPGA profile has smaller values.

The eight-lane GEMV core has an explicit bridge in [gemv_execution.sv](/Users/sayat/Documents/GitHub/tinyML_accelerator/src/gemv_execution.sv:52). Consequently, the difference between 32-element outer tiles and eight compute lanes is not itself proof of a core deadlock. The confirmed problems are configuration consistency, memory capacity, and matching compiler/layout parameters.

The current nominal default hierarchy allocates approximately:

| Storage | Nominal payload |
|---|---:|
| Gowin_SP main memory | 32 KiB |
| 16 vector buffers × 4 KiB | 64 KiB |
| 2 matrix buffers × 16 KiB | 32 KiB |
| GEMV x/result memories | 8 KiB |
| Total | 136 KiB |

Even the first three terms exceed the board's 103.5 KiB raw BSRAM capacity. This is a source-level allocation calculation, not a new synthesis utilization result. Width/depth packing and trimming affect actual mapped usage. The existing 8/32-bit memory arrangements also cannot treat every parity bit as usable payload.

The main memory IP has a 15-bit address port, while the current top defaults to 16 bits: [Gowin_SP declaration](/Users/sayat/Documents/GitHub/tinyML_accelerator/src/gowin_sp/gowin_sp.v:18), [connection](/Users/sayat/Documents/GitHub/tinyML_accelerator/src/simple_memory.sv:285). This aliases the upper 32 KiB if the wider address range is used.

Generate both simulation and board configuration from an explicit target selection, and use the same generated configuration for compilation, elaboration, and deployment. Add elaboration/compiler checks for every capacity and width limit. Consolidate the three overlapping RTL implementations around shared modules with technology-specific memory wrappers.

### 4. Reported operating frequency and end-to-end throughput need qualification

The [board constraint](/Users/sayat/Documents/GitHub/tinyML_accelerator/src/fpga_project_1.sdc:2) specifies 27 MHz. The [accelerator instance](/Users/sayat/Documents/GitHub/tinyML_accelerator/src/fpga_top.sv:231) uses that clock directly; UART also assumes 27 MHz. No clock multiplication is present in the inspected board hierarchy.

If the reported 25,470-cycle count applies, its compute latency is:

- At 27 MHz: approximately **0.943 ms**.
- At 89.2 MHz: approximately **0.286 ms**.

The second number is conditional on an implementation actually operating at that frequency. An achievable Fmax report and an operating clock must be reported separately. Preserve actual timing reports, device/speed grade, tool version, clock configuration, and source revision.

At the configured 115200-baud 8N1 UART, transmitting only the 784 input bytes requires at least **68 ms**, before protocol overhead or output transfer. Report core latency, input/output latency, and model-loading latency separately. Core inferences/second should not be presented as sustained host-to-board throughput.

The MLP contains 10,112 useful MACs. Dividing by the reported 25,470 cycles gives about 0.397 useful MAC/cycle, or **5% of an eight-MAC/cycle arithmetic ceiling**. This derived figure includes noncompute overhead and does not isolate the bottleneck, but it strongly motivates profiling stalls and data movement before adding more lanes.

### 5. Current tests do not justify the strongest bit-exactness claims

Both heavy test suites set `pass_exact_match = True` and `pass_max_error = True`: [simulation suite](/Users/sayat/Documents/GitHub/tinyML_accelerator/test/heavy_test/test_full_mnist.py:550), [FPGA-tier suite](/Users/sayat/Documents/GitHub/tinyML_accelerator/test/heavy_test_fpga/test_full_mnist.py:556). A passing run therefore does not imply zero numerical mismatches. This does not prove historical outputs were wrong; it means the pass criterion cannot establish the documented claim.

Restore unconditional exact-output assertions for integer-reference versus RTL validation. Keep model accuracy evaluation as a separate metric; a correctly implemented classifier can still misclassify an input. Compare the independently specified quantized model against float inference as a separate third check.

The FPGA-tier simulation also uses a different wrapper and simulated memory/fetch path. Test the actual board hierarchy with faithful synchronous memory behavior, repeated inference, command transactions, and explicit failure handling. Include long transfers, partial tiles, channel/tile boundaries, signed extremes, and capacity rejection. The existing Conv regression is useful but has only one small spatial/channel geometry with two ReLU settings.

### 6. Board control and CNN portability require architectural work

The [fetch PC](/Users/sayat/Documents/GitHub/tinyML_accelerator/src/fetch_unit_fpga.sv:49) resets on reset, not on a new inference request. Program completion returns to idle without rewinding it. Meanwhile, [fpga_top.sv](/Users/sayat/Documents/GitHub/tinyML_accelerator/src/fpga_top.sv:53) forces `system_on` high and leaves the reset-button toggle commented out. A robust repeat-inference interface needs an explicit start-PC/reset mechanism and addressable input writes.

The board execution path does not implement CNN dispatch. The simulation Conv implementation relies on [asynchronous random vector reads](/Users/sayat/Documents/GitHub/tinyML_accelerator/rtl/buffer_file.sv:41), gathers patch elements serially, and keeps a full INT32 output tensor. Copying it into the FPGA directory is not sufficient to obtain a good BSRAM implementation.

There are also hidden limits beyond the visible configuration: [10-bit patch/spatial counts](/Users/sayat/Documents/GitHub/tinyML_accelerator/rtl/execution_unit/conv2d_execution.sv:64) and [12-bit accumulator indexing](/Users/sayat/Documents/GitHub/tinyML_accelerator/rtl/execution_unit/conv2d_execution.sv:297). Increasing `ACCUM_DEPTH` or ISA fields alone will not remove these limits. Use derived widths, validate geometries, replace runtime division/modulo in hot address paths with counters/precomputed strides, and make unsupported hardware opcodes raise an error.

## A practical architecture for Tang Nano 20K

The board has **20,736 LUT4s, 46 × 18-Kbit BSRAM blocks, 48 18×18 multipliers, and 64 Mbit of 32-bit SDR SDRAM**. The SDRAM is 8 MiB and is distinct from fabric BSRAM. The repository's references to this memory as PSRAM should be corrected. DSP primitive counts such as MULT36X36 and MULTADDALU18X18 must be translated into physical resource usage before comparison with the 48-multiplier specification. [Official Sipeed board specification](https://wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html).

Use one eight-lane compute engine initially, shared between FC, ordinary convolution, pointwise convolution, and eventually depthwise convolution. Feed it from a banked BSRAM scratchpad and a line/window generator. Add a pipelined integer requantizer and stream ReLU/pooling where legal. Separate instruction storage and control from bulk data traffic.

Replace whole fixed-size buffers per logical ID with descriptors containing base, size, shape/stride, and quantization metadata. The existing liveness allocator is a useful starting point, but reusing IDs does not reduce the physically instantiated buffer bank. Allocate physical bytes/banks based on peak simultaneous live tensors, including port requirements and block rounding.

An illustrative initial memory budget is below. It is a design allocation to validate through synthesis, not an achieved result:

| Function | Payload target | Blocks at 2 KiB payload/block |
|---|---:|---:|
| Program and command buffering | 4 KiB | 2 |
| Two weight tiles | 16 KiB | 8 |
| Two activation tiles | 16 KiB | 8 |
| Partial sums/scratch | 16 KiB | 8 |
| Line/window storage | 8 KiB | 4 |
| Bias, scale, descriptor metadata | 4 KiB | 2 |
| Total | 64 KiB | 32 of 46 |

Port duplication and packing may consume additional blocks. The two 8-KiB weight tiles require actual matrix/layer tiling for layers larger than one tile; they cannot directly replace the existing complete 9,408-byte first MLP matrix without changing execution.

The existing SmallCNN has only 2,324 unpadded INT8 weights. Its largest INT8 feature map is 2,704 bytes; storing that map as INT32 takes 10,816 bytes. These are small enough that a carefully allocated on-chip implementation is plausible. An SDRAM controller is not a prerequisite for demonstrating this network.

After on-chip bring-up, use the 8-MiB SDRAM for weights and larger activations with burst DMA and double buffering. SDRAM introduces refresh, initialization, arbitration, and variable access latency. The current fixed-latency byte-memory interface needs a request/response handshake. At least 23 byte-address bits are needed to address 8 MiB, and wider address fields must reach every relevant module and host command.

## The research direction worth pursuing

**Recommended research question:** Under a fixed physical BSRAM budget, can a compiler jointly choose operator fusion, tile geometry, quantization boundaries, and memory banking to improve achievable model size and energy/latency while preserving a defined accuracy target?

The contribution would be the optimization method, its hardware support, and measured tradeoffs. An objective could minimize measured/predicted inference energy subject to latency, accuracy, DSP/LUT limits, and a physical block-memory budget. The cost model should account for block fragmentation, required read/write ports, weight reloads, halo recomputation, INT32 partial sums, quantization passes, and SDRAM bursts. Calibrate it against implemented designs and report prediction error.

There are three useful experiments, in priority order:

1. **Static-scale streaming as a strong baseline.** Fix scale semantics, use calibrated scales/per-channel weights, finish small output tiles, and requantize immediately. This removes the whole-tensor scale-discovery barrier. It is the most practical path to a reliable board implementation, but established quantization and streaming techniques are not new research by themselves.

2. **Exact storage reduction for dynamic quantization.** For Conv → ReLU → MaxPool, retain the original scale reduction over every completed biased convolution output, while storing only pooled INT32 maxima. Once the global scale is known, quantize those pooled maxima and apply ReLU. For a fixed monotone quantizer `Q`, `max(ReLU(Q(a_i))) = ReLU(Q(max(a_i)))`. For the first SmallCNN layer, the saved INT32 tensor could shrink from 10,816 to 2,704 bytes, excluding line buffers and metadata. Crucially, the global max-absolute reduction must still include negative values and all original outputs, including pixels discarded by odd pooling borders. Pooling first and computing the scale from pooled values would change semantics. This requires completing outputs before pooling; changing the existing weight-tile-outer loop can increase weight rereads. Measure the storage/bandwidth tradeoff. Quantization/pooling commutation is established practice, so the identity alone is not a novelty claim. [TensorRT quantization propagation documentation](https://docs.nvidia.com/deeplearning/tensorrt/11.2.1/inference-library/quantized-types-explicit-quantization.html).

3. **A compiler that selects among these schedules under real BSRAM constraints.** Compare full-tensor dynamic scaling, pooled dynamic scaling, static streaming, and SDRAM spilling using the same compute engine. Include calibration requirements and accuracy in comparisons where arithmetic changes. Derive the break-even conditions rather than presenting only a favorable network. A useful research outcome would explain when spending bandwidth/recomputation saves enough memory to enable a larger network or reduce energy.

Mixed INT4/INT8 execution or narrower accumulators can be a later extension. They require error/overflow analysis and training support; merely making bit width configurable is insufficient novelty. Arbitrary sparsity and a larger systolic array are lower priorities until profiling shows their overhead is justified.

## Relevant prior work and novelty boundaries

| Prior work | Why it matters to this project |
|---|---|
| [FINN-R](https://arxiv.org/abs/1809.04570) | Already provides automated quantized FPGA design exploration with resource/performance modeling. ONNX-to-accelerator generation and configurability need a more specific contribution. |
| [Open-source FPGA-ML codesign for MLPerf Tiny](https://arxiv.org/abs/2206.11791) | Establishes FINN/hls4ml-based TinyML evaluation as relevant prior work. Use comparable workloads and accuracy rather than MNIST alone. |
| [CFU Playground](https://arxiv.org/abs/2201.01863) | Demonstrates a full hardware/software TinyML workflow and accelerator/CPU design exploration. An open toolchain is valuable but not sufficient novelty. |
| [MCUNetV2](https://arxiv.org/abs/2110.15352) | Patch scheduling and network/schedule co-design already reduce activation memory on MCUs. An FPGA contribution must account for different physical memory and parallel-access constraints. |
| [msf-CNN](https://arxiv.org/abs/2505.11483) | Recent work searches multi-stage patch fusion schedules on MCUs. A generic fusion search would overlap with an established line of research. |
| [A2Q](https://arxiv.org/abs/2308.13504) | Accumulator-aware quantization and overflow-constrained FPGA resource reduction are established topics. Narrow accumulators need differentiation. |
| [MicroCNN on Tang Nano 20K](https://github.com/SweiryDev/MicroCNN-TangNano20k) | A public implementation already targets CNN inference on this exact board. It is an engineering comparator, not a peer-reviewed performance baseline; its reported measurements were not independently verified here. |

This comparison establishes that the broad idea is crowded. It does not prove that the proposed joint optimization is novel. Before committing the paper to that claim, inspect the full closest papers and formulate exactly which scheduling/quantization/memory constraint they do not solve. The possibility of identifying that gap is the research potential, not a result already delivered by the repository.

## Experiments needed for a competitive paper

Use MNIST for bring-up, then move to at least three meaningful workloads. A practical progression is an anomaly-detection MLP/autoencoder, keyword spotting with a depthwise-separable CNN, and visual wake words or CIFAR-10 classification. These tasks have established references in [MLPerf Tiny](https://mlcommons.org/benchmarks/inference-tiny/). Depthwise convolution is needed for the proposed KWS/MobileNet path; residual Add is needed for a ResNet path. Match an explicitly chosen benchmark version, model, preprocessing, and quality target. Modified models should be identified as research variants rather than presented as official benchmark results.

For each workload, report:

- Float and quantized quality on the complete held-out set, including calibration/training details.
- Integer-reference/RTL/board mismatches, ideally zero under a fixed integer specification.
- Actual operating clock, post-route timing, LUT/FF/BSRAM/physical DSP use, and memory capacity.
- Per-layer cycles and stalls, effective MAC utilization, bytes transferred, and peak live storage.
- Core latency, complete application latency, and model-loading cost with clear boundaries.
- Measured energy per inference, idle power, and active board power with the measurement boundary specified.
- Source revision, build scripts, tool versions, generated configuration, model hashes, and raw logs.

The most important baselines are on the same board: the corrected existing architecture; a conventional static-quantized tiled implementation; an implementation using liveness-based memory reuse alone; and the complete proposed method. Add a reproducible public same-board design where models can be matched. FINN/hls4ml and MCU implementations provide wider context, but do not assume their vendor flows directly target Gowin. Cross-device numbers must not substitute for the controlled same-device ablations. An interpreted Python golden model is a correctness reference, not a credible optimized performance baseline.

Useful ablations independently enable physical memory reuse, streaming/line buffers, quantization changes, pooling fusion, and scheduling. Sweep several SRAM budgets and lane/tile configurations. If feasible, validate the cost model on a second FPGA or block-memory geometry so the contribution extends beyond one board. Report unfavorable cases and compilation/implementation failures as well as improvements.

Possible internal go/no-go targets are a substantial memory reduction versus a corrected tuned baseline, an improvement in measured energy or latency without exceeding a predeclared quality-loss budget, and at least one realistic workload enabled by the new method. These are experiment targets, not predicted gains or guarantees of acceptance.

## Development order

1. **Establish a trustworthy baseline:** repair quantization semantics and unsupported-op handling; restore strict checks; unify board configuration/source selection; fix repeat-run control; reproduce a board MLP build at a declared clock.
2. **Fit the current SmallCNN on chip:** share compute resources, use synchronous BSRAM interfaces, allocate scratchpad storage from tensor lifetimes, and verify repeated CNN inference through the real board path. Publish the first complete build/measurement package.
3. **Add one representative workload family:** depthwise/pointwise support and correctly folded BatchNormalization for keyword spotting is a focused next step. Add SDRAM/DMA when a selected model requires it.
4. **Test the research hypothesis:** implement competing memory/quantization schedules, construct a measured cost model, and automate the selection. Run controlled ablations across models and budgets.
5. **Write around the demonstrated contribution:** present the specific bottleneck, algorithm, correctness conditions, hardware support, and measured improvements. A completed board port alone remains an engineering demonstration; a general method with strong evidence could support a competitive FPGA/architecture paper.

The immediate milestone should be **one reproducible, numerically correct CNN executing repeatedly on the Tang Nano 20K from the same RTL that is tested**. The subsequent paper should explain what this project teaches about efficient TinyML execution under severe physical memory constraints.
