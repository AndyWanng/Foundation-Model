# 第一次正式 24k MRI/PET 预训练：workstation 运行手册

本仓库在原 `mri_pet_geomc_pipeline_2026-08-09` preliminary 实现上加入正式 pretraining 路径。默认协议是一套单模型、单次正式训练，不执行 ablation，也不使用 performance gate 或 performance early stopping。本次代码同步不包含 finetuning、`mri_pet_geomc_downstream` 或 downstream task evaluation。

## 1. 这次训练的边界

- 数据只按 workstation 上实际存在的 FOMO MRI 和 ADNI PET 做 physical inner join。PT030、缓存目录、manifest-only PET 以及不存在的 ADNI MRI 都不进入训练。
- split 固定为 dataset-stratified、subject-level 80/10/10；同一 subject 永远不跨 split。
- MRI 与 PET 没有 subject-level 配对关系，因此不创建 MRI-PET edge，也不伪造 pseudo-pair。
- 每个 train observation 在每个 coverage 中恰好做一次 anchor；当前配置的 50 coverages 即 50 次完整无放回覆盖。
- coverage 1–2 只做 self-JEPA；3–5 逐步加入同 subject companion；6–50 使用 50% self、30% 同 session 跨序列/互补 product、15% 同 acquisition 类型纵向 context、5% 同 session repeat。
- companion 只是可见 context。所有样本的 target 都是 anchor 自身的 EMA latent；不强迫跨时间、跨序列或跨 tracer latent 相等。
- validation 只记录健康度，永远不决定是否继续训练，也不按 validation performance 改变训练协议。只有无法恢复的结构/哈希/数值/CUDA OOM 错误才会停止训练。

这里所说的“无 QC”是指没有人工图像质量评分、主观 reliability 权重或质量阈值过滤。文件无法解码、仿射奇异、没有有限体素、注册程序报错等仍属于必要的结构错误：单个 case 会写入 failure ledger 后继续，其余数据不会因此停止预处理。

按来源工作目录当时的 bundled metadata、用户提供的完整 hierarchy 与最终 seed `20260822` 做的只读 planning audit 为：6,608 subjects、23,886 observations（MRI 14,628 / PET 9,258）；subject split 5,286 / 661 / 661，对应 observation split 19,171 / 2,401 / 2,314；legal same-subject relations 68,601，MRI–PET edge、subject leakage、PT030、cache entry、UNKNOWN sequence/tracer 与 locator mismatch 均为 0。这些数字是历史 planning reference，不是当前仓库内的数据清单、失败比例或 performance gate；workstation 会按运行时真实存在的文件重新 physical inner join 并在日志中冻结实际数量。

## 2. 代码仓库与本地资产边界

Git 仓库包含正式 pretraining 源码、配置、环境文件、分离的 inspect/preprocess/train 命令和对应测试。本次同步有意不包含以下本地或第三方资产：

- `metadata/` 下的 FOMO mapping/header/license/DUA 与 ADNI PET manifests；
- `external/sat3d/` 下的第三方 SAT3D Python source 和预训练权重；
- `external/reference/` 下的 1.5 mm source reference/mask；
- raw MRI/PET、cache、run、checkpoint 和分析产物。

运行前必须从已获授权的本地来源把这些资产放到两份 workstation YAML 声明的 project-relative 路径。预期文件、大小及 SHA-256 见 `BUNDLED_ASSETS.json`；inspect/preprocess/train 会 fail closed。该清单不代表对第三方数据、代码或权重授予新的再分发许可。

## 3. 磁盘与资源预留

23,886 个完整 case 若都成功，float16 image shards 约 93.30 GiB，bit-packed structural support 约 5.83 GiB，二者合计约 99.14 GiB。还要预留 catalog、reference/FEM、注册临时文件、日志和 checkpoint 空间；建议在 raw data 之外至少预留 250 GiB，若希望保留更多 milestone checkpoint 则预留 350–400 GiB。

第一轮默认使用一张 RTX PRO 6000 Blackwell 96 GB：physical micro-batch 8，gradient accumulation 2，global batch 16。真实显存和吞吐只能由 workstation 的真实 SAT3D/CUDA 路径确认；本地 tiny smoke 不能用于估计正式显存、耗时或效果。

## 4. 一次性环境配置

准备好上述本地资产后，在项目根目录运行：

```bash
bash scripts/bootstrap_workstation_env.sh
```

该命令创建/更新独立的 `foundation-model-test` conda 环境，安装 CUDA 12.8 PyTorch、SimpleITK、dcm2niix、Rich 及项目本身，并执行 `pip check`。最后只做 PyTorch/CUDA 版本、可见 GPU 与 BF16 的轻量环境探测；不创建模型，不做 forward/backward，也不会开始预处理或训练。

然后编辑 `configs/workstation_formal.yaml` 中两个 raw-data 占位路径：

```yaml
paths:
  fomo_root: /absolute/path/to/FOMO_MRI
  adni_pet_root: /absolute/path/to/ADNI_PET
```

其余资产路径是 project-relative，但资产本身不在 Git 中；必须另行提供与 `BUNDLED_ASSETS.json` 哈希一致的 metadata、SAT3D source/checkpoint 和 source reference/mask。

## 5. 先做只读检查

```bash
bash scripts/inspect_workstation.sh
```

inspect 会检查配置、raw roots、metadata、SAT3D checkpoint/source tree、reference、已有 catalog/cache/checkpoint 的 contract。若预处理产物已存在，它还会只读解析 catalog/relation、全部成功 receipt、每个 mmap shard header、reference receipt 和 FEM receipt；只有这些合同全部一致且三个 split 都仍有成功 case，`ready_for_training` 才会为 `true`。它不预处理图像，也不启动 CUDA 训练。

## 6. 独立运行可恢复预处理

```bash
bash scripts/preprocess_workstation.sh
```

默认 `--resume`。每个 case 使用确定性 shard/slot、原子 receipt 和 failure ledger；中断后再次执行同一命令即可。终端使用一个干净的 Rich live progress view，同时同步追加：

```text
workstation_cache/logs/preprocess.log
workstation_cache/logs/preprocess.metrics.jsonl
workstation_cache/failures.jsonl
```

预处理不会做主观 QC 或 performance threshold。成功 case 被冻结到 `cache_index.jsonl`；失败 case 会记录 dataset、source locator、exception type 和 message，然后继续。

若中断恰好发生在 catalog 多文件提交或 FEM/receipt 提交之间，`--resume` 只清理那组可识别的 partial generated files 后确定性重建；完整但合同不一致的产物不会被覆盖。

开发时可用 `foundation-preprocess --config configs/workstation_formal.yaml --limit 2` 检查控制流，但它必须写入独立的 development cache，且不能当作正式数据验证。

## 7. 可选的低资源代码 smoke

```bash
conda run --no-capture-output --name foundation-model-test \
  foundation-train --config configs/workstation_formal.yaml --smoke
```

这个路径只使用 tiny synthetic volumes/encoder，目的是检查 integration、loss、optimizer、checkpoint/resume 和日志。它不会读取正式 cache，不会加载真实 SAT3D，也不产生科学证据。

## 8. 独立启动或恢复正式训练

```bash
bash scripts/train_workstation.sh
```

默认 `--resume`。checkpoint 只在完整 optimizer step（gradient accumulation 边界）原子提交，并绑定 config、catalog、split、relation、cache、reference/FEM、SAT3D source/checkpoint 和 metadata vocabulary hashes。中断后重新执行相同命令，会从最近完整 checkpoint 的 next-anchor cursor 恢复；DataLoader prefetch 不会提前推进持久化 cursor。

训练终端会在 training、每 coverage 的 fixed monitor、每 5 coverage 的 full validation 和最终 held-out test 之间切换干净的 Rich 进度条。它展示 coverage/anchor 进度、ETA、total/MRI/PET/relation-type 分层的同一 self-JEPA loss、prediction/target effective rank、各 optimizer group LR、EMA momentum、gradient norm、anchor/encoded-volume throughput、loader wait、CPU RSS 和 GPU allocated/reserved/peak。有限 monitor subset 会确定性保持 MRI/PET 覆盖；validation/test 先在 subject 内平均，再严格用 MRI/PET 0.5/0.5 做 modality macro，不会在缺模态时静默改权重。这些数值仍只观察和记录。同步日志位于：

```text
runs/workstation_formal/logs/train.log
runs/workstation_formal/logs/train.metrics.jsonl
runs/workstation_formal/checkpoints/
```

正式进程固定完成配置中的 50 coverages。validation 数值、milestone/final checkpoint 记录、76 GiB VRAM soft warning 和其他非致命 warning 都不会触发 early stop。测试集只在固定训练 horizon 完成后读取一次，并有独立原子完成状态；再次 resume 不会重复读取。

### 已有 run 遇到 optional companion 可见 support 为空时

> 以下是历史 128-channel 补丁版本的操作记录，仅适用于其 manifest 精确匹配的旧源码。当前可配置通道版本不属于该旧 migration；不要将本节当作新版源码恢复旧 run 的授权。新的 384-channel run 请使用第 11 节。

不要删除或重建 workstation 上的 `workstation_cache/`、`runs/workstation_formal/`，也不要重新执行 preprocessing。保留服务器上已经改好的 `configs/workstation_formal.yaml`，只用本次补丁版本覆盖下列程序/合同文件：

```text
src/mri_pet_geomc/formal/config.py
src/mri_pet_geomc/formal/integration.py
src/mri_pet_geomc/formal/runtime/training.py
src/mri_pet_geomc/formal/workflows.py
scripts/check_resume_patch.py
scripts/check_resume_patch.sh
RESUME_COMPATIBILITY.json
BUNDLED_ASSETS.json
```

然后在同一个项目根目录先执行只读检查：

```bash
bash scripts/check_resume_patch.sh
```

预期输出至少包含：

```json
{
  "epochs": 50,
  "resume_compatible": true,
  "resume_mode": "pinned_pre_fix_checkpoint_migration"
}
```

该检查会验证补丁源码、固定 migration manifest、服务器现有 resolved config、run contract、latest pointer 和 checkpoint 大小；完整 checkpoint SHA-256 仍由真正的严格 resume 加载过程验证。检查成功后再执行：

```bash
bash scripts/train_workstation.sh
```

恢复从 `latest.json` 指向的最近一次完整 optimizer-step checkpoint 开始。崩溃前已经计算、但尚未进入该 checkpoint 的 coverage 片段会确定性重放；已提交的 coverage 不会重训。遇到本次情形时，anchor 和它自身的 EMA target 均保留，只丢弃当次不可用的 optional companion，relation 记录为 `same_observation`，并同步写入 `optional_companion_visible_support_empty` warning。正常 companion 路径不变。

补丁同时修正旧源码中把 warmup coverage 隐式除以 100 的假设；现在 `warmup_coverages: 2` 会严格按当前 `epochs: 50` 换算成两个完整 coverage 的 optimizer updates。这个 correction 会连同源码哈希迁移一起冻结到 `runs/workstation_formal/contracts/resume_source_compatibility.json`，不会静默发生。

## 9. 恢复与异常语义

- 正常中断：重新运行同一 preprocess/train 命令。
- checkpoint 是不可覆盖的 phase/commit 文件；若中断发生在 checkpoint 与 latest pointer 两次原子提交之间，旧 latest 仍可恢复。JSONL 只允许修复末尾未完成行，并写入 `journal_tail_recovered`；中段损坏会 fail closed。
- 配置、catalog、cache、SAT3D 或 geometry hash 改变：拒绝错误 resume；使用新的 cache/run root 启动新协议。
- 单个预处理 case 失败：写 ledger 并继续；修复外部文件/工具后可 resume 重试。
- CUDA OOM、non-finite loss/gradient、checkpoint corruption、模型 state mismatch：属于无法安全忽略的 fatal error。日志会在退出前同步；不要用放宽 performance gate 的方式掩盖这些错误。
- `--no-resume` 不会删除已有数据。若目标目录非空，应改用新的 `cache_root` 或 `run_root`，避免覆盖可恢复状态。

## 10. 结果解释边界

本项目的本地 unit/smoke tests 只证明代码契约和小型合成路径。本次交付没有在本机执行 24k preprocessing、真实 SAT3D 128³ forward/backward、96 GB GPU memory probe 或 50-coverage training，因此不能提前声称实际耗时、显存峰值、收敛或下游效果。

## 11. 独立的 384-channel 下一轮预训练

当前版本已将第一轮的固定值改为可配置参数：latent/condition 宽度、predictor 宽度/层数/heads/query 上限、activation checkpointing 开关、总 batch、零 warmup、checkpoint 保留数量/间隔、完整 validation 间隔和配准线程数。配置仍会检查正数/有限值、attention heads 整除、mask query 数一致，以及 microbatch × accumulation = global batch。GeoMC 内部宽度等继续从 YAML 传入。

`model.projector.output_dim` 可写为 `auto`、`null` 或省略，由 `model.latent_dim` 决定；若写了显式数字，必须与 latent_dim 一致。SAT3D 输出仍为 384，所以 `projector.input_dim` 不随 field 宽度改变。其他 `128`（例如输入边长、谱模态数量、metadata 宽度、query 数）不能全局替换。

本次没有改变模型 forward、EMA 范围或 GeoMC field 的位置。仍保留实际未实现替代路径的检查：SAT3D 的 128³/8³ 输入输出接口、现有同模态 companion curriculum、数据与 subject split 规则，以及 checkpoint/cache/hash 与防泄漏约束。不能仅删除这些检查后声称相应功能已实现。

这是一轮新的预训练，不是将旧的 128-channel checkpoint 扩维后继续训练。旧的 `configs/workstation_formal.yaml` 和 `scripts/train_workstation.sh` 保留不变；新配置为 `configs/workstation_formal_latent384.yaml`，结果单独写入 `runs/workstation_formal_latent384/`。

新配置仅改变 run 标识、latent/projector output 从 128 到 384、GeoMC hidden dimension 从 256 到 768，以及 physical microbatch/accumulation 的保守起点。SAT3D output 仍为 384，predictor 仍为 256 channels、6 blocks、8 heads，metadata condition 仍为 128；数据 split、已有 2 mm/128³ cache、masking、EMA 范围、loss、学习率、50 coverages 和 effective batch 16 均沿用原协议。通道增宽不等于提高图像空间分辨率，也不保证下游提升。

先将本次更新的项目源码、新配置和新脚本同步到 workstation；本地文件更新不代表已经同步到服务器。同步时保留服务器现有的 `workstation_cache/`、`runs/` 和旧配置中的实际数据路径，不要用本地空目录覆盖它们。将新配置中的 `fomo_root`、`adni_pet_root` 改成与服务器旧配置相同的真实路径；如果已有 cache 使用非默认位置，也保持相同的 `cache_root`。

在服务器的项目根目录重新注册当前源码，然后启动新 run：

```bash
cd ~/projects/foundation_model_test
conda run --no-capture-output --name foundation-model-test \
  python -m pip install --no-deps --editable .

bash scripts/train_workstation_latent384.sh
```

以上 wrapper 委托原有训练入口，保留 Rich 终端界面、同步日志、conda 环境选择和严格 `--resume`。也可使用等价的通用命令，或给新 wrapper 传入自己的配置：

```bash
bash scripts/train_workstation.sh configs/workstation_formal_latent384.yaml
# Optional: a separate, user-edited 384-channel configuration.
bash scripts/train_workstation_latent384.sh configs/my_latent384.yaml
```

已有 preprocessing/cache contract 一致时，无需重新预处理。只有修改了会影响 cache contract 的数据、reference、空间预处理等设置，才需要相应的新 cache；不要为了开始这次通道增宽实验删除现有 cache。

新配置采用 microbatch 4 × accumulation 4 = effective batch 16，作为单张 96 GB GPU 的保守显存起点，**尚未实测，不保证一定放得下**。如需改动，应在这轮首次启动前修改配置；保持 effective batch 16 可用 2 × 8。已经产生 checkpoint 后不要直接修改 batch 配置再强行 resume，而应保留旧 run 并使用新的 run root。

新 384-channel run 中断后，重复同一命令即可恢复它自己的 checkpoint。旧 128-channel run 不能作为这个 run 的 `--resume` 来源；不要复制旧 `latest.json` 或 checkpoint 到新目录。新版源码还会改变 training source hash，可能阻止旧 run 在未经授权迁移时恢复；不要删除或伪造 hash/contract 保护来绕过检查。需要恢复旧 run 时应使用其原始代码与配置，或另行提供明确、经验证的兼容迁移。此处没有授权旧 run 的源码哈希迁移。
