# Robot-boxing

将双人拳击视频转换为可在 [AITViewer](https://github.com/eth-ait/aitviewer) 中回放的双人 SMPL-X 参数序列。

## 主要功能

- Multi-HMR 单帧 SMPL-X 姿态估计。
- YOLO + ByteTrack 多人跟踪，并优先选择持续运动、处于画面主体位置且保持拳击防守姿态的两名拳手。
- 轨迹拼接和双人身份稳定，减少转身、交叉时的身份交换。
- 每名拳手使用固定 shape，双手使用固定握拳姿态。
- 四元数时序滤波、旋转速度限制和根位移跳变修复。
- 根据二维出拳接触锚点修正三维地面平面位置，减少“正面看命中、俯视图打空”的问题。
- 使用 SMPL-X 躯干、头部、骨盆和腿部胶囊检查身体互穿；拳头和手臂不参与穿模删除判定。
- 无法修复的空间错位或身体穿模区间会被删除，并保存为多个连续子片段，避免跨区间拼接产生瞬移。
- 输出自动质量报告 `selection_report.json`。

当前流水线版本：`boxing_smplx_v4_spatial_contact`。

## 仓库结构

```text
scripts/
  multi_hmr_boxing_video.py       单视频转换核心流程
  batch_boxing_smplx.py           按 altview manifest 批量转换
  play_boxing_smplx.py            AITViewer 双人回放
  run_batch_boxing_smplx.sh       批处理启动脚本
  run_boxing_smplx_glfw.sh        AITViewer GLFW 启动脚本
  temporal_refine_smplx.py        姿态时序平滑工具
  remove_translation_jumps.py     根位移跳变修复工具
  apply_smplx_ground_offsets.py   地面高度修正工具
  prepare_teacher_demo.py         演示片段整理工具
```

## 外部依赖

本仓库不包含数据集、模型权重或 SMPL-X 模型文件。运行前需要自行准备：

- Python 3.12 环境；
- PyTorch（CUDA 版本）；
- Multi-HMR 代码与 `multiHMR_672_S.pt`；
- Ultralytics YOLO 权重；
- SMPL-X neutral 模型；
- AITViewer。

Python 包参考 [requirements.txt](requirements.txt)。SMPL-X 模型受其官方下载许可约束，不应提交到公开仓库。

当前脚本会自动查找：

```text
/media/ubuntu22/Elements
/media/ubuntu22/Elements1
```

如果推理 Python 不在默认位置，可设置：

```bash
export BOXING_PYTHON=/path/to/python
```

回放环境不在 `/home/ubuntu22/aitviewer` 时，可设置：

```bash
export AITVIEWER_ROOT=/path/to/aitviewer
```

## 输入格式

批处理会扫描：

```text
处理后的boxing视频/**/altview/manifest.json
```

manifest 中的 `clips[].file_name` 应指向对应视频。

## 批量转换

```bash
cd scripts

./run_batch_boxing_smplx.sh \
  "/media/ubuntu22/Elements/处理后的boxing视频" \
  --output-root "/media/ubuntu22/Elements/smplx文件"
```

先测试一段：

```bash
./run_batch_boxing_smplx.sh \
  "/media/ubuntu22/Elements/处理后的boxing视频" \
  --output-root "/media/ubuntu22/Elements/smplx文件" \
  --limit 1 \
  --force
```

缺少当前流水线版本号或 QC 未通过的旧结果不会被跳过。

## 输出格式

每个通过质量检查的 clip 目录包含：

```text
person_1_smplx.npz
person_2_smplx.npz
selection_report.json
conversion.log
segments/
  segment_001/
    person_1_smplx.npz
    person_2_smplx.npz
```

NPZ 主要字段包括：

- `global_orient`
- `body_pose`
- `left_hand_pose`
- `right_hand_pose`
- `jaw_pose`
- `transl`
- `betas`
- `fps`
- `source_joints_2d`

## AITViewer 回放

```bash
cd scripts

./run_boxing_smplx_glfw.sh \
  "/media/ubuntu22/Elements/smplx文件/50/clip_000001"
```

回放某个连续子片段：

```bash
./run_boxing_smplx_glfw.sh \
  "/media/ubuntu22/Elements/smplx文件/50/clip_000001/segments/segment_001"
```

## 数据质量说明

自动 QC 能排除大量身份交换、跳变、身体互穿和二维/三维接触不一致，但单目三维人体重建无法提供绝对的物理真值。正式发布数据集前，仍建议对通过结果进行抽样多视角回放检查，并将 QC 失败目录隔离。

