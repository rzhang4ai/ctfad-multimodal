# S3 Experiments v1 — 3D 生成探索性技术评估

CTFAD 项目第三阶段：系统评估 Image→3D 生成技术在中国传统插花领域的可行性。

## 目录结构

```
s3_experiments_v1/
├── README.md                  # 本文件
├── golden_test_set/           # Golden Test Set 构建（选图、质量评估、标注）
├── phase1_understanding/      # Phase 1: VLM 理解力评估（文→文、图→文）
├── phase2_2d_generation/      # Phase 2: 2D 生成增强（文→图、图→图）
├── phase3_3d_baseline/        # Phase 3: 3D 基线测试（SDF-based vs 3DGS-based）
├── phase4_enhanced_path/      # Phase 4: 增强路径（增强图→3D、多视图→3D、LoRA）
├── phase5_decomposition/      # Phase 5: 部件分解路径（SAM2 分割 + 分部件生成）
├── phase6_analysis/           # Phase 6: 深度分析（失败模式分类、方法论对比）
├── evaluation/                # 通用评估工具（技术指标、视觉评分、领域评分）
├── utils/                     # 公共工具函数（路径管理、图片处理、API 调用）
└── docs/                      # 代码层面的文档和实验日志
```

## 数据位置

- **输入数据（源）：** `/Volumes/aiworkbench/datasets/06_classified/subset_arrangement/`
- **Golden Test Set：** `/Volumes/aiworkbench/datasets/07_golden_test_set/`
- **3D 输出：** `/Volumes/aiworkbench/datasets/08_3d_outputs/`

代码中通过相对路径或配置文件引用数据，数据本身不入 Git。

## 环境

```bash
conda activate ctfad-s3-exp   # 待创建
# 或临时使用
conda activate ctfad-s3
```

## 上游遗产

从 `s3_experiments_v0/` 复用的脚本已放在 `golden_test_set/` 目录下：
- `s3_assess_image_quality.py` — 图像质量评估（路径已更新为 v1）
- `s3_remove_background.py` — 去背景工具（路径已更新为 v1）

原始版本保留在 `s3_experiments_v0/`，不修改。

## 设计文档

详细实验设计见 Obsidian 笔记：`obsidian/03.S3_experiments_v1/00.S3 Experiments_v1.md`
