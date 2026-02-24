# CTFAD Project — 命名规范与术语表

**文档性质：** 项目管理参考文档（永久维护）
**创建日期：** 2026-02-24
**适用范围：** 所有代码、文件名、目录名、Git commit、论文写作

---

## 一、项目全称与缩写

| 层级 | 内容 |
|---|---|
| **中文全称** | 中国传统插花多模态数据集研究管线 |
| **英文全称** | Chinese Traditional Flower Arrangement Dataset — Multimodal Pipeline |
| **标准缩写** | **CTFAD** |
| **GitHub 仓库名** | `ctfad-multimodal` |
| **本地项目目录** | `/Volumes/aiworkbench/projects/ctfad-pipeline/` |

### 为什么用 "Chinese Traditional Flower Arrangement"

这是中国插花花艺联合会（CFAF / China Flower Arrangement Federation）的官方英文译名，也是中国插花领域国际交流时的标准表达。在论文、数据集描述、对外介绍时统一使用此表达，避免与日式花道（Ikebana）混淆。

---

## 二、目录命名规范

### 2.1 数据集目录（`/Volumes/aiworkbench/datasets/`）

| 目录名 | 内容 | 命名规则 |
|---|---|---|
| `01_raw_scans/` | 原始 PDF 扫描件 | 两位数字前缀，表示 Pipeline 阶段 |
| `02_layout/` | MinerU 版面分析输出 | 同上 |
| `04_structured/` | 图文对 JSONL + 裁剪图片 | 跳过 03 留给未来的清洗阶段 |
| `06_classified/` | S2 分类结果 | 跳过 05 留给未来的 embedding |
| `book_001/` ~ `book_030/` | 单本书数据 | 三位数字，固定宽度，方便排序 |

> ⚠️ `book_003_t` 是历史遗留命名（`_t` 表示测试），在 S2 合并时统一处理为 `book_003`。

### 2.2 代码项目目录（`/Volumes/aiworkbench/projects/ctfad-pipeline/`）

```
ctfad-pipeline/
├── config/          ← 配置文件（yaml, json），全部入 Git
├── docs/            ← 项目文档（操作手册、备案文档）
├── s1_extraction/   ← S1 阶段脚本：MinerU → Label Studio → 图文对
├── s2_classification/ ← S2 阶段脚本：分类、合并、子集提取
├── s3_3d_prep/      ← S3 阶段脚本：背景去除、Prompt 工程、3D 准备
└── tests/           ← 沙盒测试脚本（20 张图验证用）
```

---

## 三、文件命名规范

### 3.1 配置文件（`config/`）

| 文件名 | 用途 |
|---|---|
| `image_text_categories.yaml` | **图像内容类别**定义（本文档配套） |
| `domain_taxonomy.yaml` | 插花领域知识分类体系（形式/器型/花材/比例等）— S2 后期启用 |
| `model_config.yaml` | 模型选择与参数配置（Ollama 端点、云端 API 等）|
| `path_config.yaml` | 本地路径配置（数据集根目录等，不入 Git，用 `.gitignore` 排除）|

> ⚠️ `path_config.yaml` 包含本机绝对路径，**不入 Git**，在 `.gitignore` 中排除。

### 3.2 脚本文件

命名格式：`{阶段}_{动词}_{对象}.py`

| 示例文件名 | 含义 |
|---|---|
| `s1_extract_layout.py` | S1 阶段，提取版面信息 |
| `s2_merge_pairs.py` | S2 阶段，合并图文对 |
| `s2_classify_round1.py` | S2 阶段，Round 1 粗分类 |
| `s2_classify_round2.py` | S2 阶段，Round 2 细分类 |
| `s2_extract_subset.py` | S2 阶段，子集提取 |
| `s3_remove_background.py` | S3 阶段，背景去除 |

### 3.3 数据文件（JSONL）

| 文件名 | 位置 | 说明 |
|---|---|---|
| `image_text_pairs.jsonl` | `04_structured/book_XXX/` | 自动生成的原始图文对 |
| `image_text_pairs_edited.jsonl` | `04_structured/book_XXX/` | **人工审核后的最终版本**（优先使用） |
| `merged_pairs.jsonl` | `06_classified/` | 多书合并后的全量数据（含 book_id） |
| `round1_coarse.jsonl` | `06_classified/raw_predictions/` | Round 1 分类结果 |
| `round2_fine.jsonl` | `06_classified/raw_predictions/` | Round 2 分类结果 |
| `classified_final.jsonl` | `06_classified/reviewed/` | 人工审核后的最终分类 |

---

## 四、字段命名规范（JSONL 数据字段）

| 字段名 | 类型 | 说明 |
|---|---|---|
| `global_pair_id` | str | 全局唯一 ID，格式：`{book_id}_{local_pair_id}`，如 `b001_0042` |
| `book_id` | str | 书籍编号，格式：`b001` ~ `b030` |
| `image_path` | str | 图片绝对路径（本机路径，不入 Git） |
| `caption` | str | 图片说明文字 |
| `context` | str | 图片周边的正文文本 |
| `coarse_label` | str | Round 1 粗分类结果（top-level label_id） |
| `fine_label` | str | Round 2 细分类结果（leaf-level label_id） |
| `confidence` | float | 模型置信度（0.0 ~ 1.0） |
| `review_status` | str | 人工审核状态：`pending` / `approved` / `rejected` / `corrected` |
| `domain_tags` | dict | 领域标签（S2 后期）：形式、器型、花材等 |

---

## 五、Git Commit 规范

格式：`{类型}({范围}): {简短描述}`

| 类型 | 使用场景 | 示例 |
|---|---|---|
| `feat` | 新增功能或脚本 | `feat(s2): add merge_pairs script` |
| `config` | 修改配置文件 | `config: update image_text_categories v1.1` |
| `fix` | 修复 bug | `fix(s1): handle missing image path` |
| `docs` | 更新文档 | `docs: add S2 planning notes` |
| `test` | 沙盒测试相关 | `test: sandbox classification 20 images` |
| `refactor` | 重构代码（不改功能） | `refactor(s2): split classify into two rounds` |
| `data` | 数据处理记录（仅描述，不含数据） | `data: completed book_004 S1 pipeline` |

---

## 六、模型与 API 标准称谓

在代码注释、文档、commit message 中统一使用以下称谓：

| 称谓 | 指代 |
|---|---|
| `qwen-vl-3b` | Ollama 本地 qwen2.5-vl:3b（S2 主力分类模型）|
| `qwen-vl-7b` | Ollama 本地 qwen2.5-vl:7b（S1 caption 嗅探）|
| `qwen-vl-max` | 阿里百炼云端 Qwen-VL-Max（垂类术语理解）|
| `gemini-flash` | Google Gemini 1.5 Flash（长文本处理）|
| `claude-sonnet` | Anthropic Claude Sonnet（高精度视觉推理）|

---

## 七、版本控制原则

1. **数据永远不入 Git**：`datasets/` 目录与 Git 仓库物理分离，`.gitignore` 作为第二道防线排除所有 `.jsonl`、`.json`（数据）、图片文件
2. **配置文件全部入 Git**：`config/*.yaml` 是代码的一部分，必须版本控制
3. **本机路径配置不入 Git**：`path_config.yaml` 含绝对路径，加入 `.gitignore`
4. **每个 S 阶段完成后打 tag**：`git tag s1-complete`、`git tag s2-complete` 等，方便回溯

---

*本文档随项目演进持续更新。重大命名变更需同步更新本文档。*
