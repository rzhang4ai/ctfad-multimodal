# S2 阶段总结：数据分类与本地 VLM 工作流

**文档性质：** 阶段性总结 + 经验备案
**编写日期：** 2026-02-24
**阶段范围：** S2 Step A～Step E（环境准备 → Round 1 全量分类）
**研究者：** MSc 学生，Vibe Coder（Cursor + Cline 为主力开发方式）

---

## 一、本阶段完成的工作

### 1.1 工作流程总览

```
Step A  环境准备
  └─ ollama pull qwen2.5vl:3b（3.2GB，M4主力分类模型）
  └─ pip install rembg（S3背景去除准备）

Step B  数据合并
  └─ s2_merge_pairs.py
  └─ 产出：merged_pairs.jsonl（432条，含book_id + global_pair_id）

Step C  项目工程化
  └─ GitHub仓库建立（ctfad-multimodal，Public）
  └─ 本地项目目录：/Volumes/aiworkbench/projects/ctfad-pipeline/
  └─ 配置文件：image_text_categories.yaml，NAMING_CONVENTIONS.md

Step D  沙盒测试（40张）→ Prompt迭代 → 验证
  └─ s2_classify_sandbox.py（v1）→ 准确率 60%
  └─ s2_error_analysis.py → 定位问题
  └─ Prompt优化（v2）→ 准确率 75%
  └─ s2_analyze_sandbox.py → 生成分析报告

Step E  全量运行（进行中）
  └─ s2_classify_full.py → 392条（沙盒40条自动跳过）
```

### 1.2 核心产出文件

| 文件 | 位置 | 说明 |
|---|---|---|
| `merged_pairs.jsonl` | `06_classified/` | 432条合并图文对 |
| `image_text_categories.yaml` | `config/` | 图像内容分类体系 v1.0 |
| `NAMING_CONVENTIONS.md` | `docs/` | 项目命名规范 |
| `sandbox_predictions.jsonl` | `06_classified/sandbox/` | 40条沙盒预测结果（v2 prompt）|
| `sandbox_reviewed.jsonl` | `06_classified/sandbox/` | 40条人工审核标注 ⭐ |
| `sandbox_analysis_report.md` | `06_classified/sandbox/` | 准确率分析报告 |
| `error_analysis.html` | `06_classified/sandbox/` | 错误案例可视化报告 |
| `full_predictions.jsonl` | `06_classified/` | 全量分类结果（运行中）|

### 1.3 分类体系 v1.0

**Round 1 粗分类（6类）：**

| 标签 | 含义 | 沙盒准确率 |
|---|---|---|
| `arrangement` | 插花作品图（实拍/手绘/古画）| 87% |
| `vessel` | 花器单独特写 | 25% ⚠️ |
| `plants` | 花材/植物素材图 | 80% |
| `photograph` | 其他实拍照片 | 0% ⚠️ |
| `diagram` | 示意图/步骤图 | 83% |
| `other` | 无法归类 | — |

**真实数据分布（40张样本）：**
- `arrangement` 60% / `plants` 15% / `diagram` 15% / `other` 5% / `vessel` 2.5% / `photograph` 2.5%

**决策依据：** `vessel` 和 `photograph` 准确率虽低，但两者合计仅占约5%（约20条），人工审核兜底成本极低，不影响整体质量，决定直接全量运行。

---

## 二、遇到的问题与解决方案

### 2.1 GitHub 推送认证问题

**问题：** GitHub 不再支持账号密码推送，报 `Authentication failed`。

**解决方案：**
1. 在 GitHub → Settings → Developer settings → Personal access tokens 生成 PAT
2. push 时 Username 填 GitHub 用户名，Password 填 PAT（不是账号密码）
3. `git config --global credential.helper osxkeychain` 让 Mac 记住 token

**经验：** PAT 是一次性配置，之后 Mac 钥匙串自动处理，无需再手动输入。

### 2.2 Git 分支名冲突（master vs main）

**问题：** 本地 `git init` 默认创建 `master` 分支，GitHub 默认用 `main`，push 失败。

**解决方案：**
```bash
git branch -m master main          # 本地分支改名
git config pull.rebase false       # 设置合并策略
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### 2.3 .gitignore 文件名丢失

**问题：** 从浏览器下载 `.gitignore`，Mac 把文件名保存为 `gitignore`（去掉了点）。

**解决方案：**
```bash
cp ~/Downloads/gitignore /path/to/project/.gitignore
# cp 命令支持复制时重命名，一步完成
```

### 2.4 Round 1 分类准确率低（60%→75%）

**问题：** 初版 prompt 准确率只有 60%，主要混淆对：
- `vessel` → 应为 `diagram`（4次）：步骤图中花器占主体，模型抓住器形特征
- `diagram` → 应为 `arrangement`（4次）：速写风格插花图，模型看图像判成作品图

**根本原因：** 模型没有充分利用图注文字信息（"步骤2"、"速写作品赏析"等），只依赖视觉判断。

**解决方案：** 改进 prompt，加入文字线索优先规则：
```
规则1：图注含「步骤」字样 → 优先选 diagram（无论图像内容）
规则2：图注含「赏析」「速写」字样 → 优先选 arrangement
规则4：vessel 严格限定为空花器，步骤图中的花器不算 vessel
```

**效果：** 准确率从 60% 提升到 75%，`diagram` 准确率从 29% → 83%。

**判断方法（prompt问题 vs 模型能力问题）：**
- 错误图片图注文字有明确线索但模型忽略 → **prompt 问题**（可修复）
- 图片本身模糊/类别边界模糊，即使换强模型也判错 → **模型能力/数据问题**
- 改 prompt 后同张图判对了 → 确认是 prompt 问题

---

## 三、本地 VLM 直连方式经验总结

> 这是本阶段发现的重要经验，对后续所有本地模型调用均适用。

### 3.1 直连 Ollama vs 通过 One API 网关

本阶段所有分类脚本均采用**直接调用本地 Ollama API**的方式，绕过了 One API 网关：

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b"
```

**对比分析：**

| 维度 | 直连 Ollama | 通过 One API |
|---|---|---|
| 配置复杂度 | ✅ 零配置，开箱即用 | ❌ 需配置通道、API key映射 |
| 调用稳定性 | ✅ 本地直连，无中间层 | ⚠️ 多一层转发，多一层故障点 |
| 调试难度 | ✅ 报错直接来自 Ollama | ❌ 报错可能来自网关层 |
| 适用场景 | 本地模型批量任务 | 多模型统一管理、云端API切换 |
| 切换模型 | 改一行 MODEL_NAME | 改 One API 配置 |

**结论：** 对于**固定使用本地模型的批量任务**（如本阶段的分类），直连 Ollama 更简单、更稳定。One API 的价值在于需要**同时管理多个云端 API**或**在本地/云端之间动态切换**的场景。

### 3.2 三条铁律（M4 16GB 硬件约束下）

详见 S2 备案文档第三节，核心摘要：

```python
# 铁律1：强制冷却
time.sleep(2.5)                    # 每次调用后
if (i+1) % 20 == 0: time.sleep(10) # 每20次额外冷却

# 铁律2：图片预压缩
max_size = 768  # 最大边长，JPEG质量85

# 铁律3：断点续传
if gid in done_ids: continue       # 跳过已处理
out_f.write(json.dumps(record) + "\n")
out_f.flush()                      # 处理一条写一条
```

### 3.3 模型选择实践结论

本阶段实测 `qwen2.5vl:3b`（3.2GB）在 M4 16GB 上：
- 推理速度：约 5-8 秒/张（含 2.5 秒冷却）
- 内存占用：约 2.5-3.5GB
- 可与 Docker 服务（One API/Label Studio）共存
- **适合：** 全量批量任务（每条都要过模型）
- **不适合：** 需要深度垂类语义理解的任务（用云端 Qwen-VL-Max）

---

## 四、沙盒测试工作流（可复用方法论）

本阶段建立了一套标准化的沙盒测试→优化→全量运行流程，适用于所有后续分类任务：

```
1. 抽样（40条，按书均匀分布，覆盖不同页码段）
        ↓
2. 跑模型，生成预测结果
        ↓
3. 人工审核界面（sandbox_review.html）
   - 每张图显示：图片 + 预测标签 + 置信度 + 图注文本
   - 操作：✅正确 / ❌错误+选正确类别
   - 导出：sandbox_reviewed.jsonl
        ↓
4. 自动分析（s2_analyze_sandbox.py）
   - 总体准确率
   - 各类别准确率
   - 混淆矩阵
   - Prompt 优化建议
        ↓
5. 错误案例分析（s2_error_analysis.py）
   - 可视化每张错误图
   - 诊断框：prompt问题 vs 模型能力问题
        ↓
6. 决策
   ├─ 准确率 ≥ 85%：直接全量运行
   ├─ 70-85%：评估错误分布，低频类别可接受则直接全量
   └─ < 70%：优化 prompt 或换模型，重跑沙盒
        ↓
7. 全量运行（沙盒40条自动跳过，断点续传）
```

**关键经验：** 沙盒的人工标注结果（`sandbox_reviewed.jsonl`）应永久保留，在全量结果合并时直接复用，避免重复审核。

---

## 五、下一步（S2 剩余工作）

```
当前：全量 Round 1 运行中（392条，约1.5小时）

完成后：
Step E-2  合并沙盒标注
  └─ s2_merge_results.py
  └─ 产出：all_predictions.jsonl（432条，含沙盒人工标注）

Step E-3  人工审核全量结果
  └─ 全量审核界面（复用/扩展 sandbox_review.html）
  └─ 40条沙盒已审核条目：预填标注，快速确认
  └─ 392条新条目：正常审核
  └─ 产出：classified_final.jsonl

Step F    Round 2 细分类
  └─ 只对需要细分的类别跑（arrangement/diagram/vessel/plants）
  └─ 预计准确率更高（类别更少、更具体）

Step G    子集提取 + 3D 准备
  └─ 提取 arrangement_photo + arrangement_drawing
  └─ rembg 背景去除
  └─ 图片质量评估
```

---

## 六、Git 提交记录（本阶段）

| Commit | 内容 |
|---|---|
| `config: initialize project structure and categories v1.0` | 项目初始化 |
| `feat(s2): add merge_pairs script — 432 pairs from 3 books` | 合并脚本 |
| `feat(s2): add sandbox classification pipeline` | 沙盒分类三件套 |
| `feat(s2): add full classification script round1` | 全量分类脚本 |

---

*本文档在 S2 阶段完成后归档至 `docs/`，后续阶段另建文档。*
