"""
paths.py — S3 Experiments v1 统一路径配置

所有脚本通过 `from utils.paths import *` 引用路径，
不在各脚本中硬编码绝对路径。修改数据位置时只需改此文件。
"""

from pathlib import Path

AIWORKBENCH = Path("/Volumes/aiworkbench")

# ── 数据目录 ──────────────────────────────────────────
DATASETS_ROOT       = AIWORKBENCH / "datasets"
RAW_SCANS           = DATASETS_ROOT / "01_raw_scans"
CLASSIFIED          = DATASETS_ROOT / "06_classified"
SUBSET_ARRANGEMENT  = CLASSIFIED / "subset_arrangement"
GOLDEN_TEST_SET     = DATASETS_ROOT / "07_golden_test_set"
OUTPUTS_3D          = DATASETS_ROOT / "08_3d_outputs"

# ── 代码目录 ──────────────────────────────────────────
PROJECT_ROOT        = AIWORKBENCH / "projects" / "ctfad-pipeline"
S3_V1_ROOT          = PROJECT_ROOT / "s3_experiments_v1"
S3_V0_ROOT          = PROJECT_ROOT / "s3_experiments_v0"

# ── Golden Test Set 子目录 ────────────────────────────
GTS_IMAGES          = GOLDEN_TEST_SET / "images"
GTS_ANNOTATIONS     = GOLDEN_TEST_SET / "annotations"
GTS_RMBG            = GOLDEN_TEST_SET / "images_rmbg"
GTS_REPORTS         = GOLDEN_TEST_SET / "reports"

# ── 3D 输出子目录（按 Phase 组织）────────────────────
OUTPUTS_PHASE3      = OUTPUTS_3D / "phase3_baseline"
OUTPUTS_PHASE4      = OUTPUTS_3D / "phase4_enhanced"
OUTPUTS_PHASE5      = OUTPUTS_3D / "phase5_decomposition"


def ensure_dirs():
    """创建所有必要的数据目录（幂等）"""
    for d in [
        GOLDEN_TEST_SET, GTS_IMAGES, GTS_ANNOTATIONS, GTS_RMBG, GTS_REPORTS,
        OUTPUTS_3D, OUTPUTS_PHASE3, OUTPUTS_PHASE4, OUTPUTS_PHASE5,
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("✅ 所有数据目录已就绪：")
    print(f"   Golden Test Set: {GOLDEN_TEST_SET}")
    print(f"   3D Outputs:      {OUTPUTS_3D}")
