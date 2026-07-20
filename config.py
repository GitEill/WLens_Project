import os

# -------- 路径设置 --------
RAW_DIR = "raw images"
OUTPUT_DIR = "preprocessed_results"
REPORT_PATH = "process_report.csv"

# -------- 图像格式 --------
IMAGE_EXTENSIONS = ('*.tif', '*.tiff', '*.TIF', '*.TIFF')
TARGET_SIZE = (512, 512)          # 统一 resize 尺寸
BATCH_SIZE = None                 # None 表示读取全部

# -------- 图像类型判定阈值 --------
CORNER_SIZE_RATIO = 10            # 角落区域取 min(50, h//10, w//10)
FLUORESCENT_CORNER_MEAN = 0.25
FLUORESCENT_GLOBAL_STD = 0.25
FLUORESCENT_LOW_INTENSITY_RATIO = 0.5
FLUORESCENT_WEIGHT_CORNER = 0.4
FLUORESCENT_WEIGHT_STD = 0.3
FLUORESCENT_WEIGHT_LOW = 0.3
FLUORESCENT_SCORE_THRESHOLD = 0.5

# -------- 预处理步骤默认顺序 --------
DEFAULT_ORDER = [
    'denoise',
    'background_correction',
    'illumination_correction',
    'deconvolution',
    'clahe'
]

# -------- 背景校正 --------
POLYFIT_DOWNSAMPLE = 0.25
ROLLING_BALL_RADIUS = 50

# -------- 去卷积 --------
PSF_SIGMA = 1.5
RL_ITERATIONS = 10
BLIND_PSF_ITERATIONS = 5
BLIND_REGULARIZATION = 1e-6

# -------- 去噪 --------
DENOISE_KERNEL_LARGE = (5, 5)     # noise_estimate > 0.05
DENOISE_KERNEL_SMALL = (3, 3)

# -------- CLAHE --------
CLAHE_CLIP_LIMIT_FLU = 2.0
CLAHE_CLIP_LIMIT_BRI = 1.5        # 明场通常不用，但可保留
CLAHE_TILE_GRID = (8, 8)

# -------- 评估指标判定阈值 --------
CONTRAST_RATIO_GOOD = 1.1
SHARPNESS_RATIO_GOOD = 1.1
NOISE_RATIO_GOOD = 0.9

# -------- 并行处理 --------
PARALLEL_WORKERS = 4