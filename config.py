import os

# -------- Pathway --------
RAW_DIR = "raw images"
OUTPUT_DIR = "preprocessed images"
REPORT_PATH = "process_report.csv"

# -------- Image format --------
IMAGE_EXTENSIONS = ('*.tif', '*.tiff', '*.TIF', '*.TIFF')
TARGET_SIZE = (512, 512)          
BATCH_SIZE = None                 

# -------- Image type determing --------
CORNER_SIZE_RATIO = 10            # 角落区域取 min(50, h//10, w//10)
FLUORESCENT_CORNER_MEAN = 0.25
FLUORESCENT_GLOBAL_STD = 0.25
FLUORESCENT_LOW_INTENSITY_RATIO = 0.5
FLUORESCENT_WEIGHT_CORNER = 0.4
FLUORESCENT_WEIGHT_STD = 0.3
FLUORESCENT_WEIGHT_LOW = 0.3
FLUORESCENT_SCORE_THRESHOLD = 0.5

# -------- Flagging --------
QF_S = 83
CV_S = 0.15
BF_BC_S = (1.3 , 0.05)
FLU_BC_S = (1.5, 0.08)
LAP_VAR_T = 100
TENE_T = 500
SNR_S = 25
DENOISE_T_FLU = 0.02
NEED_CLACHE_BRI = False
CLAHE_L = (1.2, 1.5)



# -------- Preprocessing order --------
DEFAULT_ORDER = [
    'denoise',
    'background_correction',
    'illumination_correction',
    'deconvolution',
    'clahe'
]

# -------- Preprocessing Tags --------
PROCESS_TAGS = {
    'image_type': 'flu',
    'need_illumination_correction': False, 'need_deconvolution': False,
    'need_clahe': False, 'need_denoise': False,
    'need_background_correction': False,
    'metrics':{'noise_estimate': 0.03}}


# -------- Background Correction --------
POLYFIT_DOWNSAMPLE = 0.25
ROLLING_BALL_RADIUS = 50

# -------- Deconvolution --------
PSF_SIGMA = 1.5
RL_ITERATIONS = 5
BLIND_PSF_ITERATIONS = 5
BLIND_REGULARIZATION = 1e-6
REG_TYPE = 'tikhonov'
DECON_M = 'RL'
GAUSSIAN_SHAPE = (15,15)

#--------- Preprocessing -----------
AUTOMODE = True

# -------- Denoise --------
DENOISE_KERNEL_LARGE = (5, 5)     # noise_estimate > 0.05
DENOISE_KERNEL_SMALL = (3, 3)
# -------- CLAHE --------
CLAHE_TILE_GRID = (10, 10)



# -------- eva matrics threshold --------

CONTRAST_RATIO_GOOD = 1.1
SHARPNESS_RATIO_GOOD = 1.1
NOISE_RATIO_GOOD = 0.9

# -------- parallel process --------
PARALLEL_WORKERS = 4

DEFAULTS = {
    'QF_S': QF_S,
    'CV_S': CV_S,
    'BF_BC_S': BF_BC_S, 
    'FLU_BC_S': FLU_BC_S, 
    'LAP_VAR_T': LAP_VAR_T,
    'TENE_T': TENE_T,
    'SNR_S': SNR_S,
    'CLAHE_L': CLAHE_L,                    # tuple
    'CLAHE_TILE_GRID': CLAHE_TILE_GRID,    # tuple
    'ROLLING_BALL_RADIUS': ROLLING_BALL_RADIUS,
    'POLYFIT_DOWNSAMPLE': POLYFIT_DOWNSAMPLE,
    'PSF_SIGMA': PSF_SIGMA,
    'RL_ITERATIONS': RL_ITERATIONS,
    'BLIND_PSF_ITERATIONS': BLIND_PSF_ITERATIONS,
    'REGULARIZATION': BLIND_REGULARIZATION,
    'DECON_M': DECON_M,
    'GAUSSIAN_SHAPE': GAUSSIAN_SHAPE,
    'REG_TYPE': REG_TYPE,
    'DENOISE_KERNEL_LARGE': DENOISE_KERNEL_LARGE,
    'DENOISE_KERNEL_SMALL': DENOISE_KERNEL_SMALL,
    'DENOISE_THRESHOLD': DENOISE_T_FLU,
    'TARGET_SIZE': TARGET_SIZE,
}

def make_params(overrides=None, **kwargs):
    """generate a copy of dict, allows to override on time"""
    params = DEFAULTS.copy()
    if overrides:
        params.update(overrides)
    params.update({k: v for k, v in kwargs.items() if v is not None})
    return params