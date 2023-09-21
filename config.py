from pathlib import Path
import os

HOME_DIR = os.getcwd()
MODELS_DIR = Path(HOME_DIR,'models')
MODEL_PATH = Path(MODELS_DIR, 'deeplabv3.tflite')
ASSETS_PATH = Path(HOME_DIR,'assets')
AUTOROTO_PATH = Path(ASSETS_PATH, 'autoroto_output')
