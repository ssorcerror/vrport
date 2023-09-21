from src.autoroto.autoroto import AutoRoto
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(module)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)
from pathlib import Path



if __name__ == '__main__':
    logger.info('Script Started.')
    video_path = "assets/upnatem1_trimmed.mp4"
    save_path = "assets/upnatem_cover_1/upnatem1_trimmed.png"
    ar = AutoRoto(video_path, save_path)
    ar.rem_bg()
    print('done.')