from src.autoroto.autoroto import AutoRoto
from src.posemocap.posemocap import PoseMoCap
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(module)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)
from pathlib import Path
from src.jobs.vrport_job import VRPortJob


if __name__ == '__main__':
    logger.info('Script Started.')
    video_path = "assets/upnatem1_trimmed.mp4"
    save_path = "assets/upnatem_cover_1/upnatem1_trimmed.png"
    job = VRPortJob(video_path, save_path, display_img=True)
    #ar = AutoRoto(job)
    #ar.rem_bg()
    pmc = PoseMoCap(job)
    pmc.get_poses()
    print('done.')