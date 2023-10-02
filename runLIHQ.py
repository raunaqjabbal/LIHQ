import os
import glob
import shutil
from pathlib import Path
import time
import subprocess
import argparse

from procedures.av_scripts import *
os.chdir('./first_order_model')
from LIHQ.procedures.fomm_scripts import FOMM_chop_refvid, FOMM_run
os.chdir('..')
from procedures.wav2lip_scripts import wav2lip_run
from first_order_model.demo import load_checkpoints
from procedures.qvi_scripts import qvi_config
os.chdir('./QVI')
from LIHQ.QVI.demo import main as qvi_main
os.chdir('..')



def run(face, audio_super = '/content/LIHQ/input/audio/', ref_vid = '/content/LIHQ/input/ref_vid/syn_reference.mp4', ref_vid_offset = [0], clear_outputs=True, save_path = None):

    #Miscellaneous things
    print("Initializing")
      #Turning face &offset to arrays as needed
    if not isinstance(face, list):
        face = [face]
    if not isinstance(ref_vid_offset, list):
        ref_vid_offset = [ref_vid_offset]


      #Deleteing output files
    if clear_outputs == True:
        for path in Path("./output").glob("**/*"):
            if path.is_file():
                path.unlink()

      #A/V Set up
    R1start = time.time()
    if audio_super[-1:] != '/':
        audio_super = audio_super + '/'
    aud_dir_names = get_auddirnames(audio_super)
    for adir in aud_dir_names:
        combine_audiofiles(adir, audio_super)

      #Expanding face array as needed
    while len(face) < len(aud_dir_names):
        face.append(face[0])

    #FOMM
      #Cropping reference video
    FOMM_chop_refvid(aud_dir_names, ref_vid, audio_super, ref_vid_offset)

      #Running FOMM (Mimicking facial movements from reference video)
    print("Running First Order Motion Model")
    generator, kp_detector = load_checkpoints(config_path='./first_order_model/config/vox-256.yaml', checkpoint_path='./first_order_model/vox-cpk.pth.tar')
    i = 0
    for adir in aud_dir_names:
        sub_clip = f'./first_order_model/input-ref-vid/{adir}/{adir}.mp4'
        FOMM_run(face[i], sub_clip, generator, kp_detector, adir, Round = "1")
        i+=1
    print("FOMM Success!")

    #Wav2Lip (Generating lip movement from audio)
    print("Running Wav2Lip")
    for adir in aud_dir_names:
        wav2lip_run(adir)
    w2l_folders = sorted(glob.glob('./output/wav2Lip/*'))
    if len(w2l_folders) < len(aud_dir_names):
        print('Wav2Lip could not generate at least one of your videos.\n'
            'Possibly bad audio, unrecognizable mouth, bad file paths, out of memory.\n'
            'Run below command in a separate cell to get full traceback.\n'
            '###########################################################\n'
            '###########################################################\n'
            'import os\n'
            'adir = \'Folder1\' # The audio folder that failed. See Wav2Lip output folder to see whats missing.\n\n'
            'vid_path = f\'''{os.getcwd()}/output/FOMM/Round1/{adir}.mp4\'''\n'
            'aud_path = f\'''{os.getcwd()}/input/audio/{adir}/{adir}.wav\'''\n'
            '%cd /content/LIHQ/Wav2Lip\n'
            '!python inference.py --checkpoint_path ./checkpoints/wav2lip.pth --face {vid_path} --audio {aud_path} --outfile /content/test.mp4  --pads 0 20 0 0\n\n'
            )
        sys.exit()
    else:
        print('Wav2Lip Complete')
