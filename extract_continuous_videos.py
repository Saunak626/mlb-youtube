import os
import json
import string
import random
import subprocess
import multiprocessing
from tqdm import tqdm

def local_clip(filename, start_time, duration, output_filename, output_directory):
    end_time = start_time + duration
    command = ['ffmpeg',
               '-i', '"%s"' % filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'copy', '-an',
               '-threads', '1',
               '-loglevel', 'panic',
               os.path.join(output_directory,output_filename)]
    command = ' '.join(command)

    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(err.output)
        return err.output


input_directory = 'videos/'
output_directory = 'data/continuous_videos/'


def wrapper(clip_item):
    clip_id, clip_data = clip_item
    duration = clip_data['end']-clip_data['start']
    filename = clip_data['url'].split('=')[-1]
    output_filename = clip_id + '.mp4'
    
    # Check if the file already exists to implement checkpointing
    if os.path.exists(os.path.join(output_directory, output_filename)):
        return 0 # Skip if already processed

    local_clip(os.path.join(input_directory,filename+'.mp4'), clip_data['start'], duration, output_filename, output_directory)
    return 0
    

with open('data/mlb-youtube-continuous.json', 'r') as f:
    data = json.load(f)
    os.makedirs(output_directory, exist_ok=True)
    
    clips_to_process = list(data.items())

    if not clips_to_process:
        print("No clips found in JSON file.")
    else:
        print(f"Total clips to process: {len(clips_to_process)}")

        # Automatically determine the number of processes based on CPU cores
        num_processes = multiprocessing.cpu_count()
        print(f"Using {num_processes} processes for video segmentation.")

        pool = multiprocessing.Pool(processes=num_processes)
        # Using tqdm to show progress
        with tqdm(total=len(clips_to_process), desc="Processing Continuous Videos") as pbar:
            for _ in pool.imap_unordered(wrapper, clips_to_process):
                pbar.update(1)
        
        pool.close()
        pool.join()
        print("Video segmentation complete.")
    
