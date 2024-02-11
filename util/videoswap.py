'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:52
Description: 
'''
import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False, use_mask =False, start = 0.0, end = 1.0, start_1 = 0.0, end_1 = 1.0):
    video_forcheck = VideoFileClip(video_path)
    # audio check 
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck
    # if has audio
    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0
    # video frame number
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frames per second-fps = frame_num/duration
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    if use_mask: # use mask for better results
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

    # while ret:
    star_point = int(start*frame_count)
    end_point = int(end*frame_count)
    star_point_1 = int(start_1*frame_count)
    end_point_1 = int(end_1*frame_count)
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read() # read one frame 
        # swap part of video
        if frame_index in range (star_point, end_point) or frame_index in range (star_point_1, end_point_1):

            if  ret:
                
                detect_results = detect_model.get(frame,crop_size)

                if detect_results is not None:
                    # print(frame_index)
                    if not os.path.exists(temp_results_dir):
                            os.mkdir(temp_results_dir)
                    frame_align_crop_list = detect_results[0]
                    frame_mat_list = detect_results[1]
                    swap_result_list = []
                    frame_align_crop_tenor_list = []
                    for frame_align_crop in frame_align_crop_list:

                        # BGR TO RGB
                        # frame_align_crop_RGB = frame_align_crop[...,::-1]

                        frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                        swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                        cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                        swap_result_list.append(swap_result)
                        frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                        

                    reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                        os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)

                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    if not no_simswaplogo:
                        frame = logoclass.apply_frames(frame)
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
            else:
                break
        else:
            if not os.path.exists(temp_results_dir):
                os.mkdir(temp_results_dir)
            frame = frame.astype(np.uint8)
            if not no_simswaplogo:
                frame = logoclass.apply_frames(frame)
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)


    clips.write_videofile(save_path,audio_codec='aac',)

