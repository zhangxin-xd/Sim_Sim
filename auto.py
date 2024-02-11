import os 
cmd = 'python test_video_swapsingle.py \
--crop_size 224 --name people \
--Arc_path arcface_model/arcface_checkpoint.tar \
--pic_a_path ./demo_file/Iron_man.jpg \
--video_path ./test_sample/3de6ea4b2f5fc722.mp4 \
--output_path ./output/3de6ea4b2f5fc722.mp4 \
--temp_path ./temp_results '
os.system(cmd)