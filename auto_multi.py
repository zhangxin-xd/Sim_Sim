import os 
# multi swap to one face
# cmd = 'python test_video_swapmulti.py \
# --crop_size 224 --name people \
# --Arc_path arcface_model/arcface_checkpoint.tar \
# --pic_a_path ./demo_file/Iron_man.jpg \
# --video_path ./test_sample/dd080990df822892.mp4 \
# --output_path ./output/dd080990df822892.mp4 \
# --temp_path ./temp_results \
# --start 0.2 --end 0.7 \
# --start_1 1.0 --end_1 1.0 '
# os.system(cmd)

# cmd = 'python test_video_swapmulti.py \
# --crop_size 224 --name people \
# --Arc_path arcface_model/arcface_checkpoint.tar \
# --pic_a_path ./demo_file/Iron_man.jpg \
# --video_path ./test_sample/3dbe8314b86284eb.mp4 \
# --output_path ./output/3dbe8314b86284eb.mp4 \
# --temp_path ./temp_results \
# --start 0.2 --end 0.7 \
# --start_1 1.0 --end_1 1.0 '
# os.system(cmd)

# only swap one person
# cmd = 'python test_video_swapspecific.py \
# --crop_size 224 --name people \
# --pic_specific_path ./demo_file/multispecific_our/SRC_01.jpg \
# --Arc_path arcface_model/arcface_checkpoint.tar \
# --pic_a_path ./demo_file/multispecific_our/DST_01.jpg \
# --video_path ./test_sample/3dbe8314b86284eb.mp4 \
# --output_path ./output/3dbe8314b86284eb.mp4 \
# --temp_path ./temp_results \
# --start 0.2 --end 0.7 \
# --start_1 1.0 --end_1 1.0 '
# os.system(cmd)

# multi swap multi faces to multi faces
cmd = 'python test_video_swap_multispecific.py \
--crop_size 224 --name people \
--Arc_path arcface_model/arcface_checkpoint.tar \
--video_path ./test_sample/3dbe8314b86284eb.mp4 \
--output_path ./output/3dbe8314b86284eb.mp4 \
--multisepcific_dir ./demo_file/multispecific \
--temp_path ./temp_results \
--start 0.2 --end 0.7 \
--start_1 1.0 --end_1 1.0 '
os.system(cmd)
