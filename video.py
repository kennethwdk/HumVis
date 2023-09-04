import cv2
import os

def generate_video_results(input_video, model, output_path):
    video_name = input_video
    video_basename = os.path.basename(video_name).split('.')[0]
    
    output_type = ['pose', 'mask', 'part', 'mesh']
    final_output_video_list = []
    ret_filename = []
    for out in output_type:
        _, info, _ = get_video_info(video_name)
        ret_filename.append('{}_{}.mp4'.format(video_basename, out))
        savepath = os.path.join(output_path, '{}_{}.mp4'.format(video_basename, out))
        info['savepath'] = savepath

        info['fourcc'] = cv2.VideoWriter_fourcc(*'mp4v')

        write_stream = cv2.VideoWriter(
            *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
        # if not write_stream.isOpened():
        #     print("Try to use other video encoders...")
        #     ext = info['savepath'].split('.')[-1]
        #     fourcc, _ext = recognize_video_ext(ext)
        #     info['fourcc'] = fourcc
        #     info['savepath'] = info['savepath'][:-4] + _ext
        #     write_stream = cv2.VideoWriter(
        #         *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
        
        assert write_stream.isOpened(), 'Cannot open video for writing'

        final_output_video_list.append(write_stream)

    cap = cv2.VideoCapture(video_name)
    assert cap.isOpened()

    cnt = 0
    while True:
        frameState, frame = cap.read()
        cnt += 1
        if frameState == False: break
        
        frame_pose, frame_mask, frame_part, frame_mesh = model.get_analysis_results(frame, tracking=True)
        

        final_output_video_list[0].write(frame_pose)
        final_output_video_list[1].write(frame_mask)
        final_output_video_list[2].write(frame_part)
        final_output_video_list[3].write(frame_mesh)

    print(cnt)
    print('fps: ', info['fps'])
    cap.release()
    final_output_video_list[0].release()
    final_output_video_list[1].release()
    final_output_video_list[2].release()
    final_output_video_list[3].release()
    
    convert_filenames = []
    for gen_file in ret_filename:
        savepath = os.path.join(output_path, gen_file)
        basename = os.path.basename(savepath).split('.')[0]
        newpath = os.path.join(output_path, '{}_final.mp4'.format(basename))
        os.system('ffmpeg -i {} -vcodec libx264 {}'.format(savepath, newpath))
        convert_filenames.append('{}_final.mp4'.format(basename))

    return convert_filenames


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), 'Cannot capture source'
    # self.path = input_source
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # bitrate = int(stream.get(cv2.CAP_PROP_BITRATE))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
    stream.release()

    return stream, videoinfo, datalen

def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'MP4V'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print("Unknow video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'

# from models import Model
# model = Model(backbone_name='hrnet32')
# video_name = 'examples/taiji.mp4'

# generate_video_results(video_name, model, 'examples')