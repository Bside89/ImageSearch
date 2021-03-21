import os
from os import path
import cv2
import core.utils as utils
from absl import flags, app
from absl.flags import FLAGS

flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('input', 'samples/vehicle-speed/Set01_video01.h264', 'Input video file')
flags.DEFINE_string('output', 'output/video-frames', 'Path to output extracted video frames')


def main(_argv):
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    input_file = FLAGS.input
    output_folder = FLAGS.output

    print('Input video: ' + input_file)
    print('Output frames directory: ' + output_folder)

    if not (path.exists(output_folder)):
        os.mkdir(output_folder)

    vid_cap = cv2.VideoCapture(input_file)
    success, image = vid_cap.read()
    count = 0
    if success:
        print('Video successful loaded. Extracting frames...')
    while success:
        cv2.imwrite('{}/frame{}.jpg'.format(output_folder, count), image)  # Save frame as JPEG file
        success, image = vid_cap.read()
        print('Read a new frame: {}'.format(count))
        count += 1


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
