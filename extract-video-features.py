import argparse
import cv2
import keras
import numpy as np

from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input

parser = argparse.ArgumentParser(description='Extract video features frame by frame using pre-trained models.')
parser.add_argument('-f', '--file', dest='video_file', help='path to the video file', required=True)
parser.add_argument('-o', '--output', dest='feature_file', help='path to the video file', required=True)


def extract_frames(filename):
    """
    Produce frame vector and hog vector
    """
    features = []
    video = cv2.VideoCapture(filename)

    frames = []
    diffs = []
    count = 0

    previous_frame = None

    while True:
        ret, frame = video.read()

        if not ret: break

        if previous_frame is None:
            previous_frame = frame
            continue
        
        diff = cv2.absdiff(previous_frame, frame)

        # Skip stationary video
        mean = np.mean(diff)
        if mean < 1:
            previous_frame = frame
            continue
        
        diffs.append(diff)
        frames.append(frame)

        count += 1

        previous_frame = frame
    
    video.release()

    return (frames, diffs)


def bottleneck_features(frames):
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    batch_size = 32
    n = len(frames)
    resnet_model.summary()

    print('Total frames %d, frame dimension %s' % (n, frames[0].shape))
    X = resnet_model.predict(np.asarray(frames), batch_size=batch_size, verbose=1)
    X = X.reshape((n, 2048))
    return X


if __name__ == '__main__':
    args = parser.parse_args()

    frames, diffs = extract_frames(args.video_file)
    X = bottleneck_features(list(map(preprocess_input, frames)))

    np.save(args.feature_file, X)
