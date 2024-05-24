import cv2
import argparse

from fast_track import Pipeline
from fast_track.trackers import get_tracker
from fast_track.detectors.third_party.yolo_nas.yolo_nas import YOLONAS

CLASSES = ['product']

def run_video_track(input_video, tracker_name , custom_detector_type, custom_detector_weights, custom_detector_names):
    camera = cv2.VideoCapture(input_video)
    weights_path = custom_detector_weights
    names = custom_detector_names
    detector_type = custom_detector_type
    
    print(f"weigths path: {weights_path}")
    print(f"detector type: {detector_type}")
    
    # detector = get_detector(weights_path=weights_path,
    #                         detector_type=detector_type,
    #                         names=names,
    #                         image_shape=(camera.get(3), camera.get(4)))
    
    detector = YOLONAS(model_arch = detector_type, weights_path=weights_path, names=CLASSES, image_shape=(camera.get(3), camera.get(4)))
    
    tracker = get_tracker(tracker_name=tracker_name,
                          names=names,
                          visualize=True)
    with Pipeline(camera=camera, detector=detector, tracker=tracker, outfile="models/output.mp4") as p:
        outfile = p.run()
        print(f"Output: {outfile}")
    
parser = argparse.ArgumentParser()
parser.add_argument("input_video", help="Path to the input video")
parser.add_argument("--tracker_name", help="Name of the tracker", default="ByteTrack")
parser.add_argument("--custom_detector_type", help = "Type of the detector", default="YOLO-NAS S")
parser.add_argument("--custom_detector_weights", help = "Path to the weights file", default="models/yolo_nas.pth")
parser.add_argument("--custom_detector_names", help = "path to text file containin class names", default="config/names.txt")
args = parser.parse_args()

output_path = run_video_track(args.input_video, args.tracker_name, args.custom_detector_type, args.custom_detector_weights, args.custom_detector_names)
