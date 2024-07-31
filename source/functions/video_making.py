import cv2
import numpy as np

def make_video(frames, output_path='videos/output_video.mp4', fps=10):
    # Get shape of the frames
    height, width, _ = frames[0].shape
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame in frames:
        out.write(frame)
    
    # Release video
    out.release()


