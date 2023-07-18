"""
Read the MOT17 official video data (as images), return each image at each frame.
"""

import cv2


class video_manager:

    def __init__(self, path, video_length):
        self.frame_id = 1
        self.path = path
        self.length = video_length
        tmp = cv2.imread(self.path + str(self.frame_id).zfill(6) + ".jpg")
        self.img_shape = (tmp.shape[1], tmp.shape[0])

    def next_frame(self):
        if self.frame_id > self.length:
            return None
        path = self.path + str(self.frame_id).zfill(6) + ".jpg"
        self.frame_id += 1
        return cv2.imread(path)
