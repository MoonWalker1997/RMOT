"""
Read the MOT result (as a txt file) and return the tracking of one each frame.
"""


class external_tracker_manager:

    def __init__(self, path, video_length, gt = False):
        self.frame = 1
        self.index = 0
        self.length = video_length
        with open(path) as file:
            self.data = file.readlines()
        self.correspondence = {}  # format: {tracker ID: [outside_track ID, count]}

        self.gt = gt
        if gt:
            self.data.sort(key=lambda x: int(x.split(",")[0]))

    def next_frame(self):
        if self.frame > self.length:
            return None
        ret = []
        for i in range(self.index, len(self.data)):
            if int(self.data[i].split(",")[0]) == self.frame:
                if self.gt and int(self.data[i].split(",")[-2]) != 1:
                    continue
                else:
                    ret.append([float(each) for each in self.data[i].strip("\n").split(",")])
            else:
                self.index = i
                self.frame += 1
                break
        # format: [frame, object ID, top left x, top left y, width, height, score, -, -, -]
        # tlwh in short
        return ret

    def update_correspondence(self, external_tracker_ID, outside_track_ID):
        if external_tracker_ID in self.correspondence:
            if self.correspondence[external_tracker_ID][0] == outside_track_ID:
                self.correspondence[1] += 1
        else:
            self.correspondence.update({external_tracker_ID: [outside_track_ID, 1]})