"""
The file is named with "offline", since it works offline to get the tracking, it is not about whether "the method is
offline" or not. It is still classified as "online tracking", since only previous frames are used to generate the
tracking.

It is called "offline" since it needs the result from multiple trackers in advance. It is okay to use such a method on
MOT17 challenge.
"""

import argparse
import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

sys.path.append(os.path.join(os.getcwd(), "TrackEval-master"))
import time

import cv2
import matplotlib.pyplot as plt

plt.figure()
import numpy as np

np.set_printoptions(2, suppress=True)
import trackeval
from loguru import logger

import Matching as matching
from ExternalTrackerManager import external_tracker_manager
from OutsideTrack import outside_track
from Utils import generate_color_hist, img_patch, cleanness, compare_color_hist
from VideoManager import video_manager
from Visualize import plot_tracking


def make_parser():
    parser = argparse.ArgumentParser("RMOT-demo hard-coded sameDetA offline")
    parser.add_argument("-v", "--video-name", type=str, default="MOT17-04-DPM", help="the name of the MOT17 video")
    parser.add_argument("-vl", "--video-length", type=int, default=1050, help="how many frames are in the video")
    parser.add_argument("-f", "--framerate", type=int, default=30, help="the fps of the video")
    parser.add_argument("-e", "--external-tracker-name", type=str, default="MOT17-04-DPM-4",
                        help="the results (txt) of the tracker that you want to improve")
    parser.add_argument("-s", "--save-file-name", type=str, default="res",
                        help="the name of the txt and video you want to save")
    parser.add_argument("-it", "--iou-similarity-thresh", type=float, default=0.7,
                        help="the iou similarity threshold, from 0 to 1")
    parser.add_argument("-clt", "--cleanness-lower-thresh", type=float, default=0.3,
                        help="the lower bound to say whether a box is occluded, from 0 to 1")
    parser.add_argument("-its", "--iou-similarity-thresh-stricter", type=float, default=0.8,
                        help="the stricter iou similarity threshold, from 0 to 1")
    parser.add_argument("-at", "--appearance-similarity-thresh", type=float, default=0.6,
                        help="the appearance similarity threshold, from 0 to 1")
    parser.add_argument("-bt", "--box-score-thresh", type=float, default=0,
                        help="the box score threshold, from 0 to 1")

    return parser


if __name__ == "__main__":

    args = make_parser().parse_args()

    # max_lost_track_tolerance = 15  # > 0, integer
    iou_similarity_thresh = args.iou_similarity_thresh  # [0, 1]
    cleanness_lower_thresh = args.cleanness_lower_thresh  # [0, 1]
    iou_similarity_thresh_stricter = args.iou_similarity_thresh_stricter  # [0, 1]
    appearance_similarity_thresh = args.appearance_similarity_thresh  # [0, 1]
    box_score_thresh = args.box_score_thresh  # [0, 1]

    # read external tracker results and images

    video_length = args.video_length
    imgs_path = "./data/MOT17/MOT17/train/" + args.video_name + "/img1/"

    external_trackers = {
        args.external_tracker_name: "./data/trackers/" + args.external_tracker_name + ".txt"
    }

    video_manager = video_manager(imgs_path, video_length)
    for each in external_trackers:
        external_trackers[each] = external_tracker_manager(external_trackers[each], video_length)

    frame_id = 1
    outside_track_ID = 1
    outside_tracks = {}
    results = []
    vid_writer = \
        cv2.VideoWriter("./outputs/" + args.save_file_name + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), args.framerate,
                        video_manager.img_shape)

    for _ in range(video_length):

        # for each frame

        time_s = time.time()

        # get the image
        img = video_manager.next_frame()

        # pop outdated outside tracks
        tmp = []
        for each in outside_tracks:
            outside_tracks[each].retire()
            # if outside_tracks[each].life < -max_lost_track_tolerance:
            if outside_tracks[each].life < 0:
                tmp.append(each)
        for each in tmp:
            outside_tracks.pop(each)

        for each_external_tracker in external_trackers:
            # for each external tracker

            # get the tracking of this frame
            tracking = external_trackers[each_external_tracker].next_frame()

            tlbr_tracking = np.array([outside_track.tlwh_to_tlbr(each[2: 6]) for each in tracking])
            # there will be a final outside track updating at the end of each frame
            # but might also be updated while processing
            tlbr_outside_tracks = np.array([outside_tracks[each].tlbr for each in outside_tracks])

            # iou similarities between external tracker results and outside tracks, the higher the similar
            iou_similarity = 1 - matching.iou_distance(tlbr_tracking,
                                                       tlbr_outside_tracks,
                                                       img.shape)

            # iou similarities among external trackers, used to find these "pure" (without overlapping) boxes
            # used for updating the appearance model
            inter_iou_similarity = 1 - matching.iou_distance(tlbr_tracking,
                                                             tlbr_tracking,
                                                             img.shape)

            cls = [None for _ in range(len(tracking))]  # cleanness of each box of an external tracker
            hists_target = [None for _ in range(len(tracking))]  # color hist of each box of an external tracker

            # outside tracks are "selected" by external tracker boxes
            # that's why this method can have the same "DetA" as the original tracker, since the boxes are 100% kept

            # first, find those certain (close in IoU distance and similar in appearance) matches
            tracking_used = []
            OT_used = []

            for i in range(len(tracking)):

                # get the cleanness
                if cls[i] is not None:
                    cl = cls[i]
                else:
                    cl = cleanness(tlbr_tracking[i], tlbr_tracking, inter_iou_similarity, i, img)
                    cls[i] = cl

                # get the appearance model
                if hists_target[i] is not None:
                    hist_target = hists_target[i]
                else:
                    hist_target = generate_color_hist(img_patch(tlbr_tracking[i], img))
                    hists_target[i] = hist_target

                for j, each_track in enumerate(outside_tracks):

                    if j in OT_used:
                        continue

                    if iou_similarity[i, j] > iou_similarity_thresh:
                        if compare_color_hist(hist_target, outside_tracks[each_track].appearances) \
                                > appearance_similarity_thresh:
                            tracking_used.append(i)
                            outside_tracks[each_track].to_update.append([tracking[i][2: 6],
                                                                         tracking[i][6],
                                                                         hist_target,
                                                                         max(0.1, cl)])
                            break

            # second, find those occluded (track pred and box have a high IoU, but the cleanness is low) matches
            for i in range(len(tracking)):

                if i in tracking_used:
                    continue

                # get the cleanness
                if cls[i] is not None:
                    cl = cls[i]
                else:
                    cl = cleanness(tlbr_tracking[i], tlbr_tracking, inter_iou_similarity, i, img)
                    cls[i] = cl

                # get the appearance model
                if hists_target[i] is not None:
                    hist_target = hists_target[i]
                else:
                    hist_target = generate_color_hist(img_patch(tlbr_tracking[i], img))
                    hists_target[i] = hist_target

                for j, each_track in enumerate(outside_tracks):

                    if j in OT_used:
                        continue

                    if iou_similarity[i, j] > iou_similarity_thresh_stricter and cl < cleanness_lower_thresh:
                        tracking_used.append(i)
                        outside_tracks[each_track].to_update.append([tracking[i][2: 6],
                                                                     tracking[i][6],
                                                                     hist_target,
                                                                     max(0.1, cl)])
                        break

            # third, for those external boxes with no matching, create them a new outside track

            for i in range(len(tracking)):

                if i in tracking_used:
                    continue

                # get the cleanness
                if cls[i] is not None:
                    cl = cls[i]
                else:
                    cl = cleanness(tlbr_tracking[i], tlbr_tracking, inter_iou_similarity, i, img)
                    cls[i] = cl

                # get the appearance model
                if hists_target[i] is not None:
                    hist_target = hists_target[i]
                else:
                    hist_target = generate_color_hist(img_patch(tlbr_tracking[i], img))
                    hists_target[i] = hist_target

                if outside_track_ID == 170:
                    print(1)

                outside_tracks.update({outside_track_ID: outside_track(ID=outside_track_ID,
                                                                       tlwh=tracking[i][2:6],
                                                                       score=tracking[i][6],
                                                                       appearances=hist_target,
                                                                       appearance_score=max(0.1, cl))})
                external_trackers[each_external_tracker].correspondence.update({tracking[i][1]: [outside_track_ID, 1]})
                outside_track_ID += 1

        # pick the best to update/initialize
        for each in outside_tracks:

            if len(outside_tracks[each].to_update) != 0:
                outside_tracks[each].to_update.sort(key=lambda x: x[1] * x[3])  # box score and appearance cleanness
                d = outside_tracks[each].to_update[-1]

                if not outside_tracks[each].initialized:
                    outside_tracks[each].initialize(*d)
                else:
                    if outside_tracks[each].life >= 0:
                        if d[-1] > outside_tracks[each].appearance_score or d[-1] > 0.7:
                            outside_tracks[each].update(*d)
                        else:
                            outside_tracks[each].update(tlwh=d[0], score=d[1])
                    else:
                        outside_tracks[each].initialize(*d)

        # fps calc.
        time_e = time.time()
        fps = 1. / max(1e-5, time_e - time_s)
        if frame_id % 20 == 0:
            logger.info("Processing frame {} ({:.2f} fps)".format(frame_id, fps))

        # txt/mp4 writer
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for each in outside_tracks:

            if outside_tracks[each].updated:

                tlwh = outside_tracks[each].tlwh
                tid = each
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > 10 and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(outside_tracks[each].score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},"
                        f"{outside_tracks[each].score:.2f},-1,-1,-1\n"
                    )
        online_im = plot_tracking(
            img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=fps
        )
        vid_writer.write(online_im)

        # outside tracks do predict
        for each in outside_tracks:
            if not outside_tracks[each].updated:
                outside_tracks[each].mean[7] = 0
            outside_tracks[each].predict()
            outside_tracks[each].updated = False
            outside_tracks[each].updated_appearance = False

        frame_id += 1

    # save video (mp4)/tracking (txt)
    res_file = "./outputs/" + args.save_file_name + ".txt"
    with open(res_file, "w") as f:
        f.writelines(results)
    logger.info(f"save results to ./outputs/" + args.save_file_name + ".txt")
    if not os.path.exists("./TrackEval-master/data/trackers/mot_challenge/MOT17-train/hAIMOT/data/"):
        os.makedirs("./TrackEval-master/data/trackers/mot_challenge/MOT17-train/hAIMOT/data/")
    shutil.copy("./outputs/" + args.save_file_name + ".txt",
                "./TrackEval-master/data/trackers/mot_challenge/MOT17-train/hAIMOT/data/" + args.video_name + ".txt")
    logger.info(f"save results to ./outputs/" + args.save_file_name + ".mp4")

    # eval
    # the following code if copied from
    #   https://github.com/JonathonLuiten/TrackEval/blob/master/scripts/run_mot_challenge.py
    # ==================================================================================================================

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    dataset_config["GT_FOLDER"] = "./TrackEval-master/data/gt/mot_challenge/"
    dataset_config["TRACKERS_FOLDER"] = "./TrackEval-master/data/trackers/mot_challenge/"
    dataset_config["SEQ_INFO"] = {args.video_name: None}

    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    trackeval.Evaluator(eval_config)
    evaluator.evaluate(dataset_list, metrics_list)
