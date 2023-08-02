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
from Utils import generate_color_hist, img_patch, cleanness_neighbors, compare_color_hist, sim_4d, det_dif
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
    parser.add_argument("-it", "--iou-similarity-thresh", type=float, default=0.9,
                        help="the iou similarity threshold, from 0 to 1")
    parser.add_argument("-its", "--iou-similarity-thresh-stricter", type=float, default=0.1,
                        help="the stricter iou similarity threshold, from 0 to 1")
    parser.add_argument("-at", "--appearance-similarity-thresh", type=float, default=0.1,
                        help="the appearance similarity threshold, from 0 to 1")
    parser.add_argument("-bt", "--box-score-thresh", type=float, default=0.8,
                        help="the box score threshold, from 0 to 1")
    # clt, the higher, the more clear-cut of clusters
    parser.add_argument("-clt", "--cleanness-thresh", type=float, default=0.8,
                        help="the cleanness threshold, from 0 to 1")
    # clts, the higher, the more compact (smaller) of clusters
    parser.add_argument("-clts", "--cleanness-thresh-smoother", type=float, default=0.6,
                        help="the smoother cleanness threshold, from 0 to 1")
    parser.add_argument("-mf", "--ot-max-life", type=int, default=10,
                        help="the outside track max life, > 0")

    return parser


if __name__ == "__main__":

    args = make_parser().parse_args()

    # max_lost_track_tolerance = 15  # > 0, integer
    iou_similarity_thresh = args.iou_similarity_thresh  # [0, 1]
    iou_similarity_thresh_stricter = args.iou_similarity_thresh_stricter  # [0, 1]
    appearance_similarity_thresh = args.appearance_similarity_thresh  # [0, 1]
    box_score_thresh = args.box_score_thresh  # [0, 1]
    cleanness_thresh = args.cleanness_thresh  # [0, 1]
    cleanness_thresh_smoother = args.cleanness_thresh_smoother  # [0, 1]
    ot_max_life = args.ot_max_life  # > 0

    # read external tracker results and images

    video_length = args.video_length
    imgs_path = "./data/MOT17/MOT17/train/" + args.video_name + "/img1/"

    external_trackers = {
        args.external_tracker_name: "./data/trackers/" + args.external_tracker_name + ".txt",
        "gt": "./data/MOT17/MOT17/train/" + args.video_name + "/gt/gt.txt"
    }

    video_manager = video_manager(imgs_path, video_length)
    for each in external_trackers:
        if each == "gt":
            external_trackers[each] = external_tracker_manager(external_trackers[each], video_length, True)
        else:
            external_trackers[each] = external_tracker_manager(external_trackers[each], video_length)

    frame_id = 1
    outside_track_ID = 1
    outside_tracks = {}
    results = []
    vid_writer = \
        cv2.VideoWriter("./outputs/" + args.save_file_name + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), args.framerate,
                        video_manager.img_shape)
    vid_writer_dif_t_ot = \
        cv2.VideoWriter("./outputs/" + args.save_file_name + "_dif_t_ot.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                        args.framerate, video_manager.img_shape)
    vid_writer_dif_gt_t = \
        cv2.VideoWriter("./outputs/" + args.save_file_name + "_dif_gt_t.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                        args.framerate, video_manager.img_shape)
    vid_writer_dif_gt_ot = \
        cv2.VideoWriter("./outputs/" + args.save_file_name + "_dif_gt_ot.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                        args.framerate, video_manager.img_shape)

    # for each frame
    for _ in range(video_length):

        if frame_id == 969 or frame_id == 976:
            print(1)

        tracking = None

        time_s = time.time()  # fps calc. util

        img = video_manager.next_frame()

        # pop outdated outside tracks
        tmp = []
        for each in outside_tracks:
            outside_tracks[each].retire()
            if outside_tracks[each].life < 0:
                tmp.append(each)
        for each in tmp:
            outside_tracks.pop(each)

        for each_external_tracker in external_trackers:
            # for each external tracker

            if each_external_tracker == "gt":
                continue

            # boxes info.
            tracking = external_trackers[each_external_tracker].next_frame()
            tlbr_tracking = np.array([outside_track.tlwh_to_tlbr(each[2: 6]) for each in tracking])
            # there will be a final outside track updating at the end of each frame
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
            nebs = [None for _ in range(len(tracking))]  # neighbors of each box in tracking
            nebs_bk = [None for _ in range(len(tracking))]  # background
            hists_target = [None for _ in range(len(tracking))]  # color hist of each box of an external tracker
            for i in range(len(tracking)):
                cls[i], nebs[i], nebs_bk[i] = \
                    cleanness_neighbors(tlbr_tracking[i], tlbr_tracking, inter_iou_similarity, iou_similarity, i, img)
                hists_target[i] = generate_color_hist(img_patch(tlbr_tracking[i], img))

            # outside tracks are "selected" by external tracker boxes
            # that's why this method can have the same "DetA" as the original tracker, since the boxes are 100% kept

            tracking_used = set()
            ot_used = set()

            # first, find those certain (close in IoU distance and similar in appearance) matches
            # except for the above requirement, we also need the box is kinda "isolated", say it is isolated from the
            # other boxes

            for i in range(len(tracking)):

                # get the cleanness and neighbors
                cl, neb = cls[i], nebs[i]

                # get the appearance model
                hist_target = hists_target[i]

                tmp = []

                for j, each_track in enumerate(outside_tracks):

                    sim = compare_color_hist(hist_target, outside_tracks[each_track].appearances)
                    # isolated, close, similar
                    if cl > cleanness_thresh \
                            and iou_similarity[i, j] > iou_similarity_thresh \
                            and sim > appearance_similarity_thresh:
                        tmp.append([each_track, tracking[i][2: 6], tracking[i][6], hist_target, max(0.1, cl), sim])

                if len(tmp) != 0:
                    tmp.sort(key=lambda x: x[2] + x[4] + x[5])

                    tracking_used.add(i)
                    ot_id = tmp[-1][0]
                    ot_used.add(list(outside_tracks.keys()).index(ot_id))

                    # update correspondence
                    if tracking[i][1] in external_trackers[each_external_tracker].correspondence \
                            and ot_id == external_trackers[each_external_tracker].correspondence[tracking[i][1]][0]:
                        external_trackers[each_external_tracker].correspondence[tracking[i][1]][1] += 1
                    else:
                        external_trackers[each_external_tracker].correspondence.update({tracking[i][1]: [ot_id, 1]})

                    outside_tracks[tmp[-1][0]].to_update.append([tmp[-1][1],
                                                                 tmp[-1][2],
                                                                 tmp[-1][3],
                                                                 tmp[-1][4],
                                                                 tmp[-1][5]])

            # ==========================================================================================================

            # second, generate boxes clusters
            # for those boxes entangled, find a cluster to describe them

            clusters = []
            isolated_boxes = []

            for i in range(len(tracking)):

                if i in tracking_used:
                    continue

                # get the cleanness and neighbors
                cl, neb = cls[i], nebs[i]

                if cl < cleanness_thresh:  # it is entangled with its neighbors
                    # local cluster generation
                    tmp = {i}
                    for each_n in neb:
                        if cls[each_n] < cleanness_thresh_smoother:
                            tmp.add(each_n)

                    mark = False
                    for j in range(len(clusters)):
                        # if this box is in one cluster
                        if i in clusters[j]:
                            clusters[j] = set.union(clusters[j], tmp)
                            mark = True
                            break
                        # if its neighbors are in one cluster
                        elif len(clusters[j] & tmp) != 0:
                            clusters[j] = set.union(clusters[j], tmp)
                            mark = True
                            break

                    # a brand-new cluster
                    if not mark:
                        clusters.append(tmp)
                else:
                    isolated_boxes.append(i)

            # ==========================================================================================================

            # third, process isolated boxes first
            # isolated boxes are not entangled with other boxes too much, so the appearance can still be used as an
            # evidence (not necessary depend on it)

            for i in isolated_boxes:

                # get the cleanness and neighbors
                cl, neb, neb_bk = cls[i], nebs[i], nebs_bk[i]

                # get the appearance model
                hist_target = hists_target[i]

                tmp = []

                for j in neb_bk:

                    if j in ot_used:  # index of outside track
                        continue

                    # outside track ID, for reference
                    ot_id = list(outside_tracks.keys())[j]

                    sim_app, sim_box, sim_aspr, sim_area = sim_4d(hist_target,
                                                                  outside_tracks[ot_id].appearances,
                                                                  iou_similarity,
                                                                  i,
                                                                  j,
                                                                  tlbr_tracking,
                                                                  tlbr_outside_tracks)

                    sim_comp = 0

                    # if they are visually similar, this score will be added, causing a natural advantage
                    # else, this score is not added
                    if sim_app > appearance_similarity_thresh:
                        sim_comp += sim_app

                    # comprehensive similarity
                    sim_comp += sim_box + sim_aspr + sim_area
                    sim_comp /= 4

                    # candidate associations
                    tmp.append([ot_id, tracking[i][2: 6], tracking[i][6], hist_target, max(0.1, cl), sim_comp])

                if len(tmp) != 0:
                    tmp.sort(key=lambda x: x[-1])

                    tracking_used.add(i)
                    ot_id = tmp[-1][0]
                    ot_used.add(list(outside_tracks.keys()).index(ot_id))

                    # update correspondence
                    if tracking[i][1] in external_trackers[each_external_tracker].correspondence \
                            and ot_id == external_trackers[each_external_tracker].correspondence[tracking[i][1]][0]:
                        external_trackers[each_external_tracker].correspondence[tracking[i][1]][1] += 1
                    else:
                        external_trackers[each_external_tracker].correspondence.update({tracking[i][1]: [ot_id, 1]})

                    outside_tracks[tmp[-1][0]].to_update.append([tmp[-1][1],
                                                                 tmp[-1][2],
                                                                 tmp[-1][3],
                                                                 tmp[-1][4],
                                                                 tmp[-1][5]])

            # ==========================================================================================================

            # fourth, process those entangled boxes
            # since these boxes are highly entangled, visual and positional evidences are not reliable
            # but, when the appearance model is very close, it is also okay for an evidence

            for each_c in clusters:

                associated_box = set()
                associated_neighbor = set()

                for i in each_c:

                    # get the cleanness and neighbors
                    cl, neb, neb_bk = cls[i], nebs[i], nebs_bk[i]

                    # get the appearance model
                    hist_target = hists_target[i]

                    tmp = []

                    if tracking[i][1] in external_trackers[each_external_tracker].correspondence:
                        # if the external track ID has a correspondence (to outside track IDs)
                        ot_id = external_trackers[each_external_tracker].correspondence[tracking[i][1]][0]

                        if ot_id in outside_tracks:

                            ot_idx = list(outside_tracks.keys()).index(ot_id)

                            if ot_idx in neb_bk:

                                sim_app, sim_box, sim_aspr, sim_area = sim_4d(hist_target,
                                                                              outside_tracks[ot_id].appearances,
                                                                              iou_similarity,
                                                                              i,
                                                                              ot_idx,
                                                                              tlbr_tracking,
                                                                              tlbr_outside_tracks)

                                # TODO, may have hyperparameters here
                                if sim_app > appearance_similarity_thresh \
                                        or (sim_box > 0.6 and sim_aspr > 0.6 and sim_area > 0.6):
                                    # if so, we think this correspondence is still valid
                                    # enhance the correspondence and update the outside tracker
                                    external_trackers[each_external_tracker].correspondence[tracking[i][1]][1] += 1

                                    outside_tracks[ot_id].to_update.append([tracking[i][2: 6],
                                                                            tracking[i][6],
                                                                            None,
                                                                            -1,
                                                                            -1])

                                    associated_box.add(i)
                                    associated_neighbor.add(ot_idx)

                                    tracking_used.add(i)
                                    ot_used.add(ot_idx)

                # start here, boxes with correspondence are associated, then bipartite matching is to be used

                tracking_tmp = list(each_c - associated_box)  # idx
                ot_tmp = list(set(neb_bk) - associated_neighbor)  # idx

                cost_matrix = np.zeros((len(tracking_tmp), len(ot_tmp)))

                for i_idx, i in enumerate(tracking_tmp):

                    if i in associated_box:
                        continue

                    # get the cleanness and neighbors
                    cl, neb, neb_bk = cls[i], nebs[i], nebs_bk[i]

                    # get the appearance model
                    hist_target = hists_target[i]

                    for j_idx, j in enumerate(ot_tmp):

                        if j in ot_used:
                            continue

                        ot_id = list(outside_tracks.keys())[j]
                        sim_app, sim_box, sim_aspr, sim_area = sim_4d(hist_target,
                                                                      outside_tracks[ot_id].appearances,
                                                                      iou_similarity,
                                                                      i,
                                                                      j,
                                                                      tlbr_tracking,
                                                                      tlbr_outside_tracks)

                        # 1 - x, since this is the COST matrix
                        cost_matrix[i_idx, j_idx] = 1 - (sim_app + sim_box + sim_aspr + sim_area) / 4

                # here we got a cost matrix for boxes without association
                # then linear assignment will be used to associate them

                matches, unmatched_a, unmatched_b = matching.linear_assignment(cost_matrix, 0.8)

                for each_m in matches:

                    i, j = each_m[0], each_m[1]

                    ot_id = list(outside_tracks.keys())[j]

                    # update correspondence
                    if tracking[i][1] in external_trackers[each_external_tracker].correspondence \
                            and ot_id == external_trackers[each_external_tracker].correspondence[tracking[i][1]][0]:
                        external_trackers[each_external_tracker].correspondence[tracking[i][1]][1] += 1
                    else:
                        external_trackers[each_external_tracker].correspondence.update({tracking[i][1]: [ot_id, 1]})

                    # update outside trackers

                    tracking_used.add(i)

                    outside_tracks[ot_id].to_update.append([tracking[i][2: 6],
                                                            tracking[i][6],
                                                            None,
                                                            -1,
                                                            -1])

            # ==========================================================================================================

            # fifth, for those external boxes with no matching, create them a new outside track

            for i in range(len(tracking)):

                if i in tracking_used or tracking[i][6] < args.box_score_thresh:
                    # associated or low quality box
                    continue

                # get the cleanness and neighbors
                cl, neb = cls[i], nebs[i]

                # get the appearance model
                hist_target = hists_target[i]

                ot = outside_track(ID=outside_track_ID,
                                   tlwh=tracking[i][2:6],
                                   score=tracking[i][6],
                                   appearances=hist_target,
                                   appearance_score=max(0.1, cl))
                ot.max_life = ot_max_life
                outside_tracks.update({outside_track_ID: ot})
                external_trackers[each_external_tracker].correspondence.update({tracking[i][1]: [outside_track_ID, 1]})
                outside_track_ID += 1

        # pick the best to update/initialize
        for each in outside_tracks:

            if len(outside_tracks[each].to_update) != 0:

                outside_tracks[each].to_update.sort(key=lambda x: x[1] * x[3])  # box score and appearance cleanness
                d = outside_tracks[each].to_update[-1][:-1]
                sim = outside_tracks[each].to_update[-1][-1]

                if not outside_tracks[each].initialized:
                    outside_tracks[each].initialize(*d)
                else:
                    if sim > appearance_similarity_thresh \
                            and (d[-1] > outside_tracks[each].appearance_score or d[-1] > 0.7):
                        outside_tracks[each].update(*d)
                    else:
                        outside_tracks[each].update(tlwh=d[0], score=d[1])

        # checking gt [debugging]

        tlbr_tracking = np.array([outside_track.tlwh_to_tlbr(each[2: 6]) for each in tracking])

        tlbr_outside_tracks = np.array([outside_tracks[each].tlbr for each in outside_tracks])

        gt = external_trackers["gt"].next_frame()
        tlbr_gt = np.array([outside_track.tlwh_to_tlbr(each[2: 6]) for each in gt])

        A = det_dif(tlbr_tracking, tlbr_outside_tracks, img)
        B = det_dif(tlbr_gt, tlbr_tracking, img)
        C = det_dif(tlbr_gt, tlbr_outside_tracks, img)
        vid_writer_dif_t_ot.write(A)
        vid_writer_dif_gt_t.write(B)
        vid_writer_dif_gt_ot.write(C)

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
    dataset_config["TRACKERS_TO_EVAL"] = ["hAIMOT"]
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
