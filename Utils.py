import copy

import cv2
import numpy as np


def generate_color_hist(img_patch):
    # BRG color hist
    if img_patch.shape[0] * img_patch.shape[1] == 0:
        return None
    img = cv2.resize(img_patch, (64, 64))
    blue_track = cv2.calcHist([img], [0], None, [25], [0, 256])
    blue_track = blue_track / blue_track.sum()
    red_track = cv2.calcHist([img], [1], None, [25], [0, 256])
    red_track = red_track / red_track.sum()
    green_track = cv2.calcHist([img], [2], None, [25], [0, 256])
    green_track = green_track / green_track.sum()

    return [blue_track, red_track, green_track]


def img_patch(tlbr, img):
    return img[
           min(img.shape[0], max(0, int(tlbr[1]))):
           min(img.shape[0], max(0, int(tlbr[3]))),
           min(img.shape[1], max(0, int(tlbr[0]))):
           min(img.shape[1], max(0, int(tlbr[2]))), :]


def cleanness(tlbr_tracking, tlbr_trackings, inter_iou_similarity, I, img):
    area = abs(tlbr_tracking[0] - tlbr_tracking[2]) * abs(tlbr_tracking[1] - tlbr_tracking[3]) + 1

    mask = np.zeros([img.shape[0], img.shape[1]])

    mask[
    min(img.shape[0], max(0, int(tlbr_tracking[1]))): min(img.shape[0], max(0, int(tlbr_tracking[3]))),
    min(img.shape[1], max(0, int(tlbr_tracking[0]))): min(img.shape[1], max(0, int(tlbr_tracking[2])))] \
        = 1

    for j in range(inter_iou_similarity.shape[1]):
        if inter_iou_similarity[I, j] > 0 and j != I:
            mask[
            min(img.shape[0], max(0, int(tlbr_trackings[j][1]))): min(img.shape[0], max(0, int(tlbr_trackings[j][3]))),
            min(img.shape[1], max(0, int(tlbr_trackings[j][0]))): min(img.shape[1], max(0, int(tlbr_trackings[j][2])))] \
                = 0

    return mask.sum() / area


def cleanness_neighbors(tlbr_tracking, tlbr_trackings, inter_iou_similarity, iou_similarity, I, img):
    area = abs(tlbr_tracking[0] - tlbr_tracking[2]) * abs(tlbr_tracking[1] - tlbr_tracking[3]) + 1

    mask = np.zeros([img.shape[0], img.shape[1]])

    mask[
    min(img.shape[0], max(0, int(tlbr_tracking[1]))): min(img.shape[0], max(0, int(tlbr_tracking[3]))),
    min(img.shape[1], max(0, int(tlbr_tracking[0]))): min(img.shape[1], max(0, int(tlbr_tracking[2])))] \
        = 1

    # ==================================================================================================================

    neighbors = []

    for j in range(inter_iou_similarity.shape[1]):
        if inter_iou_similarity[I, j] > 0 and j != I:
            neighbors.append(j)
            mask[
            min(img.shape[0], max(0, int(tlbr_trackings[j][1]))): min(img.shape[0], max(0, int(tlbr_trackings[j][3]))),
            min(img.shape[1], max(0, int(tlbr_trackings[j][0]))): min(img.shape[1], max(0, int(tlbr_trackings[j][2])))] \
                = 0

    # ==================================================================================================================

    neighbors_bk = []

    for j in range(iou_similarity.shape[1]):
        if iou_similarity[I, j] > 0:
            neighbors_bk.append(j)

    return mask.sum() / area, neighbors, neighbors_bk


def compare_color_hist(hist1, hist2):
    # multiply the color-hist similarities of three channels
    return cv2.compareHist(hist1[0], hist2[0], 0) * cv2.compareHist(hist1[1], hist2[1], 0) * cv2.compareHist(
        hist1[2], hist2[2], 0)


def aspect_ratio(tlbr):
    return abs((tlbr[0] - tlbr[2]) / (tlbr[1] - tlbr[3]))


def area(tlbr):
    return abs((tlbr[0] - tlbr[2]) * (tlbr[1] - tlbr[3]))


def sim_4d(hist_tracking, hist_ot, iou_similarity, I, J, tlbr_tracking, tlbr_ot):
    # appearance similarity, the higher the similar [0, 1]
    sim_app = compare_color_hist(hist_tracking, hist_ot)
    # iou similarity, the higher the similar [0, 1]
    sim_box = iou_similarity[I, J]
    # aspect ratio similarity, the higher the similar [0, 1]
    sim_aspr = max(0, 1 - abs(
        aspect_ratio(tlbr_tracking[I]) - aspect_ratio(tlbr_ot[J])) / aspect_ratio(tlbr_tracking[I]))
    # area similarity, the higher the similar [0, 1]
    sim_area = max(0, 1 - abs(area(tlbr_tracking[I]) - area(tlbr_ot[J])) / area(tlbr_tracking[I]))

    return sim_app, sim_box, sim_aspr, sim_area


def gt_preprocess(gt):
    gt.sort(key=lambda x: x[0])


def det_dif(tlbr_a, tlbr_b, img):
    # generate a mask with the image showing the difference of detections
    img = copy.deepcopy(img)
    mask = np.ones((img.shape[0], img.shape[1])) * -1
    for each in tlbr_a:
        mask[
        min(img.shape[0], max(0, int(each[1]))): min(img.shape[0], max(0, int(each[3]))),
        min(img.shape[1], max(0, int(each[0]))): min(img.shape[1], max(0, int(each[2])))] \
            += 1
    for each in tlbr_b:
        mask[
        min(img.shape[0], max(0, int(each[1]))): min(img.shape[0], max(0, int(each[3]))),
        min(img.shape[1], max(0, int(each[0]))): min(img.shape[1], max(0, int(each[2])))] \
            += 1
    pos = mask == 0
    img[pos] = np.array([255, 0, 0])
    return img


def show_det(tlbr, img):
    # generate a mask with the image showing the difference of detections
    img = copy.deepcopy(img)
    mask = np.ones((img.shape[0], img.shape[1])) * -1
    for each in tlbr:
        mask[
        min(img.shape[0], max(0, int(each[1]))): min(img.shape[0], max(0, int(each[3]))),
        min(img.shape[1], max(0, int(each[0]))): min(img.shape[1], max(0, int(each[2])))] \
            += 1
    pos = mask == 0
    img[pos] = img[pos] * 0.5 + np.array([255, 128, 128]) * 0.5
    return img
