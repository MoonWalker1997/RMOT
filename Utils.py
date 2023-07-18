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


def compare_color_hist(hist1, hist2):
    # multiply the color-hist similarities of three channels
    return cv2.compareHist(hist1[0], hist2[0], 0) * cv2.compareHist(hist1[1], hist2[1], 0) * cv2.compareHist(
        hist1[2], hist2[2], 0)
