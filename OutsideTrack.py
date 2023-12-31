import numpy as np

from KalmanFilter import kalman_filter


class outside_track:
    """
    An outside (outside the original tracker) copy. It can predict automatically (with a KF).
    But the updating is controlled by hard-coded rules or reasoners outside.
    """

    def __init__(self, ID, tlwh, score, appearances = None, appearance_score = 0.0):
        # data
        self.ID = ID
        self.kalman_filter = kalman_filter()
        self.mean, self.cov = self.kalman_filter.initiate(self.tlwh_to_xyah(tlwh))
        self.score = score  # the quality of a box (generated by NN)
        self.appearances = appearances  # appearance model, color-hist is used here, but can be extended
        self.appearance_score = appearance_score
        self.max_life = 10
        self.life = self.max_life  # how many frames can a track stand with no updates

        self.initialized = False
        self.updated = True  # whether it is just updated
        self.updated_appearance = True  # whether the appearance is just updated

        self.to_update = [[tlwh, score, appearances, appearance_score]]

    def retire(self, t = 1):
        self.life -= t

    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.cov = self.kalman_filter.predict(mean_state, self.cov)

    @property
    def tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.array(tlwh.copy())
        ret[2:] += ret[:2]
        return ret

    def initialize(self, tlwh, score, appearances = None, appearance_score = 0):
        #  the KF is not updated, but re-initialized
        self.life = self.max_life
        self.initialized = True
        self.updated = True
        self.updated_appearance = True  # whether the appearance is just updated

        self.kalman_filter = kalman_filter()
        self.mean, self.cov = self.kalman_filter.initiate(self.tlwh_to_xyah(tlwh))
        self.score = score  # the quality of a box (generated by NN)
        self.appearances = appearances  # appearance model, color-hist is used here, but can be extended
        self.appearance_score = appearance_score

        self.to_update = []

    def update(self, tlwh, score, appearances = None, appearance_score = 0):
        # re-active
        self.life = self.max_life
        self.updated = True

        # update the KF
        mean, cov = self.kalman_filter.update(self.mean, self.cov, self.tlwh_to_xyah(tlwh))

        # update score
        self.score = score

        # update appearance
        if appearances is not None:
            self.updated_appearance = True
            self.appearances = appearances
            self.appearance_score = appearance_score

        self.mean = mean
        self.cov = cov

        self.to_update = []
