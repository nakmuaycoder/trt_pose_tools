import cv2
import numpy as np


class HSV_Range(object):
    """This object enable the user to get a hsv range for apply color tracking
    1) instantiate
    2) use call method ( argument cv2.VideoCapture)
    - press d for change between automatic go to the next frame; or press a key for change
    - press s for save the configuration
    - press q for quit
    3) use getBounds method to return the saved value in a numpy array of shape (N_save_object, 1, 3, 2)
    fist chanel : upper bound, second canal lower bound
    """
    def __init__(self):
        self.mode = 1  # run the video or press "n" for change frame
        self.bounds = np.zeros((0, 1, 3, 2))

    def _show_trackbar(self):
        cv2.namedWindow('Trackbars')
        cv2.createTrackbar('hueLower', 'Trackbars', 0, 179, lambda s: None)
        cv2.createTrackbar('hueUpper', 'Trackbars', 179, 179, lambda s: None)
        cv2.createTrackbar('satLow', 'Trackbars', 0, 255, lambda s: None)
        cv2.createTrackbar('satHigh', 'Trackbars', 255, 255, lambda s: None)
        cv2.createTrackbar('valLow', 'Trackbars', 0, 255, lambda s: None)
        cv2.createTrackbar('valHigh', 'Trackbars', 255, 255, lambda s: None)

    def _updateParam(self):
        """update cursors value"""
        self.hueLow = cv2.getTrackbarPos('hueLower', 'Trackbars')
        self.hueUp = cv2.getTrackbarPos('hueUpper', 'Trackbars')
        self.Ls = cv2.getTrackbarPos('satLow', 'Trackbars')
        self.Us = cv2.getTrackbarPos('satHigh', 'Trackbars')
        self.Lv = cv2.getTrackbarPos('valLow', 'Trackbars')
        self.Uv = cv2.getTrackbarPos('valHigh', 'Trackbars')

    def _view_frames(self, frame, hsv):
        l_b = np.array([self.hueLow, self.Ls, self.Lv])
        u_b = np.array([self.hueUp, self.Us, self.Uv])
        FGmask = cv2.inRange(hsv, l_b, u_b)
        FG = np.concatenate([cv2.bitwise_and(frame, frame, mask=FGmask), frame], axis=1)
        cv2.imshow('FG', FG)
        c = cv2.waitKey(self.mode)
        return c

    def getBounds(self):
        return self.bounds

    def __call__(self, videocap):
        """Loop through the frames"""
        self._show_trackbar()
        ret = True
        c = ""
        while ret and c not in [ord('q'), ord('Q')]:
            if c not in [ord("u"), ord("U"), ord("s"), ord("S")]:
                ret, frame = videocap.read()
                frame = cv2.resize(frame, (300, 300))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            self._updateParam()  # get bounds
            c = self._view_frames(frame=frame, hsv=hsv)

            if c in [ord('d'), ord('D')]:
                # change display mode
                self.mode += 1
                self.mode %= 2
            elif c in [ord("s"), ord("S")]:
                s = np.asarray([[[[self.hueUp, self.hueLow], [self.Us, self.Ls], [self.Uv, self.Lv]]]])
                self.bounds = np.concatenate([self.bounds, s], axis=0)

        cv2.destroyAllWindows()

