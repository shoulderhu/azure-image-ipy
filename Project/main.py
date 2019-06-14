import cv2 as cv
import numpy as np
import sys
from multiprocessing import Process, Manager
from numpy import uint8, int32, float32, zeros_like
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import square, remove_small_objects
from skimage.util import img_as_ubyte
from time import sleep, time


class Camera(Process):
    def __init__(self, w=640, h=480):
        super(Camera, self).__init__()

        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

        self.d = Manager().dict()
        self.d["started"] = True

    def stop(self):
        self.d["started"] = False

    def run(self):
        while self.d["started"]:
            ret, frame = self.cap.read()
            if ret:
                self.d["frame"] = frame
        self.cap.release()

    def read(self):
        return self.d["frame"]


class Sift(Process):
    def __init__(self, camera, img1, low=10, high=40, h2=480, w2=640, idx=0):
        super(Sift, self).__init__()
        self.camera = camera
        self.img1 = img1
        self.MIN_MATCH = low
        self.MAX_MATCH = high
        self.idx = idx

        self.sift = cv.xfeatures2d.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(img1, None)
        self.bf = cv.BFMatcher()

        h, w = img1.shape
        self.h2, self.w2 = h2, w2
        self.pts = float32([[0, 0],
                            [0, h - 1],
                            [w - 1, h - 1],
                            [w - 1, 0]]).reshape(-1, 1, 2)
        self.kernel = square(3)

        self.d = Manager().dict()
        self.d["started"] = True
        self.d["matched"] = False

    def stop(self):
        self.d["started"] = False

    def run(self):
        while self.d["started"]:
            frame = self.camera.read()
            img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            kp2, des2 = self.sift.detectAndCompute(img2, None)
            matches = self.bf.knnMatch(self.des1, des2, k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            src_pts = []
            dst_pts = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good.append(m)
                    src_pts.append(self.kp1[m.queryIdx].pt)
                    dst_pts.append(kp2[m.trainIdx].pt)

            if self.MIN_MATCH <= len(good) <= self.MAX_MATCH:
                src_pts = float32(src_pts).reshape(-1, 1, 2)
                dst_pts = float32(dst_pts).reshape(-1, 1, 2)

                H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                if H is not None:
                    dst = int32(cv.perspectiveTransform(self.pts, H))
                    minc = max(min(dst[0][0][0], dst[1][0][0]), 0)
                    maxc = min(max(dst[2][0][0], dst[3][0][0]), self.w2 - 1)
                    minr = max(min(dst[0][0][1], dst[3][0][1]), 0)
                    maxr = min(max(dst[1][0][1], dst[2][0][1]), self.h2 - 1)
                    roi = img2[minr:maxr, minc:maxc]

                    if roi.size != 0 and roi.min() != roi.max():
                        a = roi < threshold_otsu(roi)

                        labels = label(a)
                        regions = regionprops(labels)
                        regions.sort(key=lambda x: x.area, reverse=True)

                        if len(regions) > 0:
                            a = img_as_ubyte(remove_small_objects(a, regions[0].area))
                            b = cv.morphologyEx(a, cv.MORPH_GRADIENT, self.kernel)

                            z = zeros_like(img2, dtype=uint8)
                            z[minr:maxr, minc:maxc] = b
                            self.d["z"] = z

                            cv.rectangle(img2, (minc, minr), (maxc, maxr), 255, 2)
                            # img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

                            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                               singlePointColor=None,
                                               matchesMask=matchesMask,  # draw only inliers
                                               flags=2)

                            self.d["res"] = cv.drawMatches(self.img1, self.kp1, img2, kp2, good, None, **draw_params)
                            self.d["frame"] = np.bitwise_or(frame, self.d["z"][:, :, np.newaxis])
                            self.d["matched"] = True
                        else:
                            self.d["matched"] = False
                            print("No component exist")
                    else:
                        self.d["matched"] = False
                        #if roi.size == 0:
                        #    print("ROI cannot be empty")
                        #elif roi.min() == roi.max():
                        #    print("otsh cannot use single color")
                else:
                    self.d["matched"] = False
                    #print("H cannot be estimated")
            else:
                self.d["matched"] = False
                #print(f"Not enough matches are found - {len(good)}/{self.MIN_MATCH}")

    def matched(self):
        return self.d["matched"]

    def read(self, no):
        if no == 1:
            return self.d["res"]
        elif no == 2:
            return self.d["frame"]
        elif no == 3:
            return self.d["z"]


def main(w, h):
    camera = Camera(w=w, h=h)
    camera.start()

    sleep(1)

    img = cv.imread("img/horse.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Image Not Found")
        sys.exit(0)
    sift1 = Sift(camera, img, 20, 65, h, w, 1)
    sift1.daemon = True
    sift1.start()

    img = cv.imread("img/dolphin.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Image Not Found")
        sys.exit(0)
    sift2 = Sift(camera, img, 12, 25, h, w, 2)
    sift2.daemon = True
    sift2.start()

    img = cv.imread("img/eagle.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Image Not Found")
        sys.exit(0)
    sift3 = Sift(camera, img, 10, 80, h, w, 3)
    sift3.daemon = True
    sift3.start()

    sleep(1)

    while True:
        if sift1.matched():
            cv.imshow("Cup Detection Debug 1", sift1.read(1))
            read = sift1.read(2)
            cv.putText(read, "Detect: Horse", (350, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv.imshow("Cup Detection", read)
            cv.imshow("Cup Detection Debug 2", sift1.read(3))
        elif sift2.matched():
            cv.imshow("Cup Detection Debug 1", sift2.read(1))
            read = sift2.read(2)
            cv.putText(read, "Detect: Dolphin", (350, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv.imshow("Cup Detection", read)
            cv.imshow("Cup Detection Debug 2", sift2.read(3))
        elif sift3.matched():
            cv.imshow("Cup Detection Debug 1", sift3.read(1))
            read = sift3.read(2)
            cv.putText(read, "Detect: Eagle", (350, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv.imshow("Cup Detection", read)
            cv.imshow("Cup Detection Debug 2", sift3.read(3))
        else:
            cv.imshow("Cup Detection", camera.read())

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    sift1.terminate()
    sift2.terminate()
    sift3.terminate()
    camera.terminate()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main(w=640, h=360)  # 360p


