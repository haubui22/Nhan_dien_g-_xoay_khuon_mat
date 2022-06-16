import numpy as np
import cv2

import headpose_video

class Color():
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


class Annotator():
    
    def __init__(self, im, angles=None, bbox=None, lm=None, rvec=None, tvec=None, cm=None, dc=None, b=10.0):
        self.im = im

        self.angles = angles
        self.bbox = bbox
        self.lm = lm
        self.rvec = rvec
        self.tvec = tvec
        self.cm = cm
        self.dc = dc
        self.nose = tuple(lm[0].astype(int))
        self.box = np.array([
            ( b,  b,  b), ( b,  b, -b), ( b, -b, -b), ( b, -b,  b),
            (-b,  b,  b), (-b,  b, -b), (-b, -b, -b), (-b, -b,  b)
        ])
        self.b = b

        h, w, c = im.shape
        self.fs = ((h + w) / 2) / 500
        self.ls = round(self.fs * 2)
        self.ps = self.ls

    def draw_all(self):
        self.draw_bbox()
        self.draw_landmarks()
        self.draw_axes()
        self.draw_direction()
        self.draw_info()
        return self.im

    def get_image(self):
        return self.im

    def draw_bbox(self):
        x1, y1, x2, y2 = np.array(self.bbox).astype(int)
        cv2.rectangle(self.im, (x1, y1), (x2, y2), Color.green, self.ls)

    def draw_landmarks(self):
        for p in self.lm:
            point = tuple(p.astype(int))
            cv2.circle(self.im, point, self.ps, Color.red, -1)

    box_lines = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ])
    def draw_axes(self):
        (projected_box, _) = cv2.projectPoints(self.box, self.rvec, self.tvec, self.cm, self.dc)
        pbox = projected_box[:, 0]
        for p in self.box_lines:
            p1 = tuple(pbox[p[0]].astype(int))
            p2 = tuple(pbox[p[1]].astype(int))
            cv2.line(self.im, p1, p2, Color.blue, self.ls)

    def draw_direction(self):
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, self.b)]), self.rvec, self.tvec, self.cm, self.dc)
        p1 = self.nose
        p2 = tuple(nose_end_point2D[0, 0].astype(int))
        cv2.line(self.im, p1, p2, Color.yellow, self.ls)

    def draw_info(self, fontColor=Color.yellow):
        y, x, z = self.angles
        px, py, dy, dx = int(5 * self.fs), int(25 * self.fs), int(30 * self.fs), int(30 * self.fs)
        font = cv2.FONT_HERSHEY_DUPLEX
        fs = self.fs
        cv2.putText(self.im, "X: %+06.2f" % x, (px, py), font, fontScale=fs, color=fontColor)
        cv2.putText(self.im, "Y: %+06.2f" % y, (px, py + dy), font, fontScale=fs, color=fontColor)
        cv2.putText(self.im, "Z: %+06.2f" % z, (px, py + 2 * dy), font, fontScale=fs, color=fontColor)

        if x >0:
            text_1 = "Right!"
        else:
            text_1 = "Left!"
        if y >0:
            text_2 = "up!"
        else:
            text_2 = "down!"

        cv2.putText(self.im, text_1,(px, py + 3 * dy), font, fontScale=fs, color=Color.red)
        cv2.putText(self.im, text_2, (px, py + 4 * dy), font, fontScale=fs, color=Color.red)

        angR, angL, landmarks_ = headpose_video.face_angle(self.im)

        text_3 = f"Angle Right: {round(angR,0)}"
        text_4 = f"Angle Left: {round(angL,0)}"

        cv2.putText(self.im, text_3,(px + 8*dx, py), font, fontScale=fs, color=Color.green)
        cv2.putText(self.im, text_4, (px + 8*dx, py + dy), font, fontScale=fs, color=Color.green)

        # print(landmarks_)
        # print(landmarks_[0])
        print(landmarks_[0][0][0])
        print(landmarks_[0][0][1])
        point_1_x = int(landmarks_[0][0][0])
        point_1_y = int(landmarks_[0][0][1])

        point_2_x = int(landmarks_[0][1][0])
        point_2_y = int(landmarks_[0][1][1])

        point_3_x = int(landmarks_[0][2][0])
        point_3_y = int(landmarks_[0][2][1])

        lineColor = (255, 255, 0)

        cv2.line(self.im, (point_1_x, point_1_y), (point_2_x, point_2_y),
                 lineColor, 3)
        cv2.line(self.im, (point_1_x, point_1_y), (point_3_x, point_3_y),
                 lineColor, 3)
        cv2.line(self.im, (point_2_x, point_2_y), (point_3_x, point_3_y),
                 lineColor, 3)

        pred = ''
        fontScale = 2
        color = (255, 0, 0)
        fontThickness = 3

        cv2.putText(self.im, pred, (point_1_x, point_1_y), cv2.FONT_HERSHEY_PLAIN, fontScale, color, fontThickness,
                    cv2.LINE_AA)





