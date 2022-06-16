

import argparse
import cv2
import numpy as np
import os.path as osp
import headpose
from facenet_pytorch import MTCNN
import torch

# Landmarks: [Left Eye], [Right eye], [nose], [left mouth], [right mouth]
def npAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def face_angle(im):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(image_size=160,
                  margin=0,
                  min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
                  factor=0.709,
                  post_process=True,
                  device=device  # If you don't have GPU
                  )

    bbox_, prob_, landmarks_ = mtcnn.detect(im, landmarks=True)
    None_ = None
    if bbox_ is None_:
        print("Don't see the face!")
        angR = 0
        angL = 0
        landmarks_ = np.zeros((1, 5, 2))
    # for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
    else:
        angR = npAngle(landmarks_[0][0], landmarks_[0][1], landmarks_[0][2])  # Calculate the right eye angle
        angL = npAngle(landmarks_[0][1], landmarks_[0][0], landmarks_[0][2])
    return angR, angL, landmarks_



def main(args):
    filename = args["input_file"]

    if filename is None:
        isVideo = False
        cap = cv2.VideoCapture(0)
        cap.set(3, args['wh'][0])
        cap.set(4, args['wh'][1])
    else:
        isVideo = True
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args["output_file"], fourcc, fps, (width, height))

    # Initialize head pose detection
    hpd = headpose.HeadposeDetection(args["landmark_type"], args["landmark_predictor"])

    count = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        print('\rframe: %d' % count, end='')
        ret, frame = cap.read()
        
        if isVideo:
            frame, angles = hpd.process_image(frame)
            angR, angL, landmarks_ = face_angle(frame)
            print("Angle Right: ", angR)
            print("Angle Left: ", angL)
            if frame is None: 
                break
            else:
                out.write(frame)
        else:
            frame = cv2.flip(frame, 1)
            frame, angles = hpd.process_image(frame)
            angR, angL, landmarks_ = face_angle(frame)
            print("Angle Right: ", angR)
            print("Angle Left: ", angL)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                headpose.t.summary()
                break
        count += 1

    # When everything done, release the capture
    cap.release()
    if isVideo: out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None, help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
    parser.add_argument('-wh', metavar='N', dest='wh', default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', 
                        default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())
    main(args)
