import argparse
import sys

import cv2

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display
from utils.mtcnn import TrtMtcnn

WINDOW_NAME = 'TestWindow'
BBOX_COLOR = (0, 255, 0)  # green


def parse_args():
    desc = 'Capture and display live camera video'
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40, help='minsize (in pixels) for detection [40]')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)


def detect_faces(cam, mtcnn, minsize=40):
    full_scrn = False
    # fps = 0.0
    # tic = time.time()

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            dets, landmarks = mtcnn.detect(img, minsize=minsize)
            print('{} face(s) found'.format(len(dets)))
            show_faces(img, dets, landmarks)
            cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('F') or key == ord('f'):
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    cam = Camera(args)

    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    mtcnn = TrtMtcnn()
    cam.start()
    open_window(WINDOW_NAME, width=640, height=480, title='MTCNN Window')
    detect_faces(cam, mtcnn)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()
    del mtcnn


if __name__ == '__main__':
    main()
