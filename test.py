import cv2
import time
import argparse
from deepface import DeepFace
from retinaface import RetinaFace

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--camera", type=int, help="face recognition camera")
parser.add_argument("-i", "--image", help="face recognition image path")
parser.add_argument("-r", "--result", help="face recognition result path")
args = parser.parse_args()

img_path = args.image
camera = args.camera


cap = cv2.VideoCapture(camera)

if not cap.isOpened():
    print("Cannot open camera")
    exit()


while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(img,(640,480))

    total_start = time.time()
    # start = time.time()
    # face_objs = DeepFace.extract_faces(img_path = img, enforce_detection=False)
    # end = time.time()
    # print(f"extract_faces : {end-start}")
    # for face_obj in face_objs:
    #     cv2.rectangle(img, (face_obj["facial_area"]['x'],face_obj["facial_area"]['y']), (face_obj["facial_area"]['x']+face_obj["facial_area"]['w'], face_obj["facial_area"]['y']+face_obj["facial_area"]['h']), (0,0,255), 2)

    # df = DeepFace.find(img_path = img_path, db_path = "./datasets", model_name = "Facenet", distance_metric = "euclidean_l2", detector_backend = "retinaface")
    start = time.time()
    #  "euclidean_l2"
    df = DeepFace.find(img_path = img, db_path = "./datasets", model_name = "Facenet", 
                       distance_metric = "euclidean_l2", detector_backend = "opencv", enforce_detection=False)
    end = time.time()
    print(f"find : {end-start}")
    unknow = 0

    # for idx, data in enumerate(df):
    #     last = len(data)-1
    #     if last < 0: 
    #         unknow += 1
    #         continue

    #     ori_point = (data.loc[last, 'source_x'], data.loc[last, 'source_y'])
    #     name = str(data.loc[last, 'identity']).split('\\')[1].split('/')[0]
    #     print(name)
    #     cv2.rectangle(img, ori_point, (ori_point[0]+data.loc[last,'source_w'], ori_point[1]+data.loc[last, 'source_h']), (0,255,0), 2)
    #     cv2.putText(img, name, ori_point, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = img.shape[0]*img.shape[1]/100000, color = (255, 0, 0), thickness = 1, lineType=cv2.LINE_AA) 
    
    total_end = time.time()
    print(f"{(total_end - total_start)} ms")
    print(f"{1.0/(total_end - total_start)} fps")
    # print((end - start)/1.0)
    # cv2.imshow("result", img)
    # cv2.waitKey(1)  
    # cv2.imwrite(args.result, img)

    print(str(unknow) + "unkwon person")
cap.release()
cv2.destroyAllWindows()
