import cv2
import argparse
import matplotlib.pyplot as plt
from deepface import DeepFace
from retinaface import RetinaFace

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image",help="face recognition image path")
parser.add_argument("-r", "--result",help="face recognition result path")
args = parser.parse_args()


# img_path = "./img/unknown/all.jpg"
img_path = args.image
# camera = args.camera

cap=cv2.VideoCapture(0)
img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_bgr = cv2.resize(img_bgr,(640,480))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


face_objs = DeepFace.extract_faces(img_path = img_rgb)
for face_obj in face_objs:
   cv2.rectangle(img_rgb, (face_obj["facial_area"]['x'],face_obj["facial_area"]['y']), (face_obj["facial_area"]['x']+face_obj["facial_area"]['w'], face_obj["facial_area"]['y']+face_obj["facial_area"]['h']), (0,0,255), 2)
 

# df = DeepFace.find(img_path = img_path, db_path = "./datasets", model_name = "Facenet", distance_metric = "euclidean_l2", detector_backend = "retinaface")
df = DeepFace.find(img_path = img_rgb, db_path = "./datasets", model_name = "Facenet", distance_metric = "euclidean_l2", detector_backend = "opencv")

# resp = RetinaFace.detect_faces(img_path)

# img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
# img_bgr = cv2.resize(img_bgr,(300,400))
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# for people in resp:
#   cv2.rectangle(img_rgb, (resp[people]['facial_area'][0],resp[people]['facial_area'][1]), (resp[people]['facial_area'][2], resp[people]['facial_area'][3]), (0,0,255), 3)

unknow = 0
for idx, data in enumerate(df):
    last = len(data)-1
    if last < 0: 
        unknow += 1
        continue

    
    ori_point = (data.loc[last, 'source_x'], data.loc[last, 'source_y'])
    name = str(data.loc[last, 'identity']).split('\\')[1].split('/')[0]
    print(name)
    cv2.rectangle(img_rgb, ori_point, (ori_point[0]+data.loc[last,'source_w'], ori_point[1]+data.loc[last, 'source_h']), (0,255,0), 2)
    cv2.putText(img_rgb, name, ori_point, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = img_rgb.shape[0]*img_rgb.shape[1]/100000, color = (255, 0, 0), thickness = 1, lineType=cv2.LINE_AA) 

img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
print(f"{unknow} unkwon person")
cv2.imwrite(args.result, img_rgb)