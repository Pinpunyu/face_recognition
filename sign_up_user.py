import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name",help="username")
args = parser.parse_args()


cap=cv2.VideoCapture(0)

while(1):
    ret ,frame = cap.read()
    k=cv2.waitKey(1)
    if k==27: break
    elif k==ord('s'):

        path = './datasets/'+args.name+'/'

        if os.path.exists(path) == False: os.mkdir(path)
         
        i = len(os.listdir(path))
        print("capture "+str(i))
        cv2.imwrite(path+str(i)+'.jpg',frame)
        

    cv2.imshow("capture",frame)

cap.release()
cv2. destroyAllwindows()