## 環境安裝

### install tensorflow on jeston nano
```
sudo apt-cache show nvidia-jetpack
```
check TensorFlow compatibility with NVIDIA containers and Jetpack
https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel  
TensorFlow release for Jetson Nano
https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770

### install package
```
cd deepface
pip install -e .
pip install -r requirements.txt
```

## 使用流程

### sign up user
1. 按s鍵 : 儲存照片
2. 按esc : 結束
```
python sign_up_user.py --name username 
```


### face Recognition 
(retinaface 內存無法負載)
```
python face_recognition.py -i img_path -r result_path
python face_recognition.py -i ./img/unknown/P1.jpg -r ./result/P1.jpg
sudo python3 face_recognition.py -i ./img/unknown/all.jpg -r ./result/all.jpg
sudo python3 test.py -c 0 -r ./result/test.jpg
 ```

