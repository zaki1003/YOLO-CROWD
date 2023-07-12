# YOLO-CROWD
YOLO-CROWD is a lightweight crowd counting and face detection model that is based on Yolov5s and can run on edge devices, as well as fixing the problems of face occlusion, varying face scales, and other challenges of crowd counting


## Description
Deep learning-based algorithms for face and crowd identification have advanced significantly. These algorithms can be broadly categorized into two groups: one-stage detectors like YOLO and two-stage detectors like Faster R-CNN. One-stage detectors have been widely employed in many applications due to the better balance between accuracy and speed, but as we are all aware, YOLO algorithms are significantly impacted by occlusion in crowd scenarios. In our project, we propose a real-time crowd counter and face detector called **YOLO-CROWD**, which has an inference speed of **9 ms** and contains 461 layers and 18388982 parameters. It is based on the one-stage detector YOLOv5. In order to improve the receptive field of small faces, we use a Receptive Field Enhancement module termed RFE. We then use NWD Loss to compensate for the sensitivity of IoU to the position deviation of small objects. We also employ Repulsion Loss to address face occlusion and utilize an attention module called
SEAM.

## Demo

![test-yolo-crowd](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/6aed4956-1da5-4b98-ae8a-e7d9574b4054)

![Screenshot from 2023-04-07 15-49-11](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/e435d92b-42f2-4152-bcad-b72268db8d0e)

![Screenshot from 2023-04-07 15-48-52](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/2b5e3273-a697-472c-a201-0b23e5b2faa6)


## Comparison Between Yolov5s And YOLO-CROWD

|                |          mAp@0.5      |       mAp@0.5-095   |           Precision      |          Recall         |         Box loss        |         Object loss      |     Inference Time (ms)  |
|:-------------------|:---------------|:--------------|:-------------|:-----------|:------------------|:------------------|:-----------------------------|
|         Yolov5s      |          39.4      |     0.15       |        0.754        |        0.382           |      0.120            |    0.266                  |        **7**            |                    
|       YOLO-CROWD        |            **43.6**          |         **0.158**         |      **0.756**        |        **0.424**        |         **0.091**       |  **0.158**       |       9        | 




## Environment Requirments
Create a Python Virtual Environment.   
```shell
conda create -n {name} python=x.x
```

Enter Python Virtual Environment.   
```shell
conda activate {name}
```

Install pytorch in *[this](https://pytorch.org/get-started/previous-versions/)*.  
```shell 
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other python package.   
```shell
pip install -r requirements.txt
```

## Step-Through Example
### Installation
Get the code.    
```shell
git clone https://github.com/Krasjet-Yu/YOLO-FaceV2.git
```

### Dataset

Download our Dataset [crowd-counting-dataset-w3o7w](https://universe.roboflow.com/crowd-dataset/crowd-counting-dataset-w3o7w), While exporting the datset try to choose **YOLO v5 PyTorch** Format.

![our-dataset](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/7c574121-7eb5-450c-a61d-d259643d22fb)



## Preweight
The link is [yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt)


### Training
Train your model on **crowd-counting-dataset-w3o7w** dataset.
```shell
python train.py --weights preweight.pt    
                --data data/WIDER_FACE.yaml    
                --cfg models/yolov5s_v2_RFEM_MultiSEAM.yaml     
                --batch-size 32   
                --epochs 250
```

## Postweight
The link is [yolo-crowd.pt](https://drive.google.com/file/d/1xxXVCzseuzmHv7NoMQ03RVU_tDisWXjM/view?usp=sharing)
If you want to have more inference speed try to install TensorRt and use this vesion [yolo-crowd.engine](https://drive.google.com/file/d/1-189sscpNZBFaSHOz7dnEgAaFeUALiow/view?usp=sharing)


### Test
```shell
python detect.py --weights ./preweight/best.pt --source ./data/images/test.jpg --plot-label --view-img
```

### Evaluate    

Evaluate the trained model via next code on WIDER FACE   
        
If you don't want to train, you can also directly use our trained model to evaluate.   

The link is [yolo-facev2_last.pt](https://github.com/Krasjet-Yu/YOLO-FaceV2/releases/download/v1.0/best.pt)     


```shell
python widerface_pred.py --weights runs/train/x/weights/best.pt     
                         --save_folder ./widerface_evaluate/widerface_txt_x    
cd widerface_evaluate/    
python evaluation.py --pred ./widerface_txt_x
```
Download the *[eval_tool](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip)* to show the performance.    
    
The result is shown below:    

![](data/images/eval.png)




## Results

![results-yolo-crowd](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/9e2d18ce-aaf6-4a20-91f0-d8d1eb88728c)


## Finetune
see in *[https://github.com/ultralytics/yolov5/issues/607](https://github.com/ultralytics/yolov5/issues/607)*
```shell
# Single-GPU
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve

# Multi-GPU
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  nohup python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device $i --evolve > evolve_gpu_$i.log &
done

# Multi-GPU bash-while (not recommended)
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  "$(while true; do nohup python train.py... --device $i --evolve 1 > evolve_gpu_$i.log; done)" &
done
```

## Reference
*[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)*    
    
*[[https://github.com/deepcam-cn/yolov5-face](https://github.com/Krasjet-Yu/YOLO-FaceV2)  
    
*[https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)*   
    
*[https://github.com/dongdonghy/repulsion_loss_pytorch](https://github.com/dongdonghy/repulsion_loss_pytorch)*   



## Contact

We use code's license is MIT License. The code can be used for business inquiries or professional support requests.
