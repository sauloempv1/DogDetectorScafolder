# Dog Detector — Scaffold (YOLOv8)

## Features
- A uified model for dog detection in smart cities.

## Introduction
The automatic detection of dogs in unconstrained environments represents a relevant challenge in the field of computer vision, due to the high morphological variability of animals and adverse environmental conditions. Beyond the technical aspects, this problem also holds social and legal relevance, since animal abandonment and mistreatment are classified as crimes under Brazilian legislation. In this work, the use of deep learning techniques applied to computer vision is investigated for the detection of dogs in real-world images. A pipeline based on convolutional neural networks was developed, using an authorial dataset annotated through the makesense.ai platform for training with the YOLO framework. Preprocessing and data augmentation techniques were applied in order to improve the model’s generalization capability. Performance was evaluated using well-established metrics in the literature, such as precision, recall, and mean Average Precision (mAP). The results demonstrate satisfactory performance, indicating that the proposed approach is feasible for applications in real-world scenarios, even under unconstrained conditions, highlighting the potential of deep learning as a supporting tool for the automated identification of dogs.

![DogScafold](outputs/videos/video.mp4)

## Installation
```
conda create --name DogDetectorScafolder python=3.11 -y
conda activate DogDetectorScafolder

git clone https://github.com/sauloempv1/DogDetectorScafolder.git
cd DogDetectorScafolder
pip install -r requirements.txt
```

If do you want to see the training or change parameters go to ```notebook/train_yolov8_dog.ipynb```  do this:
- Create a folder ```datasets```;
- make a download of our dataset in classroom;
- enjoy our script.

## Getting Started
```python
streamlit run app.py
```

