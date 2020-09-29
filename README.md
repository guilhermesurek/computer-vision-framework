# computer-vision-framework
A Computer Vision Framework to run CV models and tasks.

# Models
- Human Activity Recognition
- Object Detection
- Pose Estimation

# References and Citations
This project was mainly based on:
* For project architecture and action recognition: [Kenshohara et al.](https://github.com/kenshohara/3D-ResNets-PyTorch)
* For object detection (Yolov4 - Adapted): [Tianxiaomo](https://github.com/Tianxiaomo/pytorch-YOLOv4)
                               (Original): [AlexeyAB](https://github.com/AlexeyAB/darknet)
* For pose estimation (Open Pose - Adapted): [Hzzone](https://github.com/Hzzone/pytorch-openpose)
                                 (Original): [Hidalgo, Cao and Simon - CMU](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

# Use Mode

## Setting Environment
1. Clone repository
2. Create your own virtual environment (Optional but highly recommended)
    ```
    python -m venv myenv
    ```
3. Activate your new environment (Optional but highly recommended)
    ```
    myenv/scripts/activate
    ```
4. Install python required libraries
    ```
    python pip install -r requirements.txt
    ```
5. Install pytorch (pip installation is failing yet)
    ```
    python pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
6. Download model's weights for demo
    - [Yolov4 - Object Detection](https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
        * Save in /models/obj_det
    - [Body - Pose Estimation](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABaYNMvvNVFRWqyDXl7KQUxa/body_pose_model.pth?dl=0)
        * Save in /models/pose
    - [Hand - Pose Estimation](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AAApu9PiOpzGYEUqzIzsxqbFa/hand_pose_model.pth?dl=0)
        * Save in /models/pose

## Seting Options
There are two ways to set your options to run the code.
1. Passing it in command line.
    ```
    python main.py --opts_dst_path opts/opts_output.json
    ```
2. Setting it in the opts.json file and add the path to this file in the arguments.
    ```
    python main.py --opts_src_path opts/opts_input.json
    ```

## Testing Webcam
Use the code below to test your webcam read.
```
python main.py --webcam
```

## Demo Object Detection with Webcam
Use the code below to demo the OD algorithm with your webcam input.
```
python main.py --webcam --od
```

## Demo Pose Estimation with Webcam
Use the code below to demo the PE body algorithm with your webcam input.
```
python main.py --webcam --pe
```
Use the code below to demo the PE body and hands algorithm with your webcam input.
```
python main.py --webcam --pe --pe_hand True
```

# Main Contributors
- Guilherme Augusto Silva Surek
- Mateus Isaac Di Domenico
- Matheus Henrique Reis Marchioro
