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
3. Install python required libraries
    ```
    python pip install -r requirements.txt
    ```
4. Install pytorch (pip installation is failing yet)
    ```
    python pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
5. Download models weights for demo
    - [Yolov4 - Object Detection](https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
        * Save in /models/obj_det

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
Use the code below to demo the OD algorithm with your webcam read.
```
python main.py --webcam --od
```

# Main Contributors
- Guilherme Augusto Silva Surek
- Mateus Isaac Di Domenico
- Matheus Henrique Reis Marchioro
