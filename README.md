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

# Main Contributors
- Guilherme Augusto Silva Surek
- Mateus Isaac Di Domenico
- Matheus Henrique Reis Marchioro
