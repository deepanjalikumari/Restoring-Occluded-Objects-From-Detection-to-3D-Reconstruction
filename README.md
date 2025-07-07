# Restoring-Occluded-Objects-From-Detection-to-3D-Reconstruction
## Overview
This project is a comprehensive end-to-end pipeline for object-centric 3D view synthesis. It leverages state-of-the-art deep learning models for object detection, segmentation, inpainting, super-resolution, and 3D view generation. The pipeline is designed to process images, identify and extract objects, enhance their quality, and generate novel 3D perspectives using advanced diffusion models.

## Pipeline Stages
  ### Datasets Procesiong
    Download the dataset using the link: gist-ailab, “GitHub - gist-ailab/uoais: Codes of paper ‘Unseen Object Amodal Instance Segmentation via Hierarchical Occlusion Modeling’, ICRA 2022,” GitHub, 2022. http://github.com/gist-ailab/uoais(accessed Apr. 19,2025).

  ### Object Detection
      Utilizes a pre-trained YOLO model to detect and localize objects within an image.
      Detected objects are saved with bounding boxes and cropped for further processing.
      
  ### Segmentation
      Employs the Segment Anything Model (SAM) to generate precise masks for each detected object.
      Produces segmented images for focused object manipulation.

  ### Inpainting
      Uses a custom-trained autoencoder to inpaint segmented object regions, filling in missing or occluded parts.
      Ensures the object is visually complete for subsequent steps.

  ### Super-Resolution
      Applies Real-ESRGAN to enhance the resolution and quality of inpainted images.
      Produces high-quality object images suitable for 3D synthesis.

  ### 3D View Synthesis
      Integrates the Zero123Plus diffusion pipeline to generate novel 3D views of the enhanced objects.
      Outputs multiple perspectives, enabling richer visualizations.


### Install Dependencies

  #### Ensure you have Python 3.10+ and the packages
### Libraries
  ultralytics (YOLO for object detection)
  segment-anything (SAM for segmentation)
  jupyter_bbox_widget
  roboflow
  dataclasses-json
  supervision
  basicsr
  facexlib
  gfpgan
  torch (PyTorch)
  torchvision
  diffusers
  huggingface_hub
  opencv-python (cv2)
  numpy
  Pillow (PIL)
  einops

## Usage
### Configure Paths
  Set the input image path and output directories as per your dataset and requirements.
  Run the Pipeline
  Execute the notebook cells in order. Each stage will process the data and save results to the corresponding output folders.
Output
  The final output includes high-quality, inpainted, and super-resolved object images, as well as synthesized 3D views.

### Example


##### Ensure GPU support for efficient processing, especially for deep learning models.


