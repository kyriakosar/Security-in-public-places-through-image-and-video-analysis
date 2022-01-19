# Security-in-public-places-through-image-and-video-analysis
Video and image analysis with YOLOv4

python version = 3.6


## STEP1 ##
# Install Anaconda #

https://www.anaconda.com/products/individual

## STEP2 ##
# Create new conda vritual environment with anaconda navigator #

conda create -n env python=3.6

## STEP3 ##
# Activate the enviroment #

activate env

## STEP3 ##
# Install the dependencies #

pip install -r requirements.txt

## STEP4 ##
# Run the app with this command #

streamlit run model.py



### NOTES AND INSTRUCTIONS ###

## This app can detect images or video with these objects: ##
fire, smoke(low accuracy), person, mask, no-mask(low accuracy), pistol 


1. Put images and videos in the correct folder and select the folders from the app
2. The detection starts automatically
3. When the detection finish the images or videos goes to the detections folder
4. The images that can not be detected goes to nodetections folder



## THANK YOU
## Copyright © Kyriakos Aristidou, 2021

#References
- github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial
- github.com/AlexeyAB/darknet
- github.com/tzutalin/labelImg
