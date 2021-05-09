import SessionState
from tkinter import filedialog
import tkinter as tk
import time
import colorsys
import random
import shutil
import glob
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist
import math
import os
import urllib
import cv2 as cv
import tempfile
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
plt.rcdefaults()


# st.title("Image or Video Upload for detection")
# st.sidebar.markdown("YOLOv4 model Object Detection")

st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
    st.title("Security in public places through image and video analysis")
    # Set up tkinter
    root = tk.Tk()
    root.withdraw()
    # # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)

    file_type1 = ["Not selected", "SOCIAL DISTANCING / MASK"]
    file_type = ["Image", "Video"]

    choice = st.sidebar.selectbox("COVID-19", file_type1)
    if choice == "SOCIAL DISTANCING / MASK":
        choice1 = "Video"
        choice1 = st.sidebar.text("VIDEO FILE REQUIRED")
    else:
        choice1 = st.sidebar.selectbox("File type", file_type)
    dirname = ""
    mkdir = ""
    count = 0
    t = 0
    MIN_DISTANCE = 50

    session_state = SessionState.get(name='')

    if st.sidebar.button("Select folder"):
        dirname = st.text_input(
            'Selected folder:', filedialog.askdirectory(master=root))
        session_state.name = dirname

    ext = ['png', 'jpg', 'jfif', 'avi', 'mp4',
           ]    # Add image formats here
    mkdir = session_state.name
    # files = []
    videofiles = []

    # Map from YOLO labels to Udacity labels.
    UDACITY_LABELS = {
        0: 'pistol',
        1: 'fire',
        2: 'smoke',
        3: 'mask',
        4: 'person',
        5: 'no-mask'
    }

    # Create a list of colors for the labels
    colors = np.random.randint(0, 255, size=(
        len(UDACITY_LABELS), 3), dtype='uint8')
    def calculatemAP(mAPpistol, mAPfire , mAPsmoke, mAPnomask, mAPmask, mAPperson):
        a1 = len(mAPpistol)
        b1 = sum(mAPpistol)
        if a1 == 0:
            a1 = 1
        a = b1/a1

        a2 = len(mAPfire)
        b2 = sum(mAPfire)
        if a2 ==0:
            a2 = 1
        b = b2/a2

        a3 = len(mAPsmoke)
        b3 = sum(mAPsmoke)
        if a3 == 0:
            a3 = 1
        c = b3/a3

        a4 = len(mAPnomask)
        b4 = sum(mAPnomask)
        if a4 == 0:
            a4 = 1
        d = b4/a4

        a5 = len(mAPmask)
        b5 = sum(mAPmask)
        if a5 == 0:
            a5 = 1
        e = b5/a5

        a6 = len(mAPperson)
        b6 = sum(mAPperson)
        if a6 ==0:
            a6 = 1
        f = b6/a6

        return a, b , c , d , e, f


    def cvDrawBoxes(results, violate, idxs, boxes, classIDs, img, violationsnumbers, mAPmask, mAPperson,v):   
        
        if len(idxs) > 0:
            for i in idxs.flatten():
                if UDACITY_LABELS[classIDs[i]] =='mask':
                    violationsnumbers[0] += 1
                    mAPmask.append(confidences[i])

                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]

                    # draw the bounding box and label on the image
                    c = (255, 0, 0)
                    cv.rectangle(img, (x, y), (x + w, y + h), c, 1)
                    text = "{}: {:.4f}".format(UDACITY_LABELS[classIDs[i]], confidences[i])
                    cv.putText(img, text, (x, y - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, c, 1)
                if UDACITY_LABELS[classIDs[i]] =='person':
                    violationsnumbers[1] += 1
                    mAPperson.append(confidences[i])
        
        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = cdist(centroids, centroids, metric="euclidean")
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)
            
            for (i, (prob, bbox, centroid)) in enumerate(results):
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)               
            
                if i in violate:
                    color = (0, 0, 255)
                    
                cv.rectangle(img, (startX, startY), (endX, endY), color, 2)
                cv.circle(img, (cX, cY), 5, color, 1)
            text = "Social Distancing Violations: {}".format(len(violate))
            cv.putText(img, text, (5, img.shape[0] - 10),cv.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)
            v.append(len(violate))    
        im = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return im       

    def graph(violationsnumbers):
        objects = ('No-Detection', 'Pistol', 'Fire', 'Smoke', 'No-Mask', 'Mask' )
        y_pos = np.arange(len(objects))
        performance = [violationsnumbers[6], violationsnumbers[0], violationsnumbers[1],
                            violationsnumbers[2],  violationsnumbers[3], violationsnumbers[4]]

        fig, ax = plt.subplots(figsize=(7, 3))
        plt.barh(y_pos, performance, align='center', alpha=1,
                    color=['yellow','green', 'red', 'gray', 'blue','magenta'])
        plt.yticks(y_pos, objects)
        # ax.plot(y_pos, performance)
        ax.set_ylabel('Violations Rules')
        ax.set_xlabel('Violations in numbers')
        #plt.xlabel('Violations')
        plt.title('Violations bar chart')
        st.pyplot(fig)

    def draw_image_with_boxes(image, boxes, confidences, classIDs, idxs, colors, t, violationsnumbers, mAPpistol, mAPfire , mAPsmoke, mAPnomask, mAPmask, mAPperson):
        text =""
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # draw the bounding box and label on the image
                color = [int(c) for c in colors[classIDs[i]]]
                cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(UDACITY_LABELS[classIDs[i]], confidences[i])
                k=i
                cv.putText(image, text, (x, y - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        
        if choice1 == 'Image':
            if not boxes:
                violationsnumbers[6] += 1
                img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                im = Image.fromarray(img.astype(np.uint8))       
                im.save('./nodetections/nodetection(%d).png' % t)
            else:
                if 'pistol' in text:
                    violationsnumbers[0] += 1
                    mAPpistol.append(confidences[k])
                elif 'fire' in text:
                    violationsnumbers[1] += 1
                    mAPfire.append(confidences[k])
                elif 'smoke' in text:
                    violationsnumbers[2] += 1
                    mAPsmoke.append(confidences[k])
                elif 'no-mask' in text:
                    violationsnumbers[3] += 1
                    mAPnomask.append(confidences[k])
                elif 'mask' in text:
                    violationsnumbers[4] += 1
                    mAPmask.append(confidences[k])
                elif 'person' in text:
                    violationsnumbers[5] += 1   
                    mAPperson.append(confidences[k])    
                img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                im = Image.fromarray(img.astype(np.uint8))       
                im.save('./detections/detection(%d).png' % t)
        elif choice1 == 'Video':
                if 'pistol' in text:
                    violationsnumbers[0] += 1
                    mAPpistol.append(confidences[k])
                elif 'fire' in text:
                    violationsnumbers[1] += 1
                    mAPfire.append(confidences[k])
                elif 'smoke' in text:
                    violationsnumbers[2] += 1
                    mAPsmoke.append(confidences[k])
                elif 'no-mask' in text:
                    violationsnumbers[3] += 1
                    mAPnomask.append(confidences[k])
                elif 'mask' in text:
                    violationsnumbers[4] += 1
                    mAPmask.append(confidences[k])
                elif 'person' in text:
                    violationsnumbers[5] += 1   
                    mAPperson.append(confidences[k])    
                im = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                return im   
        

    def yolo_v4(image, confidence_threshold=0.45, overlap_threshold=0.25):
        # Load the network. Because this is cached it will only happen once.
        @st.cache(allow_output_mutation=True)
        def load_network(config_path, weights_path):
            net = cv.dnn.readNetFromDarknet(config_path, weights_path)
            output_layer_names = net.getLayerNames()
            output_layer_names = [output_layer_names[i[0] - 1]
                                  for i in net.getUnconnectedOutLayers()]
            return net, output_layer_names
        net, output_layer_names = load_network(
            "YOLOv4/yolov4-obj.cfg", "YOLOv4/yolov4-obj_final.weights")

        # Run the YOLO neural net.
        blob = cv.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layer_names)

        # Supress detections in case of too low confidence or too much overlap.
        boxes, confidences, class_IDs, centroids, results= [], [], [], [], []
        H, W = image.shape[:2]
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x, y = int(centerX - (width / 2)
                               ), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    centroids.append((centerX, centerY))
                    class_IDs.append(classID)
            indices = cv.dnn.NMSBoxes(
                boxes, confidences, confidence_threshold, overlap_threshold)

        if len(indices) > 0:
            # loop over the indexes we are keeping
            for i in indices.flatten():
                label = UDACITY_LABELS.get(class_IDs[i], None)
                if label =='person':
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]

                    r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                    results.append(r)
                if label is None:
                    continue

        return boxes, confidences, class_IDs, indices, results
    if session_state.name == "":
        st.subheader("Please select a folder")
    else:
        st.subheader("Please be patient the detection is running..") 
        violationsnumbers = [0, 0, 0, 0, 0, 0, 0]
        mAPpistol, mAPfire , mAPsmoke, mAPnomask, mAPmask, mAPperson = [], [], [], [], [], []
        while len(os.listdir(dirname)) != 0:
            count += 1
            # if st.sidebar.button('Terminate the app',count):
           #     app = False
            #     break
           # count += 1
            files = []
            images = []
            t = 0
            if choice1 == 'Image':
                for e in ext:
                    files.extend(glob.glob(mkdir + '/*.' + e))

                images = [cv.imread(file) for file in files]

                # while len(os.listdir(dirname)) != 0:

                # x = 1
                for x in images:                    
                    boxes, confidences, classIDs, idxs , results= yolo_v4(x, confidence_threshold=0.5,
                                                                 overlap_threshold=0.2)

                    draw_image_with_boxes(
                        x, boxes, confidences, classIDs, idxs, colors, t, violationsnumbers, mAPpistol, mAPfire , mAPsmoke, mAPnomask, mAPmask, mAPperson)
                    t += 1

                for f in os.listdir(mkdir):
                    os.remove(os.path.join(mkdir, f))

                a, b , c , d , e, f = calculatemAP(mAPpistol, mAPfire , mAPsmoke, mAPnomask, mAPmask, mAPperson)
                table_data = {'Detections': ['Person','Pistol', 'Fire', 'Smoke', 'Mask','No-Mask'], 'Number of detection': [violationsnumbers[5],
                     violationsnumbers[0], violationsnumbers[1],  violationsnumbers[2],  violationsnumbers[4],violationsnumbers[3]], 'Mean Average Precision(mAP)': [f , a, b , c , e , d]}
                data = pd.DataFrame(data=table_data)
                st.table(data.head(6))

               
                time.sleep(30)
            elif choice1 == 'Video':

                for e in ext:
                    videofiles.extend(glob.glob(mkdir + '/*.' + e))

                # videos = [cv.imread(video) for video in videofiles]
                # begin video capture
                
                for x in videofiles:
                    try:
                        vid = cv.VideoCapture(int(x))
                    except:
                        vid = cv.VideoCapture(x)

                    out = None

                    # by default VideoCapture returns float instead of int
                    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
                    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
                    fps = int(vid.get(cv.CAP_PROP_FPS))
                    codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    out = cv.VideoWriter("./detections/result(%d).avi" % t,
                                         codec, fps, (width, height))
                    t += 1

                    while vid.isOpened():
                        ret, image = vid.read()
                        start_time = time.time()
                        if not ret:
                            print('Video file finished.')
                            break

                        # Get the boxes for the objects detected by YOLO by running the YOLO model.
                        boxes, confidences, classIDs, idxs, results = yolo_v4(image, confidence_threshold=0.5,
                                                                     overlap_threshold=0.2)

                        img = draw_image_with_boxes(
                            image, boxes, confidences, classIDs, idxs, colors, t, violationsnumbers, mAPpistol, mAPfire , mAPsmoke, mAPnomask, mAPmask, mAPperson)
                         
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break
                        fps = 1.0 / (time.time() - start_time)
                        print("FPS: %.2f" % fps)
                        
                        out.write(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    vid.release()
                    out.release()

                cv.destroyAllWindows()
                time.sleep(10)
                for f in os.listdir(mkdir):
                    os.remove(os.path.join(mkdir, f))

                a, b , c , d , e, f = calculatemAP(mAPpistol, mAPfire , mAPsmoke, mAPnomask, mAPmask, mAPperson)
                table_data = {'Detections': ['Person','Pistol', 'Fire', 'Smoke', 'Mask','No-Mask'], 'Number of detection': [violationsnumbers[5],
                     violationsnumbers[0], violationsnumbers[1],  violationsnumbers[2],  violationsnumbers[4],violationsnumbers[3]], 'Mean Average Precision(mAP)': [f , a, b , c , e , d]}
                data = pd.DataFrame(data=table_data)
                st.table(data.head(6))
            
            elif choice == "SOCIAL DISTANCING / MASK":
                for e in ext:
                    videofiles.extend(glob.glob(mkdir + '/*.' + e))

                v=[0]
                # begin video capture
                for x in videofiles:
                    try:
                        vid = cv.VideoCapture(int(x))
                    except:
                        vid = cv.VideoCapture(x)

                    out = None
                    t += 1    
                    # by default VideoCapture returns float instead of int
                    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
                    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
                    fps = int(vid.get(cv.CAP_PROP_FPS))
                    codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    

                    new_height, new_width = height // 2, width // 2
                    
                    out = cv.VideoWriter("./detections/result(%d).avi" % t,
                                         codec, fps, (new_height, new_width), True)    

                    while vid.isOpened():
                        ret, image = vid.read()
                        start_time = time.time()
                        if not ret:
                            print('Video file finished.')
                            break
                        
                        frame_resized = cv.resize(image,  (new_width, new_height),
                                     interpolation=cv.INTER_LINEAR)                     
        
                        violate = set() 
                        
                        # Get the boxes for the objects detected by YOLO by running the YOLO model.
                        boxes, confidences, classIDs, idxs, results = yolo_v4(frame_resized, confidence_threshold=0.5,
                                                                     overlap_threshold=0.2)
                        
                        img = cvDrawBoxes(results, violate, idxs, boxes, classIDs, frame_resized, violationsnumbers, mAPmask, mAPperson,v)
                         
                        
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break
                        fps = 1.0 / (time.time() - start_time)
                        print("FPS: %.2f" % fps)
                        out.write(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    vid.release()
                    out.release()
                cv.destroyAllWindows()
                time.sleep(10)
                for f in os.listdir(mkdir):
                    os.remove(os.path.join(mkdir, f))
                a, b , c , d , e, f = calculatemAP(mAPpistol, mAPfire , mAPsmoke, mAPnomask, mAPmask, mAPperson)
                table_data = {'Detections': ['Person','Mask',], 'Number of detection': [violationsnumbers[0], violationsnumbers[1]],
                     'Mean Average Precision(mAP)': [f, e]}
                data = pd.DataFrame(data=table_data)
                st.text("Total videos: %d" % t)
                st.table(data.head(2))
        if choice1 == 'Image':               
            st.text("Total images: %d" % t)            
            graph(violationsnumbers) 
        elif choice1 == 'Video':
            st.text("Total videos: %d" % t)            
            graph(violationsnumbers)
        elif choice == 'SOCIAL DISTANCING / MASK': 
            st.subheader("Social Distancing Violations: %d" %sum(v))
              

main()
