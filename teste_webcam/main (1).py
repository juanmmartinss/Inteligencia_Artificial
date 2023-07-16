from ultralytics import YOLO
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import math


model = YOLO('best.pt')
image_processor = AutoImageProcessor.from_pretrained('vinvino02/glpn-nyu')
model_depth = AutoModelForDepthEstimation.from_pretrained('vinvino02/glpn-nyu')


cap = cv2.VideoCapture(0)
i = 0
depth = 0
altura = 0

while cap.isOpened():
    i += 1
    success, frame = cap.read()
    h, w, _ = frame.shape
    if success:
        #------------------------------------------------------------------------------
        results = model.predict(source=frame, classes=0, conf=0.25, max_det=1, verbose=False, show_labels=True)
        if results[0].masks == None:
            annotated_frame = results[0].plot()
            annotated_frame = cv2.resize(annotated_frame, (int(w * 1.5), int(h * 1.5)))
            cv2.imshow('', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        coords = results[0].masks[0].xy

        xy_head = coords[0][0]
        for coord in coords[0]:
            if coord[1] < xy_head[1]:
                xy_head = coord

        xy_feet = coords[0][0]
        for coord in coords[0]:
            if coord[1] > xy_feet[1]:
                xy_feet = coord

        xy_head[1] += 5
        xy_feet[1] -= 5
        #------------------------------------------------------------------------------
        if i % 30 == 0:
            image = Image.fromarray(np.uint8(frame))
            pixel_values = image_processor(image, return_tensors='pt').pixel_values

            with torch.no_grad():
                outputs = model_depth(pixel_values)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode='bicubic',
                align_corners=False,
            ).squeeze()
            output = prediction.numpy()

            depth_head = round(output[int(xy_head[1])][int(xy_head[0])] * 100, 2)
            depth_feet = round(output[int(xy_feet[1])][int(xy_feet[0])] * 100, 2)
            depth = round((depth_head + depth_feet) / 2, 2)
        #------------------------------------------------------------------------------
            f = 55
            altura = math.sqrt(math.pow((xy_head[0] * depth_head - xy_feet[0] * depth_feet) / f, 2) +
                               math.pow((xy_head[1] * depth_head - xy_feet[1] * depth_feet) / f, 2) +
                               math.pow(depth_head - depth_feet, 2))
            altura = round(altura / 10, 2)
            print(f'Distância estimada: {depth}cm, Altura estimada: {altura}cm')
        #------------------------------------------------------------------------------
        annotated_frame = results[0].plot()

        frame_pil = Image.fromarray(annotated_frame)
        draw = ImageDraw.Draw(frame_pil)
        fonte = ImageFont.truetype('Lexend-Regular.ttf', 32)
        posicao = (10, 10)
        cor = (255, 0, 0)
        draw.text(posicao, 'Distância: ' + str(depth) + 'cm', font=fonte, fill=cor)

        # if depth >= 190 and depth <= 225:
        posicao = (10, 40)
        cor = (255, 0, 0)
        draw.text(posicao, 'Altura: ' + str(altura) + 'cm', font=fonte, fill=cor)
        
        annotated_frame = np.array(frame_pil)
        annotated_frame = cv2.resize(annotated_frame, (int(w * 1.5), int(h * 1.5)))
        cv2.imshow('', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()