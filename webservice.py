from flask import Flask, jsonify, render_template, send_from_directory, Response
from utils import *
from PIL import Image
import cv2
import numpy as np
import os
import time 
import imutils
import logging
import sys
import io

app = Flask(__name__, static_url_path='')

@app.route('/icon/filename.ico')
def send_favicon(filename):
    return send_from_directory('icon', filename)

@app.route('/css/filename.*')
def send_css(filename):
    return send_from_directory('css', filename)

@app.route('/js/filename.*')
def send_js(filename):
    return send_from_directory('js', filename)

class_label = ""
@app.route('/getlabel')
def getlabel():
    global class_label
    return class_label

def gen():
        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        classes = []
        with open('coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers= [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors =np.random.uniform(0,255, size = (len(classes), 3))
        
        #load image 
        cap = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id = 0
        while True:
            _, frame = cap.read()
            
            frame_id +=1
            height, width, channels = frame.shape

            # detecting objects 
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0,0,0), True, crop = False )

            net.setInput(blob)
            outs = net.forward(output_layers)

            #showing informations on the screen 
            class_ids=[]
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5: 
                        #object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        #rectangle coordinates 
                        x = int(center_x - w /2)
                        y = int(center_y - h /2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            labels = set()
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame, label +" "+ str(round(confidence, 2)), (x,y -10), font, 1, color, 1)
                    labels.add(label)
                    
            elapsed_time = time.time() - starting_time
            fps = frame_id/elapsed_time
            cv2.putText(frame, 'FPS: ' + str(round(fps, 2)), (10,50), font, 4, (0,0,0), 3)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            byte_img = buf.getvalue()
            global class_label
            class_label = ", ".join(x for x in labels)
            yield ( b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + byte_img + b'\r\n')

        cap.release()   
        cv2.destroyAllWindows()

@app.route('/videofeed')
def videofeed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ai')
def index():
    return render_template('main.html')
@app.route('/')
def home():
        return render_template('home.html')
@app.route('/attivismo')
def attivismo():
    return render_template('attivismo.html')
@app.route('/resaicol')
def resaicol():
    return render_template('resaicol.html')

# GETs / POSTs
@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'Online'})

class WebService:
    host = None
    port = None
    debug = False

    def __init__(self):
        WebService.host = 'localhost'
        WebService.port = 8080

    def start(self, debug=False):
        info_print("Starting WebService...")
        WebService.debug = debug

        if not debug:
            cli = sys.modules['flask.cli']
            cli.show_server_banner = lambda *x: None
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

        app.run(host=WebService.host, port=WebService.port, debug=debug, use_reloader=debug)
        

if __name__ == '__main__':
    ws = WebService()
    ws.start(True)
