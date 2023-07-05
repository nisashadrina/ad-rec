# Inisialisasi library yang digunakan
from flask import Flask, render_template, Response
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import random
import ultralytics
import supervision as sv
from datetime import timedelta

app = Flask(__name__)

# Inisialisasi global variable
#     Inisialisasi path
VIDEO_PATH = 'test-5.mp4'
MODEL_PATH = 'models/best-yolov8m.pt'
CLASS_PATH = 'classes.txt'

# Inisialisasi data iklan
prod_data = pd.read_csv("product_data.csv")

#     Inisialisasi model
model = YOLO(MODEL_PATH)

#     Inisialisasi area
AREA = [
    (493, 401),
    (813, 353),
    (925, 417),
    (529, 541),
    (489, 489)
]

AREA = np.array(AREA, dtype=np.int32)

#     Inisialisasi data class yang digunakan  
FILES = open(CLASS_PATH, "r")
CLASS_DATA = FILES.read()
CLASS_LIST = CLASS_DATA.split("\n")
    
#     Inisialisasi per class untuk counter
boy_detected = set()
girl_detected = set()
man_detected = set()
woman_detected = set()
    
#      Inisialisasi variabel waktu
START_TIME = time.time()
TIME_TO_RESET = 5

# Function untuk mengetahui informasi video
def getVideoInfo(video_path):
    # Mengekstraksi informasi yang ada dalam video
    video_info = sv.VideoInfo.from_video_path(video_path)
    width, height, fps, total_frames = video_info.width, video_info.height, video_info.fps, video_info.total_frames
    
    # Menghitung durasi video
    video_length = timedelta(seconds = round(total_frames / fps))
    
    return width, height, fps, video_length, total_frames

def createScore(data):
    # Mengatur bobot kriteria
    weights = [0.5, 0.5]
    
    # Menghitung matriks ternormalisasi terbobot
    X_weighted = data['category_priority'] * weights[0] + data['product_priority'] * weights[1]
    
    # Mengalikan bobot dengan nilai atribut dan menjumlahkannya untuk mendapatkan skor
    data['score'] = X_weighted

    return data

def probability(data_res):
    # Menghitung total skor
    total_score = data_res['score'].sum()

    # Menghitung peluang kemunculan untuk setiap produk
    data_res['probability'] = data_res['score'] / total_score

    # Membuat daftar produk berdasarkan peluang kemunculan
    products = data_res['product_id'].tolist()
    probabilities = data_res['probability'].tolist()

    # Menggunakan metode random.choices untuk melakukan randomisasi dengan mempertimbangkan peluang kemunculan
    recommended_products = random.choices(products, probabilities, k=1)

    product = data_res.loc[data_res['product_id'] == recommended_products[0]]
    prod_name = product['product_name'].values[0]

    return prod_name

def randomAd(max_class):
    global prod_data
    
    data = createScore(prod_data)
    
    if max_class == 5:
        random_product = data['product_id'].tolist()
        recommended_products = random.choice(random_product)
        product = data.loc[data['product_id'] == recommended_products[0]]
        product_name = product['product_name'].values[0]
    else:
        result = data.loc[data['class'] == max_class]
        product_name = probability(result)        

    return product_name

def argmax_unique(array):
    max_index = np.argmax(array)
    if np.count_nonzero(array == array[max_index]) > 1:
        return None
    return max_index

def maxCondition(array):
    boy = array[0]
    girl = array[1]

    max_class = argmax_unique(array)

    if boy != 0 and boy > girl:
        return 0
    elif girl != 0 and boy < girl:
        return 1
    elif max_class is None:
        return 5
    else:
        return max_class

def produceVideo():
    global VIDEO_PATH, model, AREA, CLASS_LIST, START_TIME, TIME_TO_RESET, data
    global boy_detected, girl_detected, man_detected, woman_detected
    fps = 0
    ad_product = "None"
    prev_time = time.time()

    # Inisialisasi video
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        # Keluar dari loop jika sudah mencapai akhir video
        if not success:
            break

        # Untuk menghitung fps
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Resize ukuran frame
        frame = cv2.resize(frame, (1280, 720))

        # Memanggil model recognition dan model tracker
        results = model.track(frame, conf = 0.65, persist=True, tracker="botsort.yaml")

        # Mengekstrak informasi frame yang telah diprediksi
        datas = results[0].boxes.data.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy().astype("float")

        # Perulagan untuk menyimpan data class yang terdeteksi
        for box, data, conf in zip(boxes, datas, confs):
            try:
                track_id = data[4]
                class_id = data[6]
                classes = CLASS_LIST[class_id]
                conf = conf * 100
                dot = (int(box[2]), int(box[3]))
            except IndexError:
                # Skip frame jika elemen ke-6 tidak ada
                continue
            
            results = cv2.pointPolygonTest(AREA, dot, False)
            if results >= 0 and classes != 'back':
                if classes == 'boy' and track_id not in boy_detected:
                    boy_detected.add(track_id)
                    color = (0, 255, 0)
                elif classes == 'girl' and track_id not in girl_detected:
                    girl_detected.add(track_id)
                    color = (255, 255, 0)
                elif classes == 'man' and id not in man_detected:
                    man_detected.add(track_id)
                    color = (0, 255, 255)
                elif classes == 'woman' and id not in woman_detected:
                    woman_detected.add(track_id)
                    color = (255, 0, 255)
                
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.circle(frame, (box[2], box[3]), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"{classes} - %.2f" % conf, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, (0.5), (255, 255, 255), 2)
                
        cv2.polylines(frame, [np.array(AREA, np.int32)], True, (0, 0, 255), 2)
        
        count_boy = (len(boy_detected))
        count_girl = (len(girl_detected))
        count_man = (len(man_detected))
        count_woman = (len(woman_detected))

        # Cek apakah waktu sudah mencapai 5 detik
        elapsed_time = time.time() - START_TIME
        if elapsed_time >= TIME_TO_RESET:
            # Reset counting
            boy_detected = set()
            girl_detected = set()
            man_detected = set()
            woman_detected = set()
            
            # Ambil max class
            array = [count_boy, count_girl, count_man, count_woman]
            max_class = maxCondition(array)
            ad_product = randomAd(max_class)
            
            # Reset variabel waktu
            START_TIME = time.time()
        
        cv2.putText(frame, "boy counted: " + str(count_boy), (60, 90), cv2.FONT_HERSHEY_COMPLEX, (0.7), (0, 255, 0), 2)
        cv2.putText(frame, "girl counted: " + str(count_girl), (60, 120), cv2.FONT_HERSHEY_COMPLEX, (0.7), (255, 255, 0), 2)
        cv2.putText(frame, "man counted: " + str(count_man), (60, 150), cv2.FONT_HERSHEY_COMPLEX, (0.7), (0, 255, 255), 2)
        cv2.putText(frame, "woman counted: " + str(count_woman), (60, 180), cv2.FONT_HERSHEY_COMPLEX, (0.7), (255, 0, 255), 2)
        cv2.putText(frame, "Advertisement: " + str(ad_product), (800, 60), cv2.FONT_HERSHEY_COMPLEX, (0.6), (230, 216, 173), 2)
        cv2.putText(frame, "FPS: " + str(int(fps)), (1100, 600), cv2.FONT_HERSHEY_SIMPLEX, (0.7), (230, 216, 173), 2)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
    
        # Menggabungkan frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Jeda selama 1/30 detik
        time.sleep(1/30)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(produceVideo(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port = 8079, host='0.0.0.0')