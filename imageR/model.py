import face_recognition
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont

def img2emotion(img_path, p, model):
    # 0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral
    # emotion_dict = {'愤怒': 0, '悲伤': 5, '中性': 4, '厌恶': 1, '惊喜': 6, '害怕': 2, '高兴': 3}
    # emo_dict = {'愤怒': 2, '悲伤': 2, '中性': 1, '厌恶': 2, '惊喜': 0, '害怕': 2, '高兴': 0}
    # res_map = {0: "消极", 1: "消极", 2: "消极", 3:"积极", 4: "中性", 5: "消极", 6:"积极"}

    emotion_dict = {'愤怒': 0, '悲伤': 4, '中性': 6, '厌恶': 1, '惊喜': 5, '害怕': 2, '高兴': 3}
    emo_dict = {'愤怒': 2, '悲伤': 2, '中性': 1, '厌恶': 2, '惊喜': 0, '害怕': 2, '高兴': 0}
    res_map = {0: "消极", 1: "消极", 2: "消极", 3: "积极", 4: "消极", 5: "积极", 6: "中性"}
    img = cv2.imread(img_path)

    face_locations = face_recognition.face_locations(img)
    if len(face_locations):
        top, right, bottom, left = face_locations[0]
    else:
        return None, None
    face_image = img[top:bottom, left:right]

    face_image = cv2.resize(img, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    predicted_class = np.argmax(model.predict(face_image))
    print(predicted_class)
    label_map = dict((v, k) for k, v in emotion_dict.items())
    predicted_label = label_map[predicted_class]
    res_predicted_label = res_map[predicted_class]


    # 存储结果
    res = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    res = cv2AddChineseText(res, res_predicted_label, (right, top-50), (255, 0, 0))

    _, img_name = os.path.split(img_path)
    cv2.imwrite(os.path.join(p.img_results_path, img_name), res)
    predicted_number = emo_dict[predicted_label]
    print(img_path)
    return res_predicted_label, predicted_number


def cv2AddChineseText(img, text, position, textColor=(255, 0, 0), textSize=40):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)