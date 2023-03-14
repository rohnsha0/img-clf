import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

__classNameToNum = {}
__classNumToName = {}
__model = None


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1, len_image_array).astype(float)
        #result.append(class_number_to_name(__model.predict(final)[0]))
        #"""
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __classNameToNum
        })
        #"""

    return result


def class_number_to_name(class_num):
    return __classNumToName[class_num]


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __classNameToNum
    global __classNumToName

    with open(r"C:\Users\rahul\Desktop\img-cl\server\artifacts\classDict.json", "r") as f:
        __classNameToNum = json.load(f)
        __classNumToName = {v: k for k, v in __classNameToNum.items()}

    global __model
    if __model is None:
        with open(r"C:\Users\rahul\Desktop\img-cl\server\artifacts\best_model.pkl", 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(image_base64_data):
    try:
        decoded_data = base64.b64decode(image_base64_data)
        np_data = np.fromstring(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img, img.shape
    except:
        return None, None


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier(
        r"C:\Users\rahul\Desktop\img-cl\server\opencv\haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(r"C:\Users\rahul\Desktop\img-cl\server\opencv\haarcascade_eye.xml")

    image_data, image_shape = get_cv2_image_from_base64_string(image_base64_data)
    if image_data is None or image_shape is None:
        return None

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()


if __name__ == "__main__":
    load_saved_artifacts()
    #print(classify_image(get_b64_test_image_for_virat(), None))
    #print(classify_image(None, "./testIMGs/Sundar_Pichai-1019x573.webp"))
    print(classify_image(None, "./testIMGs/images (8).jpg"))
    print(classify_image(None, "./testIMGs/images (3).jpg"))
    print(classify_image(None, "./testIMGs/images (4).jpg"))
    print(classify_image(None, "./testIMGs/images (5).jpg"))
    print(classify_image(None, "./testIMGs/images (6).jpg"))