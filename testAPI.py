import cv2
import time
import base64
import requests
import json                    

def testapi():
    frame = cv2.imread('img/1.jpg')
    _, encimg = cv2.imencode(".jpg", frame)
    img_byte = encimg.tobytes()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    new_img_str = "data:image/jpeg;base64," + img_str
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}

    payload = json.dumps({"img": new_img_str})

    seconds = time.time()
    response = requests.post('http://127.0.0.1:9999/dream', data=payload, headers=headers, timeout=100)

    try:
        print('Server response', response.json()['result']['feats'])

    except requests.exceptions.RequestException:
        print(response.text)

    return
if __name__ == '__main__':
    testapi()