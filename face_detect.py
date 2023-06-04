import cv2
import sys

image_file = None
cascade_file = './haarcascades/haarcascade_frontalface_alt.xml'

args = sys.argv

if len(args) > 1:
    image_file = args[1]

image = cv2.imread(image_file)

cv2.imshow('image', image)
cv2.waitKey(0)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 画像を白黒に変換する
cv2.imshow('image', image_gray)
cv2.waitKey(0)

cascade = cv2.CascadeClassifier(cascade_file)
# 検出する顔の最小サイズを指定する
front_face_list = cascade.detectMultiScale(image_gray, minSize = (30, 30))

print(front_face_list)
# 検出した顔の座標情報を取得、出力
# 検出した顔の矩形の大きさを指定する
if len(front_face_list):
    for (x,y,w,h) in front_face_list:
        print("[x,y] = %d,%d [w,h] = %d,%d" %(x, y, w, h))
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), thickness=3)
    cv2.imshow('image', image)
    cv2.waitKey(0)
else:
    print('not detected')


# 検出した顔を切り出してカラーで保存する
for i, (x, y, w, h) in enumerate(front_face_list):
    trim = image[y: y+h, x:x+w]
    cv2.imwrite('output/face' + str(i+1) + '.jpg', trim)