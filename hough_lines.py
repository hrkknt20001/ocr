import numpy as np
import cv2
import glob
import os

paths = glob.glob('./direction_sign/org/*.jpg')

for path in paths:
    img = cv2.imread(path) # 画像読み込み
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # グレースケール化
    outLineImage = cv2.Canny(gray, 220, 250, apertureSize = 3)   # 輪郭線抽出

    cv2.imwrite("{0}_canny.png".format(os.path.basename(path)), outLineImage)

    # 確率的ハフ変換で直線を抽出
    lines = cv2.HoughLinesP(outLineImage, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=100)
    #lines = cv2.HoughLinesP(outLineImage, rho=1, theta=np.pi/180, threshold=200, minLineLength=100, maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1) # 緑色で直線を引く

    cv2.imwrite("{0}_hough.png".format(os.path.basename(path)), img)
