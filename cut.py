# 영상을 지정된 프레임 단위로 추출하는 코드

import cv2

video = cv2.VideoCapture('/Volumes/Make/test1.mp4')

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)

count = 0

while (video.isOpened()):
    ret, image = video.read()

    if (int(video.get(1)) % 10 == 0):
        print('Saved frame number : ' + str(int(video.get(1))))
        count_str = format(count, '06')
        cv2.imwrite("/Volumes/Make/frame/frame_%s.PNG" % count_str, image)
        count += 1

video.release()