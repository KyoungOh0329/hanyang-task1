import cv2

def find_best_matching_frame(big_image, video):
    # 큰 사진의 브레이크 포인트로 특징점을 추출합니다.
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(big_image, None)

    best_frame = None
    max_matches = 0

    # 영상을 읽어서 프레임마다 특징점을 추출합니다.
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray_frame, None)

        # 두 이미지의 특징점을 비교하여 일치하는 것을 찾습니다.
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 일치하는 특징점들의 개수를 기록하고, 이전에 찾은 프레임보다 많을 경우 갱신합니다.
        num_matches = len(good_matches)
        if num_matches > max_matches:
            max_matches = num_matches
            best_frame = frame

    video.release()

    if best_frame is not None:
        # 가장 일치하는 프레임을 이미지로 저장합니다.
        cv2.imwrite("best_matching_frame.jpg", best_frame)
        print("가장 일치하는 프레임을 이미지로 저장했습니다.")
    else:
        print("일치하는 프레임을 찾지 못했습니다.")

if __name__ == "__main__":
    # 큰 사진을 읽습니다.
    big_image = cv2.imread("/Users/mazoola12/Downloads/IMG_0935.jpg")

    # 영상을 읽습니다.
    video = cv2.VideoCapture("/Volumes/Make/test1.mp4")

    # 영상 중에서 가장 일치하는 프레임을 찾아서 이미지로 저장합니다.
    find_best_matching_frame(big_image, video)
