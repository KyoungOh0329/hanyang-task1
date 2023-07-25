import cv2
import os
import numpy as np

def find_similar_points(img1_path, img_folder_path):
    # 이미지 불러오기
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    # SIFT 특징점 검출
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)

    # 디렉토리 내 모든 PNG 파일 탐색
    similar_images = []
    for filename in os.listdir(img_folder_path):
        if filename.endswith(".PNG"):
            img_path = os.path.join(img_folder_path, filename)
            img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # SIFT 특징점 검출
            kp2, des2 = sift.detectAndCompute(img2, None)

            # 특징점 매칭
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # 거리가 가까운 매칭 결과 선택
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # 유사점 개수 저장
            similar_images.append((img_path, len(good_matches)))

    # 유사점 개수를 기준으로 정렬
    similar_images.sort(key=lambda x: x[1], reverse=True)

    # 가장 유사한 이미지 출력
    if len(similar_images) > 0:
        best_match_path, best_match_count = similar_images[0]
        print("가장 유사한 이미지:", best_match_path)
        print("유사점 개수:", best_match_count)
        best_match_img = cv2.imread(best_match_path)
        cv2.imshow("Best Match", best_match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("유사한 이미지를 찾을 수 없습니다.")

if __name__ == "__main__":
    image1_path = "/Users/mazoola12/Downloads/IMG_0935.jpg"
    image_folder_path = "/Volumes/Make/frame/"
    find_similar_points(image1_path, image_folder_path)
