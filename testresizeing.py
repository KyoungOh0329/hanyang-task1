import cv2
import os
import numpy as np

def find_similar_points(img_path, img_folder_path, min_good_matches=10, resize_width=300):
    # 이미지 불러오기
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 이미지 리사이징
    height, width = img.shape[:2]
    ratio = resize_width / width
    new_height = int(height * ratio)
    img_resized = cv2.resize(img, (resize_width, new_height))

    # SIFT 특징점 검출
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_resized, None)

    # 디렉토리 내 모든 PNG 파일 탐색
    similar_images = []
    for filename in os.listdir(img_folder_path):
        if filename.endswith(".PNG"):
            img_path = os.path.join(img_folder_path, filename)
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 이미지 리사이징
            height, width = frame.shape[:2]
            ratio = resize_width / width
            new_height = int(height * ratio)
            frame_resized = cv2.resize(frame, (resize_width, new_height))

            # SIFT 특징점 검출
            kp_frame, des_frame = sift.detectAndCompute(frame_resized, None)

            # 특징점 매칭
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des, des_frame, k=2)

            # 거리가 가까운 매칭 결과 선택
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # 최소 유사점 개수 이상일 경우에만 저장
            if len(good_matches) >= min_good_matches:
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
    image2_path = "/Users/mazoola12/Downloads/IMG_0936.jpg"
    image_folder_path = "/Volumes/Make/frame/"

    print("Image 1:")
    find_similar_points(image1_path, image_folder_path)

    print("\nImage 2:")
    find_similar_points(image2_path, image_folder_path)
