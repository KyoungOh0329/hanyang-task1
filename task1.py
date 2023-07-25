import cv2
import os

def compare_images(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 이미지가 제대로 로드되었는지 확인
    if image1 is None or image2 is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    # 이미지를 그레이스케일로 변환
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 이미지 유사도 측정
    similarity = cv2.matchTemplate(gray_image1, gray_image2, cv2.TM_CCOEFF_NORMED)
    return similarity[0][0] * 100  # 0부터 1 사이의 값을 0부터 100으로 변환하여 백분율로 표시

if __name__ == "__main__":
    # 비교할 이미지 경로들
    image1_path = "/Users/mazoola12/Downloads/IMG_0935.jpg"
    image_folder = "/Volumes/Make/frame/"

    # 결과를 저장할 딕셔너리
    similarity_scores = {}

    # 폴더 내의 모든 이미지와 비교
    image_files = [filename for filename in os.listdir(image_folder) if filename.endswith(".PNG")]
    image_files.sort()  # 이미지 파일 이름순으로 정렬

    for filename in image_files:
        image2_path = os.path.join(image_folder, filename)
        similarity = compare_images(image1_path, image2_path)
        similarity_scores[filename] = similarity

    # 이름순으로 정렬하여 출력
    sorted_similarities = sorted(similarity_scores.items(), key=lambda x: x[0])  # 이름순으로 정렬
    for filename, similarity in sorted_similarities:
        print(f"{filename}: {similarity:.2f}%")
