# 5명의 유명인사 데이터셋으로 얼굴 감지하기
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

# 주어진 사진에서 하나의 얼굴 추출


def extract_face(filename, required_size=(160, 160)):
    # 파일에서 이미지 불러오기
    image = Image.open(filename)
    # RGB로 변환, 필요시
    image = image.convert('RGB')
    # 배열로 변환
    pixels = asarray(image)
    # 감지기 생성, 기본 가중치 이용
    detector = MTCNN()
    # 이미지에서 얼굴 감지
    results = detector.detect_faces(pixels)
    # 첫 번째 얼굴에서 경계 상자 추출
    x1, y1, width, height = results[0]['box']
    # 버그 수정
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # 얼굴 추출
    face = pixels[y1:y2, x1:x2]
    # 모델 사이즈로 픽셀 재조정
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# 디렉토리 안의 모든 이미지를 불러오고 이미지에서 얼굴 추출


def load_faces(directory):
    faces = list()
    # 파일 열거
    for filename in listdir(directory):
        # 경로
        path = directory + filename
        # 얼굴 추출
        face = extract_face(path)
        # 저장
        faces.append(face)
    return faces

# 이미지를 포함하는 각 클래스에 대해 하나의 하위 디렉토리가 포함된 데이터셋을 불러오기


def load_dataset(directory):
    X, y = list(), list()
    # 클래스별로 폴더 열거
    for subdir in listdir(directory):
        # 경로
        path = directory + subdir + '/'
        # 디렉토리에 있을 수 있는 파일을 건너뛰기(디렉토리가 아닌 파일)
        if not isdir(path):
            continue
        # 하위 디렉토리의 모든 얼굴 불러오기
        faces = load_faces(path)
        # 레이블 생성
        labels = [subdir for _ in range(len(faces))]
        # 진행 상황 요약
        print('>%d개의 예제를 불러왔습니다. 클래스명: %s' % (len(faces), subdir))
        # 저장
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


# 훈련 데이터셋 불러오기
trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
print(trainX.shape, trainy.shape)
# 테스트 데이터셋 불러오기
testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
print(testX.shape, testy.shape)
# 배열을 단일 압축 포맷 파일로 저장
savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)
