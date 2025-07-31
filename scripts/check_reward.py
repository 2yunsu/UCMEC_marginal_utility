import scipy.io as scio
import hdf5storage
import numpy as np

print("reward.mat 파일의 키 구조 확인 중...")

try:
    # scipy.io로 시도
    print("\n=== scipy.io.loadmat으로 로드 ===")
    data_scipy = scio.loadmat('./reward.mat')
    print(f"scipy.io 키들: {list(data_scipy.keys())}")
    
    for key in data_scipy.keys():
        if not key.startswith('__'):  # 메타데이터 키 제외
            print(f"  - {key}: shape = {data_scipy[key].shape}, type = {type(data_scipy[key])}")
    
except Exception as e:
    print(f"scipy.io 로드 실패: {e}")

print("\n" + "="*50)

try:
    # hdf5storage로 시도
    print("\n=== hdf5storage.loadmat으로 로드 ===")
    data_hdf5 = hdf5storage.loadmat('./reward.mat')
    print(f"hdf5storage 키들: {list(data_hdf5.keys())}")
    
    for key in data_hdf5.keys():
        if not key.startswith('__'):  # 메타데이터 키 제외
            print(f"  - {key}: shape = {data_hdf5[key].shape}, type = {type(data_hdf5[key])}")
            
            # 데이터 내용 미리보기
            if hasattr(data_hdf5[key], 'flatten'):
                data_flat = data_hdf5[key].flatten()
                if len(data_flat) > 0:
                    print(f"    첫 5개 값: {data_flat[:5]}")
                    print(f"    통계: mean={np.mean(data_flat):.2f}, std={np.std(data_flat):.2f}")
    
except Exception as e:
    print(f"hdf5storage 로드 실패: {e}")

print("\n" + "="*50)

# 파일 존재 여부 확인
import os
if os.path.exists('./reward.mat'):
    file_size = os.path.getsize('./reward.mat')
    print(f"\nreward.mat 파일 정보:")
    print(f"  - 존재: 예")
    print(f"  - 크기: {file_size} bytes ({file_size/1024:.2f} KB)")
else:
    print(f"\nreward.mat 파일이 존재하지 않습니다.")
    
# 현재 디렉토리의 .mat 파일들 확인
mat_files = [f for f in os.listdir('.') if f.endswith('.mat')]
print(f"\n현재 디렉토리의 .mat 파일들: {mat_files}")