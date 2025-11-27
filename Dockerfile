# 1. Base Image: Ray + PyTorch + GPU 지원 버전 사용
# (사용하는 Ray 버전에 맞춰 태그를 선택하세요. 예: 2.9.0)
FROM rayproject/ray-ml:2.9.0-py310-gpu

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 설치 (캐싱 활용을 위해 코드 복사 전에 수행)
# requirements.txt 파일이 변경될 때만 이 레이어가 재실행됩니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
COPY moe_train.py .
# 추가적인 코드 파일이나 폴더가 있다면 여기에 추가
# COPY src/ ./src/

# 5. 권한 설정 (Ray 이미지는 'ray' 유저로 실행되는 것이 권장됨)
USER ray

# KubeRay가 entrypoint를 덮어씌우므로 CMD는 비워두거나 기본 쉘로 둡니다.
CMD ["/bin/bash"]
