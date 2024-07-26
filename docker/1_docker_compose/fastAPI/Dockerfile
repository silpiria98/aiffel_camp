# 베이스 이미지를 빌더로
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim AS builder

# 작업 폴더 설정
WORKDIR /app

# 파일 복사
COPY requirements.txt ./

# pip 캐시 유지 및 패키지 설치
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# 로컬의 app폴더를 컨테이너 app폴더로 복사
COPY ./app ./app

# 빌드단계 시작
FROM builder as dev-envs

# 패키지 업데이트 및 깃 설치
RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends git
EOF

# vscode라는 유저 생성 및 그룹에 추가 
RUN <<EOF
useradd -s /bin/bash -m vscode   
groupadd docker
usermod -aG docker vscodeEOF
EOF

# gloursdocker/docker의 모든 내용을 복사
COPY --from=gloursdocker/docker / /