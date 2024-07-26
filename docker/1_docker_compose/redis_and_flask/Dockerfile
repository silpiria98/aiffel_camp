FROM redislabs/redismod AS redis_builder

FROM --platform=$BUILDPLATFORM python:3.10-alpine AS  builder

WORKDIR /code

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt


COPY ./ ./

ENTRYPOINT ["python3"]
CMD [ "app.py" ]

FROM builder AS dev-envs

RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends git
apt-get install bash
EOF

RUN <<EOF
useradd -s /bin/bash -m vscode   
groupadd docker
usermod -aG docker vscode
EOF

COPY --from=gloursdocker/docker / / 

