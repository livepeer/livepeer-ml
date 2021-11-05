FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu18.04

# set the github runner version
ARG RUNNER_VERSION="2.284.0"

ENV PYTHONUNBUFFERED=1

# update package list and create user
RUN apt-get update -y && useradd -m devops

# install and configure dependencies and python 3.9
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    curl jq build-essential software-properties-common tzdata \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt install -y python3.9 python3.9-venv python3.9-dev python3-pip git \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 0 \
    && python3 -m pip install --upgrade setuptools pip distlib

# download and install GH actions runner
RUN cd /home/devops && mkdir actions-runner && cd actions-runner \
    && curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && chown -R devops ~devops

# Install OpenCV dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6

# install additional runner dependencies
RUN /home/devops/actions-runner/bin/installdependencies.sh

USER devops

WORKDIR /home/devops/actions-runner

# create script to handle graceful container shutdown
RUN echo -e '\n\
#!/bin/bash \n\
cleanup() { \n\
    ./config.sh remove --unattended --token $(cat .reg-token) \n\
}\n\
trap "cleanup" INT \n\
trap "cleanup" TERM \n\
echo "get new runner token via GitHub API" \n\
curl -sX POST -H "Authorization: token ${ACCESS_TOKEN}" https://api.github.com/repos/${ORGANIZATION}/${REPOSITORY}/actions/runners/registration-token | jq .token --raw-output > .reg-token \n\
echo "configure runner" \n\
./config.sh --url https://github.com/${ORGANIZATION}/${REPOSITORY} --token $(cat .reg-token) --name ml-gpu-runner --labels ml-gpu-runner --unattended \n\
echo "start runner" \n\
./bin/runsvc.sh & wait $! \n\
'>create_and_run.sh

RUN chmod +x create_and_run.sh

ENTRYPOINT ["sh", "./create_and_run.sh"]