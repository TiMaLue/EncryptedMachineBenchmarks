FROM python:3.9.13-bullseye as encrypted_machines_crypten_env

RUN pip install wheel

RUN git clone https://github.com/facebookresearch/crypten /crypten
RUN cd /crypten && git checkout efe8edad571be1c586d0d9cefc562d562d4e9aa1
RUN pip install /crypten

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt



FROM mpcbenchtarget_rootimg_py

COPY --from=encrypted_machines_crypten_env /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

RUN mkdir /wd

COPY . /wd
WORKDIR /wd
