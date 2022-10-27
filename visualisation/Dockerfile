FROM mpcbenchtarget_rootimg_py:latest


# RUN set -eux; \
# 	apt-get update; \
# 	apt-get install -y --no-install-recommends \
# 		libmysqlclient-dev python-dev \
# 	; \
# 	rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

RUN mkdir /.local
RUN mkdir /.config
RUN mkdir /wd
RUN chown 1000:1000 /.local /wd /.config
# COPY . /wd
WORKDIR /wd
