COPY env-main.yaml /env-main.yaml
RUN conda env create -f /env-main.yaml

RUN set -eux; \
	apt-get update; \
	apt-get install -y --install-recommends \
		git \
    	build-essential \
    	libtool \
    	autoconf \
	; \
	rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tf-encrypted/tf-encrypted /tf-encrypted
WORKDIR /tf-encrypted
RUN git checkout ed3e3ffc2c0e903feb1dd7faaadce32aeedd7fa9

RUN conda run -n main pip install -e . --no-deps

RUN CPATH="/opt/conda/envs/main/lib/python3.6/site-packages/tensorflow_core/include" conda run -n main make build

#RUN cp /tf-encrypted/tf_encrypted/operations/aux/aux_module_tf_.so \
#    /tf-encrypted/tf_encrypted/operations/aux/aux_module_tf_1.15.5.so

RUN mkdir /wd
COPY . /wd
WORKDIR /wd
