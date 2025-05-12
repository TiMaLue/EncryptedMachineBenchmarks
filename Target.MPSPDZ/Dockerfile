FROM python:3.10.3-bullseye as mpcbuild

ARG BOOST_VERSION=1.75.0

RUN apt-get update && apt-get install -y --no-install-recommends \
                automake \
                build-essential \
                clang-11 \
				cmake \
                git \
                libboost-dev \
                libboost-thread-dev \
                libclang-dev \
                libntl-dev \
                libsodium-dev \
                libssl-dev \
                libtool \
                m4 \
                texinfo \
                yasm \
                vim \
                gdb \
                valgrind \
        && rm -rf /var/lib/apt/lists/*

# mpir
COPY --from=initc3/mpir:55fe6a9 /usr/local/mpir/include/* /usr/local/include/
COPY --from=initc3/mpir:55fe6a9 /usr/local/mpir/lib/* /usr/local/lib/
COPY --from=initc3/mpir:55fe6a9 /usr/local/mpir/share/info/* /usr/local/share/info/

ENV MP_SPDZ_HOME /usr/src/MP-SPDZ
WORKDIR $MP_SPDZ_HOME
ARG USER=TiMaLue
ARG REPO=MP-SPDZ
ARG BRANCH=scheduler
# Add head version of branch, to invalidate docker cache when MP-SPDZ head changes
ADD https://api.github.com/repos/$USER/$REPO/git/refs/heads/$BRANCH version.json
RUN git clone --depth 1 --branch $BRANCH https://github.com/$USER/$REPO .
#RUN git submodule init
#RUN git submodule update
#
#WORKDIR $MP_SPDZ_HOME/deps/libOTe
#
#RUN python build.py --setup --boost --relic
#RUN python build.py -- -D ENABLE_RELIC=ON -D ENABLE_ALL_OT=ON

RUN pip install --upgrade pip ipython

ARG arch=native
ARG cxx=clang++-11
ARG use_ntl=0
ARG prep_dir="Player-Data"
ARG ssl_dir="Player-Data"

RUN echo "ARCH = -march=${arch}" >> CONFIG.mine \
        && echo "CXX = ${cxx}" >> CONFIG.mine \
        && echo "USE_NTL = ${use_ntl}" >> CONFIG.mine \
        && echo "MY_CFLAGS += -I/usr/local/include" >> CONFIG.mine \
        && echo "MY_LDLIBS += -Wl,-rpath -Wl,/usr/local/lib -L/usr/local/lib" \
            >> CONFIG.mine \
        && mkdir -p $prep_dir $ssl_dir \
        && echo "PREP_DIR = '-DPREP_DIR=\"${prep_dir}/\"'" >> CONFIG.mine \
        && echo "SSL_DIR = '-DSSL_DIR=\"${ssl_dir}/\"'" >> CONFIG.mine

# ssl keys
ARG cryptoplayers=10
ENV PLAYERS ${cryptoplayers}
RUN ./Scripts/setup-ssl.sh ${cryptoplayers} ${ssl_dir}

# build boost for libOTe
RUN cd /tmp && \
    BOOST_VERSION_MOD=$(echo $BOOST_VERSION | tr . _) && \
    wget https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_MOD}.tar.bz2 && \
    tar --bzip2 -xf boost_${BOOST_VERSION_MOD}.tar.bz2 && \
    cd boost_${BOOST_VERSION_MOD} && \
    ./bootstrap.sh --prefix=/usr/lib && \
    ./b2 install && \
    rm -rf /tmp/*
RUN make libote


ARG gfp_mod_sz=2
RUN echo "MOD = -DGFP_MOD_SZ=${gfp_mod_sz}" >> CONFIG.mine
RUN make clean
RUN make all -j 4
RUN make yao

FROM mpcbenchtarget_rootimg_py

RUN apt-get update && apt-get install -y --no-install-recommends \
                automake \
                build-essential \
                clang-11 \
                git \
                libboost-dev \
                libboost-thread-dev \
                libclang-dev \
                libntl-dev \
                libsodium-dev \
                libssl-dev \
                libtool \
                m4 \
                texinfo \
                yasm \
                vim \
                gdb \
                valgrind \
        && rm -rf /var/lib/apt/lists/*

ENV MP_SPDZ_HOME /usr/src/MP-SPDZ
COPY --from=mpcbuild /usr/src/MP-SPDZ $MP_SPDZ_HOME

# mpir
COPY --from=initc3/mpir:55fe6a9 /usr/local/mpir/include/* /usr/local/include/
COPY --from=initc3/mpir:55fe6a9 /usr/local/mpir/lib/* /usr/local/lib/
COPY --from=initc3/mpir:55fe6a9 /usr/local/mpir/share/info/* /usr/local/share/info/

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN rm /requirements.txt
# scheduler
RUN mkdir /scheduler
COPY /scheduler/smpc_ml_staticscheduler-0.0.1.tar.gz /scheduler/
RUN pip install /scheduler/smpc_ml_staticscheduler-0.0.1.tar.gz

# load config into dockerfile
COPY /scheduler/config.ini /scheduler/
RUN mkdir /wd
COPY . /wd
WORKDIR /wd

RUN $MP_SPDZ_HOME/Scripts/setup-ssl.sh 10 Player-Data


#RUN $MP_SPDZ_HOME/compile.py SimpleFFNN
#WORKDIR $MP_SPDZ_HOME
#RUN ./compile.py -R 64 tutorial
#RUN ./Scripts/dealer-ring.sh tutorial

