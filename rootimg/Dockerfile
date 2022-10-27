FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		netbase \
		wget \
        build-essential \
		software-properties-common \
		dirmngr \
		iputils-ping iproute2 \
	; \
	rm -rf /var/lib/apt/lists/*

RUN set -ex; \
	if ! command -v gpg > /dev/null; then \
		apt-get update; \
		apt-get install -y --no-install-recommends \
			gnupg \
			dirmngr \
		; \
		rm -rf /var/lib/apt/lists/*; \
	fi

COPY watch_packets.bash /bin/watch_packets.bash
COPY entrypoint.bash /bin/entrypoint.bash
RUN chmod +x /bin/watch_packets.bash
RUN chmod +x /bin/entrypoint.bash

ENV LO_ENABLE_MTU_LIMIT=0
ENV LO_MTU=1500
ENV LO_ENABLE_TC=0
ENV LO_DELAY_MS=0
ENV LO_RATE_MBIT=100000

CMD [ "/bin/entrypoint.bash", "1"]


# Tag mpcbenchtarget_rootimg

