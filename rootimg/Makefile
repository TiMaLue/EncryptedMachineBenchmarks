.PHONY: mpcbenchtarget_rootimg mpcbenchtarget_rootimg_py

all: mpcbenchtarget_rootimg mpcbenchtarget_rootimg_py mpcbenchtarget_rootimg_conda

mpcbenchtarget_rootimg: Dockerfile
	docker build -f ./Dockerfile -t mpcbenchtarget_rootimg .

mpcbenchtarget_rootimg_py: Dockerfile.pyroot
	docker build  -f ./Dockerfile.pyroot -t mpcbenchtarget_rootimg_py .

mpcbenchtarget_rootimg_conda: Dockerfile.conda
	docker build  -f ./Dockerfile.conda -t mpcbenchtarget_rootimg_conda .
	