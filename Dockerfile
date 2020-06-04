FROM nvidia/cuda:10.0-devel-ubuntu16.04
MAINTAINER Edith Langer#
RUN apt-get update -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:jonathonf/python-3.6 -y
RUN apt-get update -y
RUN apt-get install python3.6 python3.6-dev -y
RUN apt-get update && apt-get install python3-pip -y
RUN rm /usr/bin/python3 && ln -s python3.6 /usr/bin/python3
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl 
RUN pip3 install plyfile pillow scipy==1.2.1
RUN apt-get install libsparsehash-dev
#RUN apt-get install cifs-utils




#nvidia-docker run -it --network my-net --shm-size 20gb -v /home/edith/Software/SparseConvNet:/code nvidia/cuda bash 

#mount -t cifs -o credentials=/home/edith/.v4rtempcredentials //v4r-nas.acin.tuwien.ac.at/v4rtemp/ /usr/mount/v4rtemp/

#docker build -t sparseconvnet:001 --network my-net .				#builds the image, looks for a Dockerfile automatically, you can also specify a path
#nvidia-docker run  -it --shm-size 20gb -v /home/edith/Software/SparseConvNet:/code -v /usr/mount/v4rtemp:/usr/mount/v4rtemp --network my-net sparseconvnet:001 bash
#cd code && python3 setup.py develop


