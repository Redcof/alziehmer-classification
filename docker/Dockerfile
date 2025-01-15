FROM tensorflow/tensorflow:2.13

# RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
# RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
# RUN yum update -y
# RUN yum install -y wget mesa-libGL
# RUN yum install -y gcc make openssl-devel bzip2-devel libffi-devel

# Create an user
ENV username=soumen
RUN useradd -m "$username"
RUN usermod -aG wheel "$username"
USER "$username"
# workdir
WORKDIR "/home/$username/oasis"
RUN chmod 777 "/home/$username/oasis"

# copy source
COPY train.py ./
COPY kaggle.json ./
COPY requirements.sh ./

# # install dependencies
RUN python -m venv ./myvenv
ENV PATH="./myvenv/bin:$PATH" 
# tensorflow_datasets torch
RUN sh requirements.sh

ENTRYPOINT ["bash"]