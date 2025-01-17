FROM centos:8.4.2105

RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
RUN yum update -y
RUN yum install -y wget mesa-libGL python38
RUN yum install -y gcc make openssl-devel bzip2-devel libffi-devel

# Create an user
ENV username=soumen
RUN useradd -m "$username"
RUN usermod -aG wheel "$username"
# workdir
WORKDIR "/home/$username/oasis"
RUN chmod 777 "/home/$username/oasis"

# copy source
COPY train.py ./
COPY requirements.sh ./
COPY setup_test.py ./
COPY centos_amd64_deps ./centos_amd64_deps
RUN chmod 777 "train.py"
RUN chmod -R 777 "./centos_amd64_deps"

# change user
USER "$username"

# create virtual environment
RUN python3.8 -m venv ./myvenv
ENV PATH="./myvenv/bin:$PATH" 
# install dependencies
RUN bash requirements.sh
# test installations
RUN python setup_test.py

ENTRYPOINT ["bash"]
