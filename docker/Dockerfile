FROM pytorch/pytorch

RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip wget
RUN git clone https://github.com/louis-she/sfd.pytorch.git
RUN cd sfd.pytorch; mv config.py.example config.py
VOLUME ["/log","/data"]
