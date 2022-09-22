FROM jupyter/scipy-notebook as test

USER root
RUN mkdir -p /source

COPY . /source


RUN apt-get update -y && \
  apt-get install -y build-essential && \
  apt-get install -y curl

WORKDIR /source

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
  && python get-pip.py && rm get-pip.py

RUN pip install virtualenv && \
    virtualenv venv && \
    source venv/bin/activate

RUN make install-e
RUN make install-all

ENTRYPOINT ["make", "test"]
