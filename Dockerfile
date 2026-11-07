FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /opt/piadaptwin

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["piadaptwin"]
CMD ["--help"]
