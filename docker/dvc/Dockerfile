FROM ubuntu:noble

ARG UID=1000
ARG GID=1000

RUN apt update && apt install -y git-all python3 python3-pip python3-venv && ln -sf python3 /usr/bin/python \
    && rm -rf /var/lib/apt/list/*

COPY docker/dvc/requirements.txt .
ENV VIRTUAL_ENV=/myenv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# installation des bibliothèques
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]