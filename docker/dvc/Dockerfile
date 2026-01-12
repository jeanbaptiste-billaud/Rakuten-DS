FROM ubuntu:noble

RUN apt update && apt install -y git-all python3 python3-pip python3-venv && ln -sf python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY docker/dvc/requirements.txt .
ENV VIRTUAL_ENV=/myenv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# installation des bibliothèques
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /src && touch /src/__init__.py
COPY src/utils/generate_data_lineage.py /src
ENV PYTHONPATH="/"

RUN usermod -l trainusr ubuntu #par défaut uid 1000 est attribué au user ubuntu
USER trainuser

ARG WORKDIR=/data
WORKDIR ${WORKDIR}

EXPOSE 8001

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]