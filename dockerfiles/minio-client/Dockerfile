ARG VERSION

FROM minio/mc:${VERSION}

RUN mkdir -p /src
COPY dockerfiles/minio-client/minio_init.sh /src/minio_init.sh
COPY dockerfiles/minio-client/minio_backup.sh /src/minio_backup.sh

RUN chmod +x /init/init.sh

ENTRYPOINT ["/bin/sh", "-c"]
