ARG VERSION

FROM minio/mc:${VERSION}

RUN mkdir -p /init
COPY dockerfiles/minio-client/init.sh /init/init.sh

RUN chmod +x /init/init.sh

ENTRYPOINT ["/bin/sh", "-c"]
