ARG VERSION=RELEASE.2025-08-13T08-35-41Z
FROM minio/mc:${VERSION}

# Create script directory and config directory
RUN mkdir -p /src && install -d -o root -g 1000 -m 775 /config

# Create a wrapper to force mc to use the custom config directory
RUN printf '#!/bin/sh\nexec /usr/bin/mc --config-dir /config "$@"\n' \
        > /usr/local/bin/mc \
    && chmod +x /usr/local/bin/mc

# Copy scripts
COPY docker/minio-client/minio_init.sh /src/minio_init.sh
COPY docker/minio-client/minio_backup.sh /src/minio_backup.sh

# Make scripts executable
RUN chmod +x /src/minio_init.sh /src/minio_backup.sh

ENTRYPOINT ["/bin/sh", "-c"]
