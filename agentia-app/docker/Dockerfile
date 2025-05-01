FROM ghcr.io/astral-sh/uv:bookworm

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y nginx

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --all-groups --all-extras --all-packages --frozen

# Configure nginx
COPY ./agentia-app/docker/nginx.conf /etc/nginx/sites-enabled/default

# Setup the entrypoint
ENV PATH="/app/agentia-app/docker:$PATH"

CMD start