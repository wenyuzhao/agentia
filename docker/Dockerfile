FROM ghcr.io/astral-sh/uv:bookworm

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y nginx

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --all-extras --all-groups --frozen

# Install `setup` command
ENV PATH="/app/docker:$PATH"

# Configure nginx
COPY ./docker/nginx.conf /etc/nginx/sites-enabled/default

CMD start