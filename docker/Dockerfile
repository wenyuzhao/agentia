FROM ghcr.io/astral-sh/uv:bookworm

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y nginx

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --all-groups --frozen

ADD ./docker/setup.sh /app/setup
ENV PATH="/app:$PATH"

COPY ./docker/nginx.conf /etc/nginx/conf.d/default.conf

CMD ./docker/start.sh