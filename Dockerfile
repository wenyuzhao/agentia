FROM ghcr.io/astral-sh/uv:bookworm

ENV PYTHONUNBUFFERED=1

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --all-groups --frozen

ADD ./setup.sh /app/setup
ENV PATH="/app:$PATH"

CMD ["uv", "run", "agentia", "serve"]