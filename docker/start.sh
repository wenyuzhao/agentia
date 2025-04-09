#!/usr/bin/env bash


mkdir -p /app/.cache/nginx
if [ -f /app/.cache/nginx/access.log ]; then
    mv /app/.cache/nginx/access.log /app/.cache/nginx/access.log.bak
fi
if [ -f /app/.cache/nginx/error.log ]; then
    mv /app/.cache/nginx/error.log /app/.cache/nginx/error.log.bak
fi
touch /app/.cache/nginx/access.log
touch /app/.cache/nginx/error.log

# Start the first process
uv run agentia serve app &

# Start the second process
uv run agentia serve api &

nginx -g 'daemon off;' &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?