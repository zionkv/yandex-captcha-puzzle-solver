#!/bin/bash

LOCAL_PORT="$1"
GROUND_URL="$2"
PROXY="$3"
LOG_DIR="$4" # Log dir.

clean_up() {
  rm -rf "$1"
}

if [ "$LOG_DIR" = "" ] ; then
  LOG_DIR=$(mktemp -d -t proxy.XXXXXX)
  trap "clean_up '$LOG_DIR'" EXIT
fi

# Grounding server should be runned on 9001 port

if [ "$PROXY" != "" ] ; then
  gost -L=http://127.0.0.1:$((LOCAL_PORT + 2000)) -F=$PROXY &
  pids+=($!)
else
  gost -L=http://127.0.0.1:$((LOCAL_PORT + 2000)) &
  pids+=($!)
fi

# GroundingProxy: proxy that convert proxy traffic to http and send it to GroundingServer.
mitmdump --mode regular --listen-port "$((LOCAL_PORT + 1000))" \
  -s /opt/yandex_captcha_puzzle_solver/lib/mitm_addons/mitm_grounding_addon.py \
  --set ground_url=http://localhost:9001 \
  >"$LOG_DIR/mitmproxy_closing.log" 2>&1 &

pids+=($!)

# SplitProxy: proxy that split traffic:
#   Url's with solver_intercept argument to GroundingProxy
#   Other to external network
mitmdump --listen-port "$LOCAL_PORT" --ssl-insecure \
  -s /opt/yandex_captcha_puzzle_solver/lib/mitm_addons/mitm_split_addon.py \
  --mode "upstream:http://localhost:$((LOCAL_PORT + 1000))" \
  --set proxy=localhost:$((LOCAL_PORT + 2000)) \
  >"$LOG_DIR/mitmproxy_splitter.log" 2>&1 &

pids+=($!)

for pid in "${pids[@]}"; do
  wait "${pid}"
done
