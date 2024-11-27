#!/bin/bash

chrome_diagnostic() {
  rm -rf /tmp/chrome_testing_run/
  mkdir -p /tmp/chrome_testing_run/user_data
  XVFB_OUTPUT_FILE="/tmp/chrome_testing_run/xvfb.out"
  CHROME_OUTPUT_FILE="/tmp/chrome_testing_run/chrome.out"
  SCREENSHOT_FILE="/tmp/chrome_testing_run/screenshot.png"
  USER_DATA_DIR="/tmp/chrome_testing_run/user_data"

  Xvfb :99999 >"$XVFB_OUTPUT_FILE" 2>&1 &
  XVFB_PID=$!

  sleep 1
  if ! ps -p "$XVFB_PID" > /dev/null; then
    echo "Chrome diagnostic failed (Xvfb running)" >&2
    cat "$XVFB_OUTPUT_FILE" >&2
    return 1
  fi

  "$CHROME_BIN" '--remote-allow-origins=*' \
    --no-first-run \
    --no-service-autorun \
    --no-default-browser-check \
    --homepage=about:blank \
    --no-pings \
    --password-store=basic \
    --disable-infobars \
    --disable-breakpad \
    --disable-component-update \
    --disable-backgrounding-occluded-windows \
    --disable-renderer-backgrounding \
    --disable-background-networking \
    --disable-dev-shm-usage \
    --disable-features=IsolateOrigins,site-per-process \
    --disable-session-crashed-bubble \
    --disable-search-engine-choice-screen \
    --user-data-dir=/tmp/chrome_testing_run/ \
    --disable-features=IsolateOrigins,site-per-process \
    --disable-session-crashed-bubble \
    --no-sandbox \
    --remote-debugging-host=127.0.0.1 \
    --remote-debugging-port=44444 \
    --user-data-dir="$USER_DATA_DIR" \
    --timeout=60 \
    --window-size=1920,1200 \
    --headless \
    --screenshot="$SCREENSHOT_FILE" \
    "https://www.google.com" \
    >"$CHROME_OUTPUT_FILE" 2>&1 &
  CHROME_PID=$!

  START_TIME=$(date +%s)
  WAIT_TIMEOUT=30
  EXIT_CODE=1

  while true
  do
    CUR_TIME=$(date +%s)
    if [[ $((CUR_TIME - START_TIME)) -gt "$WAIT_TIMEOUT" ]]; then
      break
    fi
    if ! ps -p "$CHROME_PID" > /dev/null; then
      break
    fi
    if [ -f "$SCREENSHOT_FILE" ]; then
      EXIT_CODE=0
      break
    fi
    sleep 1
  done

  if [ -f "$SCREENSHOT_FILE" ]; then
    EXIT_CODE=0
  fi

  if [[ $EXIT_CODE == 0 ]]
  then
    echo "Chrome diagnostic success"
  else
    echo "Chrome diagnostic failed (chrome running)" >&2
    cat "$CHROME_OUTPUT_FILE" >&2
  fi

  kill "$CHROME_PID" 2>/dev/null
  wait "$CHROME_PID"

  kill "$XVFB_PID" 2>/dev/null
  wait "$XVFB_PID"

  return $EXIT_CODE
}

set -o pipefail

CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

export IN_DOCKER=true
export WORKSPACE_ROOT=/opt/yandex_captcha_puzzle_solver/var/
export PYTHONPATH=$PYTHONPATH:/opt/yandex_captcha_puzzle_solver/lib/
CHROME_BIN=$(which chrome || which chromium)

if [ "$CHROME_BIN" = "" ] ; then
  echo "Can't find chrome executable" >&2
  exit 1
fi

sudo -n find "$WORKSPACE_ROOT" -exec chown "$CURRENT_UID:$CURRENT_GID" {} \;
mkdir -p "$WORKSPACE_ROOT/log"

# Non critical - simple make chrome happy and disable some its errors.
# Start dbus for exclude chrome errors:
# Failed to connect to the bus: Failed to connect to socket /run/dbus/system_bus_socket: No such file or directory
# Failed to connect to the bus: Could not parse server address: Unknown address type
XDG_RUNTIME_DIR=/run/xdg/
sudo bash -c "
sudo service dbus start
mkdir -p '$XDG_RUNTIME_DIR'
chmod 700 '$XDG_RUNTIME_DIR'
chown '$(id -un):$(id -gn)' '$XDG_RUNTIME_DIR'"
DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
dbus-daemon --session --address="$DBUS_SESSION_BUS_ADDRESS" --nofork --nopidfile --syslog-only &

# Run diagnostic if required
if [ "$CHECK_SYSTEM" = true ] ; then
  chrome_diagnostic || exit 1
fi

# Start grounding server - web server, that will fill fake captcha form.
python3 /opt/yandex_captcha_puzzle_solver/bin/grounding_server/grounding_server.py \
  --port=9001 \
  --page-template=/opt/yandex_captcha_puzzle_solver/etc/html_templates/index.html.j2 \
  --form-page-template=/opt/yandex_captcha_puzzle_solver/etc/html_templates/form.html.j2 \
  >"$WORKSPACE_ROOT/log/grounding_server.log" 2>&1 &

# Up default proxy, that will be used for solve without proxy defined in request.
bash /opt/yandex_captcha_puzzle_solver/bin/YandexCaptchaPuzzleSolverProxyRun.sh \
  10000 "http://localhost:9001" "" "$WORKSPACE_ROOT/log/" \
  >"$WORKSPACE_ROOT/log/yandex_proxy_run.out" 2>&1 &

# Run service
ADD_PARAMS=""
if [ "$CHROME_DISABLE_GPU" = true ] ; then
  ADD_PARAMS="$ADD_PARAMS --disable-gpu"
fi

if [ "$VERBOSE" = true ] ; then
  ADD_PARAMS="$ADD_PARAMS --verbose"
fi

if [ "$DEBUG" = true ] ; then
  mkdir -p "$WORKSPACE_ROOT/debug"
  ADD_PARAMS="$ADD_PARAMS --debug-dir=$WORKSPACE_ROOT/debug"
fi

echo "Run server $(pip show yandex-captcha-puzzle-solver | grep Version | awk '{print $2}'
), chrome: $("$CHROME_BIN" --version)"

yandex_captcha_puzzle_solve_server \
  -b 0.0.0.0:8080 \
  --proxy http://127.0.0.1:10000 \
  --proxy-listen-start-port 10001 \
  --proxy-listen-end-port 20000 \
  --proxy-command 'bash /opt/yandex_captcha_puzzle_solver/bin/YandexCaptchaPuzzleSolverProxyRun.sh {{LOCAL_PORT}} "http://localhost:9001" "{{UPSTREAM_URL}}"' \
  $ADD_PARAMS \
  2>&1 | \
  tee "$WORKSPACE_ROOT/log/yandex_captcha_puzzle_solver.log"
