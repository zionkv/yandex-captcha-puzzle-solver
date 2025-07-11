import os
import sys
import re
import typing
import typing_extensions
import datetime
import copy
import platform
import uuid
import pathlib
import traceback
import logging
import argparse
import urllib3.util
import fastapi
import pydantic

import yandex_captcha_puzzle_solver

logger = logging.getLogger(__name__)

USE_GUNICORN = (
  sys.platform not in ['win32', 'cygwin'] and 'YANDEX_SOLVER_USE_UVICORN' not in os.environ
)

if USE_GUNICORN:
  import gunicorn.app.wsgiapp
else:
  import uvicorn.main

# Remove requirement for Content-Type header presence.


class RemoveContentTypeRequirementMiddleware(object):
  def __init__(self, app):
    self._app = app

  async def __call__(self, scope, receive, send):
    headers = scope["headers"]
    content_type_found = False
    for header_index, header in enumerate(headers):
      if not isinstance(header, tuple) or len(header) != 2:
        # Unexpected headers format - don't make something.
        content_type_found = True
        break
      if header[0].decode('utf-8').lower() == 'content-type':
        headers[header_index] = (b'content-type', b'application/json')
        content_type_found = True
        break
    if not content_type_found:
      headers.append((b'content-type', b'application/json'))

    return await self._app(scope, receive, send)


server = fastapi.FastAPI(
  openapi_url='/docs/openapi.json',
  docs_url='/docs',
  swagger_ui_parameters={"defaultModelsExpandDepth": -1},
  tags_metadata=[]
)

server.add_middleware(RemoveContentTypeRequirementMiddleware)

PROXY_ANNOTATION = """Proxy in format: <protocol>://(<user>:<password>@)?<host>:<port> .
Examples: socks5://1.1.1.1:2000, http://user:password@1.1.1.1:8080.
For yandex solver compatibility allowed format:
{"url": "<protocol>://<host>:<port>", "username": "<username>", "port": "<port>"}
If you use proxy with authorization and use yandex-captcha-puzzle-solver as package, please,
read instructions - need to install gost."""

solver_args = {
  'proxy_controller': None,
  'disable_gpu': False,
  'debug_dir': None
}

# Simple in-memory queue for compatibility API
_tasks: dict[str, typing.Optional[HandleCommandResponse]] = {}
_task_counter = 0


class ProxyModel(pydantic.BaseModel):
  url: str = pydantic.Field(default=None, description='Proxy url')
  username: str = pydantic.Field(default=None, description='Proxy authorization username')
  password: str = pydantic.Field(default=None, description='Proxy authorization password')


class CookieModel(pydantic.BaseModel):
  name: str = pydantic.Field(description='Cookie name')
  value: str = pydantic.Field(description='Cookie value (empty string if no value)')
  domain: str = pydantic.Field(description='Cookie domain')  # < Is required - we don't allow super cookies usage.
  port: typing.Optional[int] = pydantic.Field(default=None, description='Cookie port')
  path: typing.Optional[str] = pydantic.Field(default='/', description='Cookie path')
  secure: typing.Optional[bool] = pydantic.Field(default=True, description='Cookie is secure')
  expires: typing.Optional[int] = pydantic.Field(
    default=None, description='Cookie expire time in seconds after epoch start'
  )


class HandleCommandResponseSolution(pydantic.BaseModel):
  status: str
  url: str
  cookies: list[CookieModel] = pydantic.Field(default=[], description='Cookies got after solving')
  user_agent: typing.Optional[str] = None
  token: typing.Optional[str] = None


class HandleCommandResponse(pydantic.BaseModel):
  status: str
  message: str
  startTimestamp: float
  endTimestamp: float
  solution: typing.Optional[HandleCommandResponseSolution] = None


async def process_solve_request(
  url: str,
  yandex_key: str,
  cookies: list[CookieModel] = None,
  max_timeout: int = None,  # in msec.
  proxy: typing.Union[str, ProxyModel] = None,
):
  start_timestamp = datetime.datetime.timestamp(datetime.datetime.now())

  # Adapt proxy format for canonical representation.
  if proxy is not None and not isinstance(proxy, str):
    if proxy.url is not None:
      parsed_proxy = urllib3.util.parse_url(proxy.url)
      proxy = (
        parsed_proxy.scheme + "://" +
        (
          proxy.username + ":" + (proxy.password if proxy.password else '') + '@'
          if proxy.username else ''
        ) +
        parsed_proxy.hostname +
        (":" + str(parsed_proxy.port) if parsed_proxy.port else '')
      )
    else:
      proxy = None

  try:
    solve_request = yandex_captcha_puzzle_solver.Request()
    solve_request.yandex_key = yandex_key
    solve_request.url = url
    solve_request.cookies = [
      (cookie if isinstance(cookie, dict) else cookie.__dict__)
      for cookie in cookies
    ] if cookies else []
    solve_request.max_timeout = max_timeout * 1.0 / 1000
    solve_request.proxy = proxy

    global solver_args
    local_solver_args = copy.copy(solver_args)
    if local_solver_args['debug_dir']:
      debug_dir = os.path.join(local_solver_args['debug_dir'], str(uuid.uuid4()))
      pathlib.Path(debug_dir).mkdir(parents=True, exist_ok=True)
      local_solver_args['debug_dir'] = debug_dir
    solver = yandex_captcha_puzzle_solver.Solver(
      **local_solver_args)
    solve_response = await solver.solve(solve_request)

    return HandleCommandResponse(
      status="ok",
      message=solve_response.message,
      startTimestamp=start_timestamp,
      endTimestamp=datetime.datetime.timestamp(datetime.datetime.now()),
      solution=HandleCommandResponseSolution(
        status="ok",
        url=solve_response.url,
        cookies=[  # Convert cookiejar.Cookie to CookieModel
          CookieModel(**cookie) for cookie in solve_response.cookies
        ],
        # < pass cookies as dict's (solver don't know about rest model).
        user_agent=solve_response.user_agent,
        message=solve_response.message,
        token=solve_response.token
      )
    )

  except Exception as e:
    print(str(e))
    print(traceback.format_exc(), flush=True)
    return HandleCommandResponse(
      status="error",
      message="Error: " + str(e),
      startTimestamp=start_timestamp,
      endTimestamp=datetime.datetime.timestamp(datetime.datetime.now()),
    )


# REST API methods.
@server.post(
  "/get_token", response_model=HandleCommandResponse, tags=['Standard API'],
  response_model_exclude_none=True
)
async def Get_cookies_after_solve(
  url: typing_extensions.Annotated[
    str,
    fastapi.Body(description="Url for solve challenge.")
  ],
  yandex_key: typing_extensions.Annotated[
    str,
    fastapi.Body(description="Yandex captcha key")
  ],
  cookies: typing_extensions.Annotated[
    typing.List[CookieModel],
    fastapi.Body(description="Cookies to send.")
  ] = None,
  maxTimeout: typing_extensions.Annotated[
    float,
    fastapi.Body(description="Max processing timeout in ms.")
  ] = 60000,
  proxy: typing_extensions.Annotated[
    typing.Union[str, ProxyModel],
    fastapi.Body(description=PROXY_ANNOTATION)
  ] = None,
):
  return await process_solve_request(
    url=url,
    yandex_key=yandex_key,
    cookies=cookies,
    max_timeout=maxTimeout,
    proxy=proxy,
  )


# Compatibility API
async def _solve_background(task_id: str, **kwargs):
  global _tasks
  result = await process_solve_request(**kwargs)
  _tasks[task_id] = result


@server.post('/in.php')
async def create_task(request: fastapi.Request, background_tasks: fastapi.BackgroundTasks):
  params = dict(request.query_params)
  if request.headers.get('content-type', '').startswith('application/json'):
    params.update(await request.json())
  else:
    try:
      form = await request.form()
      params.update(form)
    except Exception:
      pass

  url = params.get('pageurl') or params.get('url')
  yandex_key = (
    params.get('yandex_key') or params.get('yandexkey') or params.get('sitekey')
  )
  cookies = params.get('cookies')
  if isinstance(cookies, str):
    try:
      cookies = fastapi.json.loads(cookies)
    except Exception:
      cookies = None
  max_timeout = int(params.get('maxTimeout') or params.get('max_timeout') or 60000)
  proxy = params.get('proxy')

  global _task_counter, _tasks
  _task_counter += 1
  task_id = str(_task_counter)
  _tasks[task_id] = None

  background_tasks.add_task(
    _solve_background,
    task_id,
    url=url,
    yandex_key=yandex_key,
    cookies=cookies,
    max_timeout=max_timeout,
    proxy=proxy,
  )

  if params.get('json') == '1':
    return {'status': 1, 'request': task_id}
  return fastapi.responses.PlainTextResponse('OK|' + task_id)


@server.get('/res.php')
async def get_result(id: str, json: int = 0):
  if id not in _tasks:
    if json:
      return {'status': 0, 'request': 'ERROR_WRONG_ID'}
    return fastapi.responses.PlainTextResponse('ERROR_WRONG_ID')

  result = _tasks[id]
  if result is None:
    if json:
      return {'status': 0, 'request': 'CAPCHA_NOT_READY'}
    return fastapi.responses.PlainTextResponse('CAPCHA_NOT_READY')

  token = result.solution.token if result.solution else None
  if json:
    return {'status': 1, 'request': token}
  return fastapi.responses.PlainTextResponse('OK|' + str(token))


@server.get('/balance')
@server.get('/getBalance')
async def get_balance(json: int = 0):
  if json:
    return {'status': 1, 'request': '0'}
  return fastapi.responses.PlainTextResponse('0')


def server_run():
  try:
    logging.basicConfig(
      format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
      handlers=[logging.StreamHandler(sys.stdout)],
      level=logging.INFO
    )

    logging.getLogger('urllib3').setLevel(logging.ERROR)

    logger.info(
      "Start yandex_captcha_puzzle_server:\n" +
      "  version: " + str(yandex_captcha_puzzle_solver.__version__) + "\n" +
      "  python version = " + ".".join([str(x) for x in list(sys.version_info)]) + "\n" +
      "  os = " + " ".join([platform.system(), platform.release(), platform.version()]) + "\n" +
      "  docker = " + os.environ.get('IN_DOCKER', "false") + "\n" +
      "  arch = " + str(platform.machine()) + "\n" +
      "  processor = " + str(platform.processor())
    )

    parser = argparse.ArgumentParser(
      description='Start yandex captcha puzzle solve server with /get_token and compatibility /in.php API.',
      epilog='Other arguments will be passed to gunicorn or uvicorn(win32) as is.')
    parser.add_argument("-b", "--bind", type=str, default='127.0.0.1:8000')
    # < parse for pass to gunicorn as is and as "--host X --port X" to uvicorn
    parser.add_argument(
      "--proxy-listen-start-port", type=int, default=10000,
      help="""Port interval start, that can be used for up local proxies on request processing"""
    )
    parser.add_argument(
      "--proxy-listen-end-port", type=int, default=20000,
      help="""Port interval end for up local proxies"""
    )
    parser.add_argument(
      "--proxy-command", type=str,
      default=None,
      help="""command template (jinja2), that will be used for up proxy for process request
      with arguments: LOCAL_PORT, UPSTREAM_URL - proxy passed in request"""
    )
    parser.add_argument("--disable-gpu", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument(
      "--debug-dir", type=str, default=None,
      help="""directory for save intermediate DOM dumps and screenshots on solving,
      for each request will be created unique directory"""
    )
    parser.add_argument("--proxy", type=str)
    parser.set_defaults(disable_gpu=False, debug=False)
    args, unknown_args = parser.parse_known_args()
    try:
      host, port = args.bind.split(':')
    except Exception:
      print("Invalid 'bind' argument value: " + str(args.bind), file=sys.stderr, flush=True)
      sys.exit(1)

    if args.verbose:
      logging.getLogger('zendriver.core.browser').setLevel(logging.DEBUG)
      logging.getLogger('yandex_captcha_puzzle_solver.yandex_captcha_puzzle_solver').setLevel(logging.DEBUG)
      logging.getLogger('uc.connection').setLevel(logging.INFO)

    global solver_args

    if args.debug_dir:
      logging.getLogger('yandex_captcha_puzzle_solver.yandex_captcha_puzzle_solver').setLevel(logging.DEBUG)
    solver_args['debug_dir'] = args.debug_dir

    sys.argv = [re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])]
    sys.argv += unknown_args

    solver_args['proxy'] = args.proxy

    # Init ProxyController
    solver_args['proxy_controller'] = yandex_captcha_puzzle_solver.proxy_controller.ProxyController(
      start_port=args.proxy_listen_start_port,
      end_port=args.proxy_listen_end_port,
      command=args.proxy_command)

    if args.disable_gpu:
      solver_args['disable_gpu'] = True

    if USE_GUNICORN:
      sys.argv += ['-b', args.bind]
      sys.argv += ['--worker-class', 'uvicorn.workers.UvicornWorker']
      sys.argv += ['yandex_captcha_puzzle_solver:server']
      sys.exit(gunicorn.app.wsgiapp.run())
    else:
      sys.argv += ['--host', host]
      sys.argv += ['--port', port]
      sys.argv += ['yandex_captcha_puzzle_solver:server']
      sys.exit(uvicorn.main.main())

  except Exception as e:
    logging.error(str(e))
    sys.exit(1)


if __name__ == '__main__':
  server_run()
