import os
import argparse
import jinja2
import flask

app = flask.Flask(__name__, template_folder = "templates/")

form_page_template = None
page_template = None
template_root = ''


@app.route('/shutdown', methods=["GET", "POST"])
def request_shutdown():
  try:
    print("Flask shutdown request got ...", flush = True)
    shutdown_fun = flask.request.environ.get('werkzeug.server.shutdown')
    shutdown_fun()
    print("Flask shutdown request processed ...", flush = True)
    return flask.Response(status = 204)
  except Exception:
    return flask.Response(status = 500)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def request_main(path):
  # init template
  # parse utm_keyword
  yandex_captcha_key = flask.request.args.get("yandex_captcha_key")
  args = {}
  args['yandex_captcha_key'] = yandex_captcha_key
  global page_template
  resp = page_template.render(args)
  return flask.Response(resp, mimetype = 'text/html')


@app.route('/send_captcha')
@app.route('/send_captcha/')
def request_send_captcha():
  # init template
  # parse utm_keyword
  smart_token = flask.request.args.get("smart-token")
  args = {}
  args['smart_token'] = smart_token
  global form_page_template
  resp = form_page_template.render(args)
  return flask.Response(resp, mimetype = 'text/html')


def run_app(args):
  app.run(host = "0.0.0.0", port = args['port'], threaded = True)


def start_app():
  parser = argparse.ArgumentParser(description = 'grounding_server.')
  parser.add_argument(
    "-p", "--port", type = int, default = 9200, help="Listen port")
  parser.add_argument(
    "-f", "--pidfile", "--pid-file", type = str, default = 'grounding_server.pid', help="Pid file")
  parser.add_argument(
    "-t", "--page-template", type = str, default = 'index.html.j2', help = "Template file")
  parser.add_argument(
    "--form-page-template", type = str, default = 'form.html.j2', help = "Template file")
  args = parser.parse_args()

  pid = os.getpid()
  with open(args.pidfile, 'wb') as f:
    f.write(str(pid).encode('utf-8'))
    f.close()

  global page_template
  page_template = jinja2.Environment(loader = jinja2.FileSystemLoader("/")).get_template(args.page_template)

  global form_page_template
  form_page_template = jinja2.Environment(loader = jinja2.FileSystemLoader("/")).get_template(args.form_page_template)

  run_app({'port': args.port, 'ssl': False})


if __name__ == "__main__":
  start_app()
