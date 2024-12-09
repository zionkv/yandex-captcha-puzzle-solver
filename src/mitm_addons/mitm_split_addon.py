import typing
import mitmproxy
from mitmproxy.script import concurrent


class Addon(object):
  _no_condition_via = None  # proxy for send external traffic (url does not subject the condition)
  _proxy_via = None  # proxy for send internal traffic (url is subject to the condition)

  def __init__(self):
    pass

  def load(self, loader):
    loader.add_option(
      name="proxy",
      typespec=typing.Optional[str],
      default=None,
      help="proxy",
    )

  def configure(self, updates):
    try:
      if "proxy" in updates:
        self._set_proxy(mitmproxy.ctx.options.proxy)
    except Exception as e:
      print("configure, exception: " + str(e), flush=True)

  def running(self):
    # We change the connection strategy to lazy so that next_layer happens before we actually connect upstream.
    # Alternatively we could also change the server address in `server_connect`.
    mitmproxy.ctx.options.connection_strategy = "lazy"
    mitmproxy.ctx.options.upstream_cert = False
    # fill default upstream (for url's subject to the condition)
    self._proxy_via = None  # set via to None for non upstream modes
    options = mitmproxy.ctx.options
    if options.mode and options.mode[0].startswith("upstream:"):
      mode = mitmproxy.proxy.mode_specs.UpstreamMode.parse(options.mode[0])
      self._proxy_via = (mode.scheme, mode.address)

  @concurrent
  def requestheaders(self, flow):
    # print("REQUEST URL: " + flow.request.pretty_url, flush=True)
    need_send_to_proxy = self._need_send_to_proxy(flow.request)

    # flow.server_conn.via is None: means that will be used upstream(internal proxy)
    if (need_send_to_proxy and (
      flow.server_conn.via is None or flow.server_conn.via != self._no_condition_via)
    ):
      # switch to use external proxy
      flow.server_conn.state = mitmproxy.connection.ConnectionState.CLOSED
      flow.server_conn.via = self._no_condition_via
    elif (not need_send_to_proxy and (
      flow.server_conn.via is not None and flow.server_conn.via == self._no_condition_via)
    ):
      # switch from use proxy to upstream
      flow.server_conn.state = mitmproxy.connection.ConnectionState.CLOSED
      flow.server_conn.via = self._proxy_via

    print("SEND URL: " + flow.request.pretty_url + " => " + str(flow.server_conn.via), flush=True)

  def _need_send_to_proxy(self, request):
    args = request.query  # args: MultiDictView
    return ("solver_intercept" not in args)

  def _set_proxy(self, parse_proxy):
    proxy_spec = mitmproxy.net.server_spec.parse(parse_proxy, "http") if parse_proxy else None
    self._no_condition_via = proxy_spec
    self._via = proxy_spec


addons = [
  Addon()
]
