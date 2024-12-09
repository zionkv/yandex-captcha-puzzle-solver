import typing
import traceback
import urllib.parse
import mitmproxy


class Addon(object):
  _ground_url: typing.Tuple[str, int]

  def __init__(self, template_root = "mtproxy_templates/"):
    self._ground_url = None

  def load(self, loader):
    loader.add_option(
      name = "ground_url",
      typespec = typing.Optional[str],
      default = None,
      help = "Ground url",
    )

  def configure(self, updates):
    try:
      if "ground_url" in updates:
        ground_url = urllib.parse.urlparse(mitmproxy.ctx.options.ground_url)
        if ground_url.hostname is not None and ground_url.port is not None:
          self._ground_url = (ground_url.hostname, ground_url.port)

    except Exception as e:
      print("configure, exception: " + str(e), flush = True)

  def running(self):
    # We change the connection strategy to lazy so that next_layer happens before we actually connect upstream.
    # Alternatively we could also change the server address in `server_connect`.
    mitmproxy.ctx.options.connection_strategy = "lazy"
    mitmproxy.ctx.options.upstream_cert = False

  def next_layer(self, nextlayer: mitmproxy.proxy.layer.NextLayer):
    """
    remove TLS for ground_url requests
    """
    try:
      print("next_layer: " + str(nextlayer.context.server.address), flush = True)
      if (
        nextlayer.context.server.address is not None and
        self._ground_url is not None
      ):
        nextlayer.context.server.address = self._ground_url
        nextlayer.context.client.alpn = b""
        nextlayer.layer = mitmproxy.proxy.layers.ClientTLSLayer(nextlayer.context)
        nextlayer.layer.child_layer = mitmproxy.proxy.layers.TCPLayer(nextlayer.context)
    except Exception as e:
      print("next_layer, exception: " + str(e), flush = True)
      traceback.print_exc()

  def server_connect(self, data: mitmproxy.proxy.server_hooks.ServerConnectionHookData):
    # non TLS override
    data.server.address = self._ground_url


addons = [
  Addon()
]
