import os
import sys
import typing
import traceback
import asyncio
import uuid
import http.cookiejar
import shutil
import logging
import numpy as np

import cv2

import zendriver_flare_bypasser as zendriver

XVFB_DISPLAY = None

logger = logging.getLogger(__name__)


class Rect(object):
  left: int
  top: int
  width: int
  height: int


"""
Trivial wrapper for browser (driver).
Allow to localize driver operations implementation and requirements,
and simplify migration to other driver.
"""


class BrowserWrapper(object):
  _zendriver_driver: zendriver.Browser = None
  _page = None

  class FakePosition(object):
    center = None

    def __init__(self, center):
      self.center = tuple(float(x) for x in center)

  class FakeNode(object):
    attributes = None

  class FakeElement(zendriver.Element):
    _position = None

    def __init__(self, page: zendriver.Tab, center_coords):
      super(BrowserWrapper.FakeElement, self).__init__(
        BrowserWrapper.FakeNode(),  # zendriver.cdp.dom.Node
        page  # zendriver.Tab
      )
      self._position = BrowserWrapper.FakePosition(center_coords)

    def _make_attrs(self):  # override for exclude exception on __init__
      pass

    # overrides for call only cdp click send in zendriver.Element.mouse_click
    async def get_position(self):
      return self._position

    async def flash(self, duration: typing.Union[float, int] = 0.5):
      pass

  def __init__(self, zendriver_driver: zendriver.Browser, user_data_dir: str = None):
    self._zendriver_driver = zendriver_driver
    self._user_data_dir = user_data_dir

  def __del__(self):
    if self._user_data_dir:
      shutil.rmtree(self._user_data_dir, ignore_errors=True)

  @staticmethod
  def start_xvfb_display():
    if sys.platform != 'win32':
      global XVFB_DISPLAY
      if XVFB_DISPLAY is None:
        from xvfbwrapper import Xvfb
        XVFB_DISPLAY = Xvfb()
        XVFB_DISPLAY.start()

  @staticmethod
  async def create(proxy = None, disable_gpu = False):
    user_data_dir = os.path.join("/tmp", str(uuid.uuid4()))  # < Each created chrome should be isolated.
    BrowserWrapper.start_xvfb_display()
    browser_args = []
    if proxy:
      browser_args.append("--proxy-server=" + proxy)
    if disable_gpu:
      browser_args += [
        "--disable-gpu",
        "--disable-software-rasterizer"
      ]
    if sys.platform == 'win32':
      browser_args += ["--headless"]

    browser_args += ["--user-data-dir=" + user_data_dir]
    browser_args += ["--ignore-certificate-errors", "--ignore-urlfetcher-cert-requests"]

    try:
      zendriver_driver = await zendriver.start(
        sandbox=False,
        browser_args=browser_args
      )
      return BrowserWrapper(zendriver_driver, user_data_dir = user_data_dir)
    finally:
      shutil.rmtree(user_data_dir, ignore_errors=True)

  # Get original driver page impl - can be used only in user command specific implementations
  def get_driver(self):
    return self._page

  async def size(self):
    image = await self.get_screenshot()
    image_height, image_width, _ = image.shape
    return image_width, image_height

  async def get_outputs(self):
    try:
      stdout_bytes, stderr_bytes = await self._zendriver_driver.communicate()
      return [stdout_bytes, stderr_bytes]
    except Exception:
      return None

  async def current_url(self):
    return self._page.url

  async def close(self):
    self._page = None
    if self._zendriver_driver:
      await self._zendriver_driver.stop()
    if self._user_data_dir:
      shutil.rmtree(self._user_data_dir, ignore_errors=True)
      self._user_data_dir = None

  async def select_text(self, css_selector):
    try:
      res = await self._page.select(css_selector, timeout=0)
      return res.text
    except asyncio.TimeoutError:
      return None

  async def select_count(self, css_selector):
    try:
      return len(await self._page.select_all(css_selector, timeout=0))  # Select without waiting.
    except asyncio.TimeoutError:
      return 0

  async def get(self, url):
    # we work only with one page - close all tabs (excluding first - this close browser)
    for tab_i, tab in enumerate(self._zendriver_driver.tabs):
      if tab_i > 0:
        await tab.close()
    self._page = await self._zendriver_driver.get(url)

  async def click(self, css_selector):
    try:
      element = await self._page.select(css_selector, timeout=0)
    except asyncio.TimeoutError:
      return False
    await element.click()
    return True

  async def click_coords(self, coords):
    # Specific workaround for zendriver
    # click by coordinates without no driver patching.
    step = "start"
    try:
      fake_node = BrowserWrapper.FakeElement(self._page, coords)
      step = "mouse_click"
      await fake_node.mouse_click()
    except Exception as e:
      print("EXCEPTION on click_coords '" + step + "': " + str(e))
      raise

  async def mouse_down(self):
    try:
      await self._page.mouse.down()
    except Exception as e:
      print("EXCEPTION on mouse_down: " + str(e) + ":\n" + traceback.format_exc())
      raise

  async def mouse_up(self):
    try:
      await self._page.mouse.up()
    except Exception as e:
      print("EXCEPTION on mouse_up: " + str(e))
      raise

  async def mouse_move(self, coords):
    try:
      await self._page.mouse.move(coords[0], coords[1])
    except Exception as e:
      print("EXCEPTION on mouse_move: " + str(e))
      raise

  async def get_user_agent(self):
    return await self._page.evaluate("window.navigator.userAgent")

  async def get_dom(self):
    res_dom = await self._page.get_content()
    return (res_dom if res_dom is not None else "")  # zendriver return None sometimes (on error)

  async def get_element_screenshot(self, css_selector) -> tuple[np.array, Rect]:
    # < Return screenshot as cv2 image (numpy array)
    tmp_file_path = None

    try:
      try:
        element = await self._page.select(css_selector, timeout=0)
      except asyncio.TimeoutError:
        return (None, None)

      if element is None:
        return (None, None)

      try:
        logger.info("To get position for '" + css_selector + "'")
        pos = await element.get_position()  # abs=True don't works
      finally:
        logger.info("From get position for '" + css_selector + "'")
      rect = Rect()
      rect.left = pos.abs_x
      rect.top = pos.abs_y
      rect.width = pos.width
      rect.height = pos.height

      try:
        if tmp_file_path is None:
          tmp_file_path = os.path.join("/tmp", str(uuid.uuid4()) + ".jpg")
        await element.save_screenshot(tmp_file_path)
        return cv2.imread(tmp_file_path), rect
      except Exception as e:
        # return None for:
        #   "not finished loading yet"
        #   "Cannot take screenshot with 0 height." - elements isn't loaded yet.
        #   "Could not find object with given id" - element dissappeared, js on page changed DOM.
        #
        msg = str(e).lower()
        if (
          "not finished loading yet" not in msg and
          "cannot take screenshot " not in msg and
          "could not find object" not in msg and
          "could not find position" not in msg
        ):
          raise
      return (None, None)
    finally:
      if tmp_file_path is not None and os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)

  async def get_screenshot(self):  # Return screenshot as cv2 image (numpy array)
    tmp_file_path = None
    try:
      while True:
        try:
          tmp_file_path = os.path.join("/tmp", str(uuid.uuid4()) + ".jpg")
          await self._page.save_screenshot(tmp_file_path)
          return cv2.imread(tmp_file_path)
        except zendriver.core.connection.ProtocolException as e:
          if "not finished loading yet" not in str(e):
            raise
        await asyncio.sleep(1)
    finally:
      if tmp_file_path is not None and os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)

  async def save_screenshot(self, image_path):
    while True:
      try:
        await self._page.save_screenshot(image_path)
        return
      except zendriver.core.connection.ProtocolException as e:
        if "not finished loading yet" not in str(e):
          raise
      await asyncio.sleep(1)

  async def set_cookies(self, cookies: list[dict]):
    # convert {"name": "...", "value": "...", ...} to array of http.cookiejar.Cookie
    cookie_jar = http.cookiejar.CookieJar()
    for c in cookies:
      # TO CHECK, that all fields filled correctly.
      cookie_jar.set_cookie(http.cookiejar.Cookie(
        None,  # version
        c.get('name', None),
        c.get('value', None),
        c.get('port', 443),
        None,  # port_specified
        c.get('domain', None),
        None,  # domain_specified
        None,  # domain_initial_dot
        c.get('path', '/'),
        None,  # path_specified
        c.get('secure', False),
        c.get('expires', None),  # < here expected float seconds since epoch time.
        None,  # discard
        None,  # comment
        None,  # comment_url
        None   # rest
      ))
    await self._zendriver_driver.cookies.set_all(cookie_jar)

  async def get_cookies(self) -> list[dict]:
    # return list of dict have format: {"name": "...", "value": "..."}
    zendriver_cookies = await self._zendriver_driver.cookies.get_all(requests_cookie_format=True)
    res = []
    # convert array of http.cookiejar.Cookie to expected cookie format
    for cookie in zendriver_cookies:
      res.append({
        "name": cookie.name,
        "value": cookie.value,
        "port": cookie.port,
        "domain": cookie.domain,
        "path": cookie.path,
        "secure": cookie.secure
      })
    return res
