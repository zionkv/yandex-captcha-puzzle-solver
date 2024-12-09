import sys
import logging
import os
import typing
import random
import datetime
import asyncio
import certifi
import contextlib

# Image processing imports
import numpy as np
import cv2

from .browser_wrapper import BrowserWrapper
from .proxy_controller import ProxyController
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)

YANDEX_CAPTCHA_SELECTORS = [
  'iframe[src*="smartcaptcha.yandexcloud.net"]',
]
USER_AGENT = None

_SHORT_TIMEOUT = 1
_REDIRECT_WAIT_TIMEOUT = 5


"""
Request for process, can be extended and some custom fields used in process_command.
"""


class Request(object):
  url: str = None
  yandex_key: str = None
  proxy: dict = None
  max_timeout: float = 60  # timeout in sec
  cookies: dict = None

  def __init__(self, _dict=None):
    if _dict:
      self.__dict__.update(_dict)

  def __str__(self):
    return str(self.__dict__)


class Response(object):
  url: str = None
  cookies: list = None
  user_agent: str = None
  token: str = None
  message: str = None

  def __str__(self):
    return str(self.__dict__)


class Solver(object):
  """
  Solver
  """
  _proxy: str = None
  _driver: BrowserWrapper = None
  _proxy_controller: ProxyController = None
  _disable_gpu: bool = False
  _screenshot_i: int = 0
  _debug_dir: str = None

  class Exception(Exception):
    step = None

    def __init__(self, message: str, step: str = None):
      super().__init__(message)
      self.step = step

  def __init__(
    self,
    proxy: str = None,
    proxy_controller = None,
    disable_gpu = False,
    debug_dir: str = None
  ):
    self._proxy = proxy
    self._driver = None
    self._proxy_controller = proxy_controller
    self._debug_dir = debug_dir
    self._disable_gpu = disable_gpu

  async def save_screenshot(
    self, step_name, image = None, mark_coords = None, mark_rect = None, mark_joints = None
  ):
    if self._debug_dir:
      screenshot_file_without_ext = os.path.join(
        self._debug_dir, str(self._screenshot_i) + '_' + step_name)

      if image is not None:
        cv2.imwrite(screenshot_file_without_ext + ".jpg", image)
      else:
        await self._driver.save_screenshot(screenshot_file_without_ext + ".jpg")

      if mark_coords or mark_rect or mark_joints:
        image = cv2.imread(screenshot_file_without_ext + ".jpg")
        if mark_coords:
          for mark_c in mark_coords:
            image = cv2.circle(image, mark_c, 5, (255, 0, 0), 2)
        if mark_rect:
          image = cv2.rectangle(
            image,
            (mark_rect[0], mark_rect[1]),
            (mark_rect[2], mark_rect[3]),
            (255, 0, 0),
            2  # < thickness
          )
        if mark_joints:
          for joint_from, joint_to in mark_joints:
            image = cv2.line(
              image,
              joint_from.start_point,
              (
                joint_from.start_point[0] + (
                  joint_from.length if joint_from.type == ImageProcessor.JointSegment.Type.HORIZONTAL else 0
                ),
                joint_from.start_point[1] + (
                  joint_from.length if joint_from.type == ImageProcessor.JointSegment.Type.VERTICAL else 0
                )
              ),
              (0, 0, 255),
              1
            )
            image = cv2.line(
              image,
              joint_to.start_point,
              (
                joint_to.start_point[0] + (
                  joint_to.length if joint_to.type == ImageProcessor.JointSegment.Type.HORIZONTAL else 0
                ),
                joint_to.start_point[1] + (
                  joint_to.length if joint_to.type == ImageProcessor.JointSegment.Type.VERTICAL else 0
                )
              ),
              (0, 0, 255),
              1
            )
        cv2.imwrite(screenshot_file_without_ext + "_mark.jpg", image)

      dom = await self._driver.get_dom()
      with open(screenshot_file_without_ext + '.html', 'w') as fp:
        fp.write(dom)
      self._screenshot_i += 1

      logger.debug("Screenshot saved to '" + screenshot_file_without_ext + "'")

  async def solve(self, req: Request) -> Response:
    # do some validations
    if req.url is None:
      raise Exception("Parameter 'url' should be defined.")

    try:
      logger.info("Solve request: " + str(req))
      res = await asyncio.wait_for(self._resolve_challenge(req), req.max_timeout)
      logger.info("Solve result: " + str(res))
    except asyncio.TimeoutError:
      raise Exception("Processing timeout (max_timeout=" + str(req.max_timeout) + ")")
    return res

  async def _resolve_challenge(self, req: Request) -> Response:
    start_time: datetime.datetime = datetime.datetime.now()
    step = 'start'
    try:
      # Use default upped proxy
      use_proxy: str = self._proxy
      proxy_holder = None

      step = 'proxy init'
      if req.proxy:
        # Up proxy with specific end proxy (for yandex requests)
        if not self._proxy_controller:
          raise Solver.Exception("For use proxy with authorization you should pass proxy_controller into c-tor")
        proxy_holder = self._proxy_controller.get_proxy(use_proxy)
        use_proxy = "socks5://127.0.0.1:" + str(proxy_holder.local_port())
      else:
        proxy_holder = contextlib.nullcontext()

      with proxy_holder:
        try:
          step = 'browser init'
          self._driver: BrowserWrapper = await BrowserWrapper.create(
            use_proxy, disable_gpu = self._disable_gpu
          )
          logger.info(
            'New instance of webdriver has been created to perform the request (proxy=' +
            str(use_proxy) + '), timeout=' + str(req.max_timeout))
          return await self._resolve_challenge_impl(req, start_time)
        finally:
          logger.info('Close webdriver')
          if self._driver is not None:
            await self._driver.close()
            logger.debug('A used instance of webdriver has been destroyed')
          if logger.isEnabledFor(logging.DEBUG) and self._driver is not None:
            # Read outputs only after driver close (when process stopped),
            # otherwise output reading can be blocked.
            outputs = await self._driver.get_outputs()
            if outputs:
              for output_i, output in enumerate(outputs):
                logger.debug(
                  "Webdriver output #" + str(output_i) + ":" +
                  "\n---------------------------------------\n" +
                  str(output.decode("utf-8")) +
                  "\n---------------------------------------\n"
                )
          self._driver = None
    except Solver.Exception as e:
      error_message = (
        "Error solving the challenge. On platform " + str(sys.platform) +
        " at step '" + str(e.step) + "': " +
        str(e).replace('\n', '\\n')
      )
      logger.error(error_message)
      raise Solver.Exception(error_message, step=e.step)
    except Exception as e:
      error_message = (
        "Error solving the challenge. On platform " + str(sys.platform) +
        " at step '" + step + "': " +
        str(e).replace('\n', '\\n')
      )
      logger.error(error_message)
      raise Solver.Exception(error_message)

  async def _check_challenge(self):
    for selector in YANDEX_CAPTCHA_SELECTORS:
      if await self._driver.select_count(selector) > 0:
        return True
    return False

  async def _wait_screenshot(self, css_selector) -> typing.Tuple[typing.Any, typing.Tuple[int, int, int, int]]:
    while True:
      captcha_frame_image, rect = await self._driver.get_element_screenshot(
        'div[class="smart-captcha"]'
      )
      if captcha_frame_image is not None:
        return (captcha_frame_image, rect)
      await self.save_screenshot('element_screenshot_step')
      await asyncio.sleep(1)

  async def _challenge_wait_and_click_loop(self) -> str:
    attempt = 0
    width, height = await self._driver.size()

    while True:
      logger.info("Challenge step #" + str(attempt))
      await self.save_screenshot('attempt')

      # Check state of captcha.
      checkbox = await self._driver.select_count('iframe[src*="smartcaptcha.yandexcloud.net/checkbox"]')

      if checkbox > 0:
        # Checkbox state (slider), can be in two states: need to solve, solved.
        captcha_frame_image, captcha_frame_rect = await self._wait_screenshot('div[class="smart-captcha"]')
        image_height, image_width, _ = captcha_frame_image.shape

        logger.info("Drag slider")
        await self.save_screenshot('attempt_to_move_slider')
        # scroller in form
        down_and_up_points = Solver._get_drag_points(captcha_frame_image)
        if down_and_up_points is not None:
          down_point = down_and_up_points[0]
          if down_point[0] < image_width / 2:
            # Need to solve checkbox
            up_point = down_and_up_points[1]
            logger.info("To move slider from " + str(down_point) + " to " + str(up_point))
            await self.save_screenshot('attempt_move_down_point', mark_coords=[down_point, up_point])
            abs_down_point = (captcha_frame_rect.left + down_point[0], captcha_frame_rect.top + down_point[1])
            abs_up_point = (captcha_frame_rect.left + up_point[0], captcha_frame_rect.top + up_point[1])
            premove_point = (down_point[0] - random.randint(5, 10), down_point[1] - random.randint(5, 10))
            await self._driver.mouse_move(premove_point)
            await self._driver.mouse_move(down_point)
            await self._driver.mouse_down()
            await self.save_screenshot('attempt_move_down', mark_coords=[down_point])
            steps = 20
            for i in range(1, steps + 1):
              x = round(abs_down_point[0] + (abs_up_point[0] - abs_down_point[0]) * (i / steps))
              y = round(abs_down_point[1] + (abs_up_point[1] - abs_down_point[1]) * (i / steps))
              await self._driver.mouse_move((x, y))
            await self._driver.mouse_up()
            await self.save_screenshot('attempt_from_move_slider_after_up')
            await asyncio.sleep(1)  # Wait advanced challenge loading.
          else:
            # Solved (slider in right part) - send form
            await self._driver.click("#submit_captcha_button")
            await asyncio.sleep(1)  # Wait form sending.
            # get token as text
            res = await self._driver.select_text('#smart_token')
            return res

      advanced = await self._driver.select_count('iframe[src*="smartcaptcha.yandexcloud.net/advanced"]')
      if advanced > 0:
        # modal mode - find puzzle position
        full_image = await self._driver.get_screenshot()
        # get internal slider position in modal
        logger.info("Solve modal window")
        await self.save_screenshot("modal_solve_start")
        modal_rect = ImageProcessor.get_modal_frame_rect(full_image)
        modal_image = full_image[
          modal_rect[1]:modal_rect[1] + modal_rect[3], modal_rect[0]:modal_rect[0] + modal_rect[2]
        ]
        find_slider_rect = (
          0,
          int(modal_rect[3] * 3 / 5),
          modal_rect[2],
          int(modal_rect[3] * 2 / 5),
        )
        slider_image = modal_image[
          find_slider_rect[1]:find_slider_rect[1] + find_slider_rect[3],
          find_slider_rect[0]:find_slider_rect[0] + find_slider_rect[2]
        ]
        down_and_up_points = ImageProcessor.get_drag_points(slider_image)

        if down_and_up_points is not None:
          assert down_and_up_points[0][0] < down_and_up_points[1][0]
          # offset down_and_up_points relative full image
          down_and_up_points = (
            (
              down_and_up_points[0][0] + modal_rect[0] + find_slider_rect[0],
              down_and_up_points[0][1] + modal_rect[1] + find_slider_rect[1]
            ),
            (
              down_and_up_points[1][0] + modal_rect[0] + find_slider_rect[0],
              down_and_up_points[1][1] + modal_rect[1] + find_slider_rect[1]
            )
          )
          await self.save_screenshot("modal_slider_pos", mark_coords=down_and_up_points)

          # get joints in modal
          find_puzzle_rect = (
            0,
            0,
            modal_rect[2],
            int(modal_rect[3] * 3 / 5),
          )
          find_puzzle_image = modal_image[
            find_puzzle_rect[1]:find_puzzle_rect[3],
            find_puzzle_rect[0]:find_puzzle_rect[2]
          ]
          logger.debug("To save modal_find_puzzle")
          await self.save_screenshot("modal_find_puzzle", image=find_puzzle_image)
          logger.debug("From save modal_find_puzzle")
          joints = ImageProcessor.get_puzzle_joints(find_puzzle_image)
          # offset joints relative full image
          for joint1, joint2 in joints:
            joint1.start_point = (
              joint1.start_point[0] + modal_rect[0] + find_puzzle_rect[0],
              joint1.start_point[1] + modal_rect[1] + find_puzzle_rect[1]
            )
            joint2.start_point = (
              joint2.start_point[0] + modal_rect[0] + find_puzzle_rect[0],
              joint2.start_point[1] + modal_rect[1] + find_puzzle_rect[1]
            )

          logger.debug("Modal solving started with " + str(len(joints)) + " joints")
          await self.save_screenshot("modal_start", mark_joints=joints)
          start_point = down_and_up_points[0]
          await self._driver.mouse_move(start_point)
          await self._driver.mouse_down()
          slider_steps = 20
          min_diff = 0
          max_diff = 10000000
          end_point = down_and_up_points[1]
          depth_step_i = 0
          select_pos = None
          while abs(max_diff - min_diff) > 0.1 and abs(end_point[0] - start_point[0]) > 2:
            min_diff = 10000000
            max_diff = 0
            new_start_point = None  # < Positions around min diff point
            new_end_point = None

            check_points: typing.List[typing.Tuple[int, int]] = []
            if end_point[0] - start_point[0] > slider_steps:
              for i in range(slider_steps + 1):
                check_points.append(Solver._middle_point(start_point, end_point, float(i) / slider_steps))
            else:
              for i in range(start_point[0], end_point[0] + 1):
                check_points.append((i, start_point[1]))

            for check_point_i, middle_point in enumerate(check_points):
              await self._driver.mouse_move(middle_point)
              full_image = await self._driver.get_screenshot()
              joints_diff = ImageProcessor.evaluate_joints_diff(full_image, joints)
              await self.save_screenshot(
                "modal_slider_pos_" + str(i) + "_" + str(joints_diff), mark_joints=joints
              )
              if new_start_point is None or joints_diff < min_diff:
                min_diff = joints_diff
                new_start_point = check_points[check_point_i - 1] if check_point_i > 0 else middle_point
                new_end_point = check_points[check_point_i + 1] if check_point_i < len(check_points) - 1 else middle_point
                select_pos = middle_point
              max_diff = max(max_diff, joints_diff)

            start_point = new_start_point
            end_point = new_end_point
            logger.debug(
              "Puzzle solve step " + str(depth_step_i) + ", min_diff = " + str(min_diff) +
              ", max_diff = " + str(max_diff) +
              ", start_point = " + str(start_point) +
              ", end_point = " + str(end_point) +
              ", select_pos = " + str(select_pos) +
              ", check_points = " + str(check_points)
            )
            depth_step_i += 1

          await self._driver.mouse_move(select_pos)
          await self.save_screenshot("modal_final_slider_pos")
          await self._driver.mouse_up()
          await asyncio.sleep(1)
          await self.save_screenshot('modal_final')

      attempt = attempt + 1
      await asyncio.sleep(_SHORT_TIMEOUT)

  async def _resolve_challenge_impl(self, req: Request, start_time: datetime.datetime) -> Response:
    step = 'solving'
    try:
      res = Response()

      step = 'navigate to url'
      # navigate to the page
      result_url = req.url
      if "?" not in result_url:
        result_url += "?"
      if result_url:
        if result_url[-1] != '?':
          result_url += '&'
        result_url += 'solver_intercept=1&yandex_captcha_key=' + str(req.yandex_key)
      logger.debug(f'Navigating to... {result_url}')
      await self._driver.get(result_url)

      logger.debug('To make screenshot')
      await self.save_screenshot('evil_logic')

      step = 'set cookies'

      # set cookies if required
      if req.cookies:
        logger.debug('Setting cookies...')
        await self._driver.set_cookies(req.cookies)
        await self._driver.get(result_url)

      step = 'solve challenge'
      token = await self._challenge_wait_and_click_loop()
      res.message = "Challenge solved!"  # expect exception if challenge isn't solved
      res.token = token

      logger.info("Challenge solving finished")
      await self.save_screenshot('solving_finish')

      # Click submit and get token text

      step = 'get cookies'
      res.url = await self._driver.current_url()
      res.cookies = await self._driver.get_cookies()
      logger.info("Cookies got")
      global USER_AGENT
      if USER_AGENT is None:
        step = 'get user-agent'
        USER_AGENT = await self._driver.get_user_agent()
      res.user_agent = USER_AGENT

      await self.save_screenshot('finish')
      logger.info('Solving finished')

      return res
    except Exception as e:
      raise Solver.Exception(str(e), step=step)

  @staticmethod
  def _middle_point(start_point: typing.Tuple[int, int], end_point: typing.Tuple[int, int], coef: float):
    return (
      int(round(start_point[0] + float(end_point[0] - start_point[0]) * coef)),
      int(round(start_point[1] + float(end_point[1] - start_point[1]) * coef))
    )

  @staticmethod
  def _get_dominant_color(image):
    a2D = image.reshape(-1, image.shape[-1])
    col_range = (256, 256, 256)  # generically: a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

  @staticmethod
  def _get_drag_points(image, logger = None, save_steps_dir: str = None, log_prefix = ''):
    image_height, image_width, _ = image.shape
    slider_color = (255, 130, 82)  # < GBR color of slider.
    slider_color_delta = 50
    mask = cv2.inRange(
      image,
      (
        max(slider_color[0] - slider_color_delta, 0),
        max(slider_color[1] - slider_color_delta, 0),
        max(slider_color[2] - slider_color_delta, 0)
      ),
      (
        min(slider_color[0] + slider_color_delta, 255),
        min(slider_color[1] + slider_color_delta, 255),
        min(slider_color[2] + slider_color_delta, 255)
      ))

    if save_steps_dir:
      cv2.imwrite(os.path.join(save_steps_dir, 'mask.png'), mask)

    broad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.dilate(mask, broad_kernel, iterations = 1)

    if save_steps_dir:
      cv2.imwrite(os.path.join(save_steps_dir, 'dilated_mask.png'), mask)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.erode(mask, erode_kernel, iterations = 1)

    if save_steps_dir:
      cv2.imwrite(os.path.join(save_steps_dir, 'eroded_mask.png'), mask)

    points_x, points_y = np.where(mask >= 255)
    if len(points_x) == 0:
      return None

    down_point_pos = random.randint(0, len(points_x) - 1)
    down_point = (points_y[down_point_pos], points_x[down_point_pos])
    up_point_pos = random.randint(0, len(points_x) - 1)
    up_point = (image_width - points_y[up_point_pos], points_x[up_point_pos])

    if save_steps_dir:
      debug_image = image.copy()
      debug_image = cv2.circle(debug_image, down_point, 5, (0, 0, 255), 2)
      debug_image = cv2.circle(debug_image, up_point, 5, (0, 0, 255), 2)
      cv2.imwrite(os.path.join(save_steps_dir, 'image_with_points.png'), debug_image)

    return [down_point, up_point]


# fix ssl certificates for compiled binaries
# https://github.com/pyinstaller/pyinstaller/issues/7229
# https://stackoverflow.com/questions/55736855/how-to-change-the-cafile-argument-in-the-ssl-module-in-python3
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

if __name__ == '__main__':
  sys.stdout.reconfigure(encoding="utf-8")
  logger.basicConfig(
    format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
    handlers=[logger.StreamHandler(sys.stdout)],
    level=logging.INFO)

  req = Request()
  req.url = 'https://knopka.ashoo.id'

  solver = Solver()
  res = solver.solve(req)
