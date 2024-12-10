import os
import random
import typing
import enum
import collections
import math
import numpy as np

import cv2


class ImageProcessor(object):

  """
  Joint representation
  """
  class JointSegment(object):
    class Type(enum.Enum):
      HORIZONTAL = 1
      VERTICAL = 2

    start_point: typing.Tuple[int, int]
    length: int
    type: Type

    def __init__(
      self,
      start_point: typing.Tuple[int, int] = None,
      length: int = None,
      type: Type = None
    ):
      self.start_point = start_point
      self.length = length
      self.type = type

  # Get rect of modal window (that contains challenge).
  @staticmethod
  def get_modal_frame_rect(
    image, color = (255, 255, 255),
    save_steps_dir: str = None,
    logger = None
  ):
    image_height, image_width, _ = image.shape
    color_delta = 20
    mask = cv2.inRange(
      image,
      (
        max(color[0] - color_delta, 0),
        max(color[1] - color_delta, 0),
        max(color[2] - color_delta, 0)
      ),
      (
        min(color[0] + color_delta, 255),
        min(color[1] + color_delta, 255),
        min(color[2] + color_delta, 255)
      )
    )

    if save_steps_dir:
      cv2.imwrite(os.path.join(save_steps_dir, 'mask.png'), mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res_box = None
    for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      if res_box is None or w * h > res_box[2] * res_box[3]:
        res_box = (x, y, w, h)

    if res_box is not None:
      compress_width = 10
      compress_height = 10
      res_box = (
        res_box[0] + int(compress_width / 2),
        res_box[1] + int(compress_height / 2),
        res_box[2] - compress_width,
        res_box[3] - compress_height
      )
      if save_steps_dir:
        debug_image = image.copy()
        debug_image = cv2.rectangle(
          debug_image,
          (res_box[0], res_box[1]),
          (res_box[0] + res_box[2], res_box[1] + res_box[3]),
          (0, 0, 255),
          2
        )
        cv2.imwrite(os.path.join(save_steps_dir, 'rect.png'), debug_image)

    return res_box

  # Get slider points (for drag from to)
  @staticmethod
  def get_drag_points(image, logger = None, save_steps_dir: str = None, log_prefix = ''):
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
      )
    )

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

  # Get puzzle joints for evaluate diff
  @staticmethod
  def get_puzzle_joints(
    image, logger = None, save_steps_dir: str = None, log_prefix = ''
  ) -> typing.List[
    typing.Tuple[  # < Pair of segments, that represent puzzle joints.
      typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]],
      typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]
    ]
  ]:
    # Return array of two elements tuples, where element is point
    # check horizontal net
    puzzle_vertical_separators, puzzle_horizontal_separators = ImageProcessor._determine_separators(
      image, save_steps_dir = save_steps_dir
    )

    if puzzle_vertical_separators is None or puzzle_horizontal_separators is None:
      return []

    # construct joints
    res_joints: typing.List[typing.Tuple[ImageProcessor.JointSegment, ImageProcessor.JointSegment]] = []
    for h_index, h in enumerate(puzzle_horizontal_separators):  # < h is Tuple[Tuple[int, int], Tuple[int, int]]
      for v_index, v in enumerate(puzzle_vertical_separators):
        vertical_indentation = 2
        horizontal_indentation = 2
        if h_index < len(puzzle_horizontal_separators) - 1:
          # add horizontal joint
          j_len = v[1][0] - v[0][1] - 2 * vertical_indentation
          res_joints.append(
            (
              ImageProcessor.JointSegment(
                start_point=(v[0][1] + horizontal_indentation, h[1][0] - vertical_indentation),
                length=j_len,
                type=ImageProcessor.JointSegment.Type.HORIZONTAL
              ),
              ImageProcessor.JointSegment(
                start_point=(v[0][1] + horizontal_indentation, h[1][1] + vertical_indentation),
                length=j_len,
                type=ImageProcessor.JointSegment.Type.HORIZONTAL
              ),
            )
          )
        if v_index < len(puzzle_vertical_separators) - 1:
          # add vertical joint
          j_len = h[1][0] - h[0][1] - 2 * horizontal_indentation
          res_joints.append(
            (
              ImageProcessor.JointSegment(
                start_point=(v[1][0] - horizontal_indentation, h[0][1] + vertical_indentation),
                length=j_len,
                type=ImageProcessor.JointSegment.Type.VERTICAL
              ),
              ImageProcessor.JointSegment(
                start_point=(v[1][1] + horizontal_indentation, h[0][1] + vertical_indentation),
                length=j_len,
                type=ImageProcessor.JointSegment.Type.VERTICAL
              ),
            )
          )

    # draw joints
    if save_steps_dir:
      debug_image = image.copy()
      for joint_from, joint_to in res_joints:
        cv2.line(
          debug_image,
          joint_from.start_point,
          (
            joint_from.start_point[0] + (
              joint_from.length if joint_from.type == ImageProcessor.JointSegment.Type.HORIZONTAL else 0),
            joint_from.start_point[1] + (
              joint_from.length if joint_from.type == ImageProcessor.JointSegment.Type.VERTICAL else 0)
          ),
          (0, 0, 255),
          1
        )
        cv2.line(
          debug_image,
          joint_to.start_point,
          (
            joint_to.start_point[0] + (joint_to.length if joint_to.type == ImageProcessor.JointSegment.Type.HORIZONTAL else 0),
            joint_to.start_point[1] + (joint_to.length if joint_to.type == ImageProcessor.JointSegment.Type.VERTICAL else 0)
          ),
          (0, 0, 255),
          1
        )
      cv2.imwrite(os.path.join(save_steps_dir, 'image_with_joints.png'), debug_image)

    return res_joints

  @staticmethod
  def evaluate_joints_diff(
    image,
    joints: typing.List[typing.Tuple[JointSegment, JointSegment]],
    evaluate_points = 10
  ) -> float:
    color_diff_sum = 0
    color_diff_count = 0
    for joint_from, joint_to in joints:
      for i in range(evaluate_points):
        point1 = (
          (
            joint_from.start_point[0] +
            int(
              joint_from.length * i / evaluate_points
              if joint_from.type == ImageProcessor.JointSegment.Type.HORIZONTAL else 0
            )
          ),
          (
            joint_from.start_point[1] +
            int(
              joint_from.length * i / evaluate_points
              if joint_from.type == ImageProcessor.JointSegment.Type.VERTICAL else 0
            )
          )
        )
        point2 = (
          (
            joint_to.start_point[0] +
            int(
              joint_to.length * i / evaluate_points
              if joint_to.type == ImageProcessor.JointSegment.Type.HORIZONTAL else 0
            )
          ),
          (
            joint_to.start_point[1] +
            int(
              joint_to.length * i / evaluate_points
              if joint_to.type == ImageProcessor.JointSegment.Type.VERTICAL else 0
            )
          )
        )
        c1 = image[point1[1], point1[0]]
        c2 = image[point2[1], point2[0]]
        color_diff_sum += math.sqrt(
          (float(c2[0]) - float(c1[0]))**2 +
          (float(c2[1]) - float(c1[1]))**2 +
          (float(c2[2]) - float(c1[2]))**2
        )
        color_diff_count += 1
    return color_diff_sum / color_diff_count if color_diff_count > 0 else 0

  @staticmethod
  def _lines_to_intervals(lines, sum_threshold) -> typing.List[
    typing.Tuple[
      int,  # pos
      int  # num of lines after pos
    ]
  ]:
    # Return closed intervals (include right line that obey threshold)
    last_interval_start = None
    intervals = []

    for i, line_sum in enumerate(lines):
      if line_sum >= sum_threshold:
        if last_interval_start is None:
          last_interval_start = i
      elif last_interval_start is not None:
        # close interval
        intervals.append([last_interval_start, i - 1])
        last_interval_start = None

    if last_interval_start is not None:
      intervals.append([last_interval_start, len(lines) - 1])

    return intervals

  @staticmethod
  def group_values(arr, radius):
    d = sorted(arr)
    m = [[d[0]]]
    for x in d[1:]:
      if x - m[-1][0] < radius:
        m[-1].append(x)
      else:
        m.append([x])
    return m

  @staticmethod
  def _get_separators_group(
    separators,
    min_len,
    max_len
  ) -> typing.List[
    typing.Tuple[
      typing.Tuple[int, int],  # left/up group of separators
      typing.Tuple[int, int]   # right/down group of separators
    ]
  ]:
    block_heights = collections.OrderedDict()
    for l_index in range(1, len(separators)):
      prev_bottom = separators[l_index - 1][1]
      cur_top = separators[l_index][0]
      block_height = cur_top - prev_bottom
      if block_height not in block_heights:
        block_heights[block_height] = []
      block_heights[block_height].append((separators[l_index - 1], separators[l_index]))

    if block_heights.keys():
      key_groups = ImageProcessor.group_values(block_heights.keys(), 4)
      for height_group in key_groups:
        avg_height = int(sum(height_group) / len(height_group))
        if (avg_height >= int(min_len) and avg_height < int(max_len)):
          # check number of separators
          res_separators = []
          for h in height_group:
            res_separators += block_heights[h]
          res_separators = sorted(res_separators, key=lambda sep: sep[0])
          if (len(res_separators) >= 3 and len(res_separators) <= 7):
            # found puzzle groups
            return res_separators

    return None

  @staticmethod
  def _determine_separators(
    image, white_percent = 0.94,
    lower_color = (210, 210, 210), upper_color = (256, 256, 256),
    save_steps_dir = None
  ):
    h, w, _ = image.shape
    mask = cv2.inRange(image, lower_color, upper_color)
    mask = mask / 255

    # determine horizontal separators
    horizontal_lines = np.sum(mask, axis = 1)
    horizontal_separators = ImageProcessor._lines_to_intervals(horizontal_lines, w * white_percent)
    # filter horizontal separators
    puzzle_horizontal_separators = ImageProcessor._get_separators_group(
      horizontal_separators, h / 20, h / 2
    )

    # determine vertical separators
    vertical_lines = np.sum(mask, axis = 0)
    vertical_separators = ImageProcessor._lines_to_intervals(vertical_lines, h * white_percent)
    # filter vertical separators
    puzzle_vertical_separators = ImageProcessor._get_separators_group(
      vertical_separators, w / 20, w / 2
    )

    if puzzle_horizontal_separators is None or puzzle_vertical_separators is None:
      return None, None

    if save_steps_dir:
      debug_image = image.copy()
      for h_el in puzzle_horizontal_separators:  # < Tuple[Tuple[int, int], Tuple[int, int]]
        cv2.line(debug_image, (0, h_el[0][1]), (w, h_el[0][1]), (255, 0, 0), 1)
        cv2.line(debug_image, (0, h_el[1][0]), (w, h_el[1][0]), (255, 0, 0), 1)
      for v_el in puzzle_vertical_separators:  # < Tuple[Tuple[int, int], Tuple[int, int]]
        cv2.line(debug_image, (v_el[0][1], 0), (v_el[0][1], h), (0, 0, 255), 1)
        cv2.line(debug_image, (v_el[1][0], 0), (v_el[1][0], h), (0, 0, 255), 1)
      cv2.imwrite(os.path.join(save_steps_dir, 'image_with_sep.png'), debug_image)

    return puzzle_vertical_separators, puzzle_horizontal_separators
