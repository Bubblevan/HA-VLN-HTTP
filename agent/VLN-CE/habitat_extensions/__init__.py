# 在 InternNav 等使用新版 habitat-lab 的环境中，config.default 需要旧版 habitat 的 Config/get_config。
# 先确保 HA-VLN 自带的 habitat-lab 在 path 前，再导入 config，这样 config.default 能正确加载。
import os
import sys
_dir = os.path.dirname(os.path.abspath(__file__))
_ha_habitat_lab = os.path.abspath(os.path.join(_dir, "..", "..", "..", "habitat-lab"))
if os.path.isdir(_ha_habitat_lab) and _ha_habitat_lab not in sys.path:
    sys.path.insert(0, _ha_habitat_lab)

from habitat_extensions.config.default import get_extended_config
from habitat_extensions import actions, measures, obs_transformers, sensors
from habitat_extensions.task import VLNCEDatasetV1
