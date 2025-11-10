from functools import partial

import core.color
from .navigator import *


class Interfaces:
    baas_thread: core.Baas_thread

    I_main_page: Navigator.Interface
    I_group_page: Navigator.Interface

    def _validator_rgb_feature(self, rgb_feature: str):
        return partial(core.color.match_rgb_feature, self.baas_thread, rgb_feature)

    def _action_click(self, x: int, y: int):
        return partial(self.baas_thread.click, x, y)

    def _gen_interfaces(self):
        self.I_main_page = Navigator.Interface(
            name="main_page",
            description="The main menu interface",
            features=[
                self._validator_rgb_feature("main_page")
            ],
            actions={
                "group_page": self._action_click(565, 648)
            }
        )
        self.I_group_page = Navigator.Interface(
            name="group_page",
            description="The group page interface",
            features=[
                self._validator_rgb_feature("group_menu")
            ],
            actions={
                "main_page": self._action_click(920, 159)
            }
        )

    def list_interfaces(self):
        interfaces = []
        for name, val in vars(self).items():
            if isinstance(val, Navigator.Interface):
                interfaces.append(val)
        return interfaces

    def __init__(self, baas_thread: core.Baas_thread):
        self.baas_thread = baas_thread
        self._gen_interfaces()
