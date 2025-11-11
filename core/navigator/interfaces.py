from core.color import match_rgb_feature
from core.picture import match_img_feature
from .navigator import *


class Interfaces:
    @staticmethod
    def _validator_rgb_feature(rgb_feature: str):
        return lambda baas_thread: match_rgb_feature(baas_thread, rgb_feature)

    @staticmethod
    def _validator_image_feature(image_feature: str):
        return lambda baas_thread: match_img_feature(baas_thread, image_feature, 0.8, 20)

    @staticmethod
    def _action_click(x: int, y: int):
        return lambda baas_thread: baas_thread.click(x, y)

    @classmethod
    def _gen_interfaces(cls):
        cls.I_main_page = Navigator.Interface(
            name="main_page",
            description="The main menu interface",
            features=[
                cls._validator_rgb_feature("main_page")
            ],
            actions={
                "collaboration_page": cls._action_click(565, 648)
            }
        )
        cls.I_group_page = Navigator.Interface(
            name="group_page",
            description="The group page interface",
            features=[
                cls._validator_image_feature("group_menu")
            ],
            actions={
                "main_page": cls._action_click(1238, 45)
            }
        )
        cls.I_collaboration_page = Navigator.Interface(
            name="collaboration_page",
            description="The collaboration event page interface",
            features=[
                cls._validator_image_feature("group_enter-button")
            ],
            actions={
                "group_page": cls._action_click(297, 380)
            }
        )

    @classmethod
    def list_interfaces(cls):
        cls._gen_interfaces()
        interfaces = []
        for name, val in vars(cls).items():
            if isinstance(val, Navigator.Interface):
                interfaces.append(val)
        return interfaces

    I_main_page: Navigator.Interface
    I_group_page: Navigator.Interface
