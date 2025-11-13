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
    def _validator_any(*args: Callable[[Navigator.Baas_thread], bool]):
        return lambda baas_thread: any(validator(baas_thread) for validator in args)

    @staticmethod
    def _action_click(x: int, y: int):
        return lambda baas_thread: baas_thread.click(x, y)

    @staticmethod
    def _action_delay(time_sec: float):
        return lambda baas_thread: time.sleep(time_sec)

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
        cls.I_group_page_reward = Navigator.Interface(
            name="group_reward",
            description="The group 10 AP reward popup interface",
            features=[
                cls._validator_image_feature("group_sign-up-reward")
            ],
            actions={
                "group_page": cls._action_click(920, 159)
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
        cls.I_main_page_news = Navigator.Interface(
            name="main_page_news",
            description="The main page news popup interface",
            features=[
                cls._validator_image_feature("main_page_news")
            ],
            actions={
                "main_page": cls._action_click(1142, 104)
            }
        )

        # WILDCARD INTERFACES
        cls.W_loading = Navigator.Interface(
            name="loading_indicator",
            description="The loading indicator",
            features=[
                cls._validator_rgb_feature("loadingNotWhite"),
                cls._validator_rgb_feature("loadingWhite")
            ],
            actions={
                "*": cls._action_delay(2)
            }
        )
        cls.W_daily_attendance = Navigator.Interface(
            name="daily_attendance",
            description="The daily attendance popup interface",
            features=[
                cls._validator_image_feature("main_page_daily-attendance")
            ],
            actions={
                "*": cls._action_click(640, 360)
            }
        )
        cls.W_network_unstable = Navigator.Interface(
            name="network_unstable",
            description="The network unstable popup interface",
            features=[
                cls._validator_image_feature("main_page_net-work-unstable")
            ],
            actions={
                "*": cls._action_click(753, 468)
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
    I_group_page_reward: Navigator.Interface
    I_main_page_news: Navigator.Interface

    W_loading: Navigator.Interface
    W_daily_attendance: Navigator.Interface
    W_network_unstable: Navigator.Interface
