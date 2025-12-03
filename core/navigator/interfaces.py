import black
import libcst
import libcst.matchers

from core.color import match_rgb_feature
from core.picture import match_img_feature


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
        cls.I_collaboration_page = Navigator.Interface(
            name="collaboration_page",
            description="The collaboration event page interface",
            features=[
                cls._validator_image_feature("group_enter-button"),
            ],
            actions={
                "group_page": cls._action_click(297, 380),
            },
        )
        cls.I_group_page = Navigator.Interface(
            name="group_page",
            description="The group page interface",
            features=[
                cls._validator_image_feature("group_menu"),
            ],
            actions={
                "main_page": cls._action_click(1238, 45),
            },
        )
        cls.I_group_page_reward = Navigator.Interface(
            name="group_reward",
            description="The group 10 AP reward popup interface",
            features=[
                cls._validator_image_feature("group_sign-up-reward"),
            ],
            actions={
                "group_page": cls._action_click(920, 159),
            },
        )
        cls.I_main_page = Navigator.Interface(
            name="main_page",
            description="The main menu interface",
            features=[
                cls._validator_rgb_feature("main_page"),
            ],
            actions={
                "collaboration_page": cls._action_click(565, 648),
            },
        )
        cls.I_main_page_news = Navigator.Interface(
            name="main_page_news",
            description="The main page news popup interface",
            features=[
                cls._validator_image_feature("main_page_news"),
            ],
            actions={
                "main_page": cls._action_click(1142, 104),
            },
        )

        # WILDCARD INTERFACES
        cls.W_daily_attendance = Navigator.Interface(
            name="daily_attendance",
            description="The daily attendance popup interface",
            features=[
                cls._validator_image_feature("main_page_daily-attendance"),
            ],
            actions={
                "*": cls._action_click(640, 360),
            },
        )
        cls.W_loading = Navigator.Interface(
            name="loading_indicator",
            description="The loading indicator",
            features=[
                cls._validator_rgb_feature("loadingNotWhite"),
                cls._validator_rgb_feature("loadingWhite"),
            ],
            actions={
                "*": cls._action_delay(2),
            },
        )
        cls.W_network_unstable = Navigator.Interface(
            name="network_unstable",
            description="The network unstable popup interface",
            features=[
                cls._validator_image_feature("main_page_net-work-unstable"),
            ],
            actions={
                "*": cls._action_click(753, 468),
            },
        )

    @classmethod
    def list_interfaces(cls):
        cls._gen_interfaces()
        interfaces = []
        for name, val in vars(cls).items():
            if isinstance(val, Navigator.Interface):
                interfaces.append(val)
        return interfaces

    I_collaboration_page: Navigator.Interface
    I_group_page: Navigator.Interface
    I_group_page_reward: Navigator.Interface
    I_main_page: Navigator.Interface
    I_main_page_news: Navigator.Interface
    W_daily_attendance: Navigator.Interface
    W_loading: Navigator.Interface
    W_network_unstable: Navigator.Interface


class InterfacesCSTEditor:
    cst_tree: libcst.Module

    class InterfaceVisitor(libcst.CSTVisitor):
        _in_Interfaces_class: bool = False
        _in_gen_interfaces_function: bool = False
        interfaces: list[dict] = []

        @staticmethod
        def _get_literal(node: libcst.CSTNode) -> str | int | float | libcst.CSTNode:
            """
            Args:
                node: A CST node representing a literal value.

            Returns:
                The literal value as str, int, float, or CSTNode itself if not a recognized literal.
            """
            if isinstance(node, libcst.SimpleString):
                return node.value.strip("'").strip('"')
            if isinstance(node, libcst.Integer):
                return int(node.value)
            if isinstance(node, libcst.Float):
                return float(node.value)
            return node  # return CSTNode as fallback

        @staticmethod
        def _unpack_attribute(node: libcst.BaseExpression) -> str:
            func_name = ""
            while isinstance(node, libcst.Attribute):
                parent_str = (
                    node.value.value if isinstance(node.value, libcst.Name) else str(node.value)
                )
                func_name += parent_str + "."
                node = node.attr.value
            func_name += node.value if isinstance(node, libcst.Name) else str(node)
            return func_name

        @classmethod
        def _parse_runnable(cls, node: libcst.BaseExpression) -> dict | libcst.CSTNode:
            if isinstance(node, libcst.Call):
                func_name = cls._unpack_attribute(node.func)
                args = [cls._get_literal(arg.value) for arg in node.args]
                return {
                    "func_name": func_name,
                    "args": [(str(x) if isinstance(x, libcst.CSTNode) else x) for x in args],
                }
            return node  # fallback if not a call

        def visit_ClassDef(self, node: libcst.ClassDef) -> bool:
            if node.name.value == "Interfaces":
                self._in_Interfaces_class = True
                return True
            return False

        def leave_ClassDef(self, original_node: libcst.ClassDef) -> None:
            if original_node.name.value == "Interfaces":
                self._in_Interfaces_class = False

        def visit_FunctionDef(self, node: libcst.FunctionDef) -> bool:
            if self._in_Interfaces_class and node.name.value == "_gen_interfaces":
                self._in_gen_interfaces_function = True
                return True
            return False

        def leave_FunctionDef(self, original_node: libcst.FunctionDef) -> None:
            if self._in_gen_interfaces_function and original_node.name.value == "_gen_interfaces":
                self._in_gen_interfaces_function = False

        def visit_Assign(self, node: libcst.Assign) -> Optional[bool]:
            if self._in_gen_interfaces_function:
                # unpack interface name
                assign_target: libcst.BaseAssignTargetExpression = node.targets[0].target
                if not (
                    isinstance(assign_target, libcst.Attribute)
                    and assign_target.value.value == "cls"
                ):
                    return None
                interface_name = assign_target.attr.value

                # unpack Interface instantiation
                call_node = node.value
                if not isinstance(call_node, libcst.Call):
                    return None

                # unpack name and description
                kwargs = {}
                for arg in call_node.args:
                    if arg.keyword:
                        kwargs[arg.keyword.value] = arg.value

                name: str = self._get_literal(kwargs.get("name"))
                desc: str = self._get_literal(kwargs.get("description"))

                # unpack features
                features: list[dict | libcst.CSTNode] = []
                features_node = kwargs.get("features")
                if isinstance(features_node, libcst.List):
                    for element in features_node.elements:
                        features.append(self._parse_runnable(element.value))

                # unpack actions
                actions = {}
                actions_node = kwargs.get("actions")
                if isinstance(actions_node, libcst.Dict):
                    for element in actions_node.elements:
                        k = self._get_literal(element.key)
                        v = self._parse_runnable(element.value)
                        actions[k] = v
                self.interfaces.append(
                    {
                        "var_name": interface_name,
                        "id": name,
                        "description": desc,
                        "features": features,
                        "actions": actions,
                    }
                )
            return True

    class InterfaceTransformer(libcst.CSTTransformer):
        new_interfaces: list[dict]

        def __init__(self, new_interfaces: list[dict]):
            self.new_interfaces = new_interfaces
            super().__init__()

        @staticmethod
        def _build_target(var_name: str):
            splits_var_name = var_name.split(".")
            target = libcst.Name(splits_var_name[0])
            for attr in splits_var_name[1:]:
                target = libcst.Attribute(value=target, attr=libcst.Name(attr))
            return target

        @staticmethod
        def _create_literal(value):
            if isinstance(value, str):
                return libcst.SimpleString(f'"{value}"')
            if isinstance(value, int):
                return libcst.Integer(str(value))
            if isinstance(value, float):
                return libcst.Float(str(value))
            return libcst.SimpleString(str(value))

        @classmethod
        def _create_runnable_node(cls, runnable_specs: dict) -> libcst.Call:
            func_target = cls._build_target(runnable_specs["func_name"])
            cst_args = [
                libcst.Arg(value=cls._create_literal(arg)) for arg in runnable_specs["args"]
            ]
            return libcst.Call(func=func_target, args=cst_args)

        @classmethod
        def _create_interface_node(cls, interface_info: dict):
            # Features list
            feat_elements = [
                libcst.Element(value=cls._create_runnable_node(f), comma=libcst.Comma())
                for f in interface_info["features"]
            ]

            # Actions dict
            act_elements = [
                libcst.DictElement(
                    key=cls._create_literal(k),
                    value=cls._create_runnable_node(v),
                    comma=libcst.Comma(),
                )
                for k, v in interface_info["actions"].items()
            ]

            return libcst.Call(
                func=libcst.Attribute(
                    value=libcst.Name("Navigator"), attr=libcst.Name("Interface")
                ),
                args=[
                    libcst.Arg(
                        keyword=libcst.Name("name"), value=cls._create_literal(interface_info["id"])
                    ),
                    libcst.Arg(
                        keyword=libcst.Name("description"),
                        value=cls._create_literal(interface_info["description"]),
                    ),
                    libcst.Arg(
                        keyword=libcst.Name("features"), value=libcst.List(elements=feat_elements)
                    ),
                    libcst.Arg(
                        keyword=libcst.Name("actions"), value=libcst.Dict(elements=act_elements)
                    ),
                ],
            )

        def leave_FunctionDef(
            self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
        ):
            if original_node.name.value == "_gen_interfaces":
                # rewrite _gen_interfaces
                new_body_statements = []

                comment_filled = False
                for interface in self.new_interfaces:
                    if not comment_filled and interface["var_name"].startswith("W_"):
                        comment = libcst.EmptyLine(comment=libcst.Comment("# WILDCARD INTERFACES"))
                        new_body_statements.append(libcst.EmptyLine())
                        new_body_statements.append(comment)
                        comment_filled = True
                    assign = libcst.Assign(
                        targets=[
                            libcst.AssignTarget(
                                target=libcst.Attribute(
                                    value=libcst.Name("cls"),
                                    attr=libcst.Name(interface["var_name"]),
                                )
                            )
                        ],
                        value=self._create_interface_node(interface),
                    )
                    new_body_statements.append(libcst.SimpleStatementLine(body=[assign]))

                return updated_node.with_changes(
                    body=libcst.IndentedBlock(body=new_body_statements)
                )

            return updated_node

        def leave_ClassDef(self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef):
            # rewrite Interfaces type hints
            if original_node.name.value == "Interfaces":
                new_body = list(updated_node.body.body)

                filtered_body = []
                for stmt in new_body:
                    is_interface_hint = False
                    if isinstance(stmt, libcst.SimpleStatementLine):
                        if isinstance(stmt.body[0], libcst.AnnAssign):
                            ann = stmt.body[0].annotation
                            if libcst.matchers.matches(
                                ann.annotation,
                                libcst.matchers.Attribute(
                                    value=libcst.matchers.Name("Navigator"),
                                    attr=libcst.matchers.Name("Interface"),
                                ),
                            ):
                                is_interface_hint = True

                    if not is_interface_hint:
                        filtered_body.append(stmt)

                for interface in self.new_interfaces:
                    ann_assign = libcst.AnnAssign(
                        target=libcst.Name(interface["var_name"]),
                        annotation=libcst.Annotation(
                            annotation=libcst.Attribute(
                                value=libcst.Name("Navigator"), attr=libcst.Name("Interface")
                            )
                        ),
                    )
                    filtered_body.append(libcst.SimpleStatementLine(body=[ann_assign]))

                return updated_node.with_changes(body=libcst.IndentedBlock(body=filtered_body))

            return updated_node

    def __init__(self, cst_tree: Optional[libcst.Module] = None):
        if not cst_tree:
            with open(__file__, "r", encoding="utf-8") as f:
                source_code = f.read()
            cst_tree = libcst.parse_module(source_code)
        self.cst_tree = cst_tree

    def read_interfaces(self) -> list[dict]:
        visitor = self.InterfaceVisitor()
        self.cst_tree.visit(visitor)
        return visitor.interfaces

    def generate_code(self, interfaces: list[dict]) -> str:
        interfaces.sort(key=lambda x: x["var_name"])
        transformer = InterfacesCSTEditor.InterfaceTransformer(interfaces)
        new_tree = self.cst_tree.visit(transformer)
        new_code = black.format_str(new_tree.code, mode=black.Mode(line_length=100))
        return new_code

    @staticmethod
    def get_interfaces_file_path() -> str:
        return __file__
