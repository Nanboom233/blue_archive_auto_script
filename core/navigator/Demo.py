from functools import partial

from .navigator import Navigator

CURRENT_NODE: str = ""


def example_action_goto(destination: str):
    global CURRENT_NODE
    print(f"Action: going from {CURRENT_NODE} to {destination}")
    CURRENT_NODE = destination


def example_failed_action():
    return


def example_feature_validator(node_name: str) -> bool:
    return CURRENT_NODE == node_name


def build_demo_specs():
    specs = [
        Navigator.Interface(
            name="A1",
            description="node A1",
            features=[
                partial(example_feature_validator, "A1")
            ],
            actions={
                "A2": partial(example_action_goto, "A2"),
                "B1": partial(example_action_goto, "B1")
            }
        ),
        Navigator.Interface(
            name="A2",
            description="node A2",
            features=[
                partial(example_feature_validator, "A2")
            ],
            actions={
                "Gate": partial(example_action_goto, "Gate")
            }
        ),
        Navigator.Interface(
            name="Gate",
            description="the gate node",
            features=[
                partial(example_feature_validator, "Gate")
            ],
            actions={
                "B2": partial(example_action_goto, "B2")
            }
        ),
        Navigator.Interface(
            name="B1",
            description="node B1",
            features=[
                partial(example_feature_validator, "B1")
            ],
            actions={
                "C1": partial(example_action_goto, "C1")
            }
        ),
        Navigator.Interface(
            name="B2",
            description="node B2",
            features=[
                partial(example_feature_validator, "B2")
            ],
            actions={}
        ),
        Navigator.Interface(
            name="C1",
            description="node C1",
            features=[
                partial(example_feature_validator, "C1")
            ],
            actions={
                "B2": partial(example_action_goto, "B2"),
                "A1": partial(example_action_goto, "A1")
            }
        )
    ]
    return specs


def main():
    interfaces = build_demo_specs()
    navigator = Navigator(interfaces)

    # test saving and reading
    assert isinstance(navigator.metadata, Navigator.FullMetadata)
    navigator.metadata.save("navigator-full.metadata")  # full metadata
    navigator.metadata.to_base().save("navigator-runtime.metadata")  # runtime base metadata
    del navigator

    # use runtime base metadata
    navigator = Navigator(interfaces, Navigator.load_metadata("navigator-full.metadata"))

    # 运行期环境
    global CURRENT_NODE
    CURRENT_NODE = "A1"

    # 2) 从 A1 前往 B2，逐步 goto
    print("\n== goto A1 -> B2 ==")
    if not navigator.goto("A1", "B2"):
        raise RuntimeError("Failed to reach B2 from A1")

    # 3) 仅修改 runnable（拓扑不变），不触发重建
    print("\n== change runnable only, topology unchanged ==")
    # 将 C1->B2 的动作替换成failed
    navigator.update_single_action(
        "C1",
        "B2",
        example_failed_action
    )

    CURRENT_NODE = "A1"
    print("\n== goto A1 -> B2 with failed runnable (fallback BFS) ==")
    if not navigator.goto("A1", "B2"):
        raise RuntimeError("Failed to reach B2 from A1")

    # 5) 增量：新增边 A1->Gate，仅重建受影响的目的地列
    print("\n== incremental: add edge A1->Gate ==")
    for interface in interfaces:
        if interface.name == "A1":
            # 新增一条到 Gate 的动作（新的 runnable）
            interface.actions["Gate"] = partial(example_action_goto, "Gate")
    # if the metadata is runtime base, it will fully rebuild
    # otherwise, it will do incremental rebuild
    navigator.rebuild(interfaces)

    # 观察路径是否更短（A1->Gate->B2）
    CURRENT_NODE = "A1"
    print("\n== goto A1 -> B2 after edge added ==")
    if not navigator.goto("A1", "B2"):
        raise RuntimeError("Failed to reach B2 from A1")

    print("\nDone.")


if __name__ == "__main__":
    main()
