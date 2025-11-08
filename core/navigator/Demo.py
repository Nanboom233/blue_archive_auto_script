from functools import partial
from typing import Optional

from navigator import NodeSpec, compile_specs, Index, lookup_next_hop, fallback_next_hop

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
        NodeSpec(
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
        NodeSpec(
            name="A2",
            description="node A2",
            features=[
                partial(example_feature_validator, "A2")
            ],
            actions={
                "Gate": partial(example_action_goto, "Gate")
            }
        ),
        NodeSpec(
            name="Gate",
            description="the gate node",
            features=[
                partial(example_feature_validator, "Gate")
            ],
            actions={
                "B2": partial(example_action_goto, "B2")
            }
        ),
        NodeSpec(
            name="B1",
            description="node B1",
            features=[
                partial(example_feature_validator, "B1")
            ],
            actions={
                "C1": partial(example_action_goto, "C1")
            }
        ),
        NodeSpec(
            name="B2",
            description="node B2",
            features=[
                partial(example_feature_validator, "B2")
            ],
            actions={}
        ),
        NodeSpec(
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


# ---------- 运行期：定位与执行 ----------
def resolve_current_node(index_instance: Index, priority: Optional[list[int]] = None) -> Optional[int]:
    node_volume = index_instance.node_volume

    # 1) 优先候选（目标/源），按顺序逐一校验
    seen: set[int] = set()
    if priority:
        for node_id in priority:
            if 0 <= node_id < node_volume and node_id not in seen:
                seen.add(node_id)
                if index_instance.validators[node_id]():
                    return node_id

    # 2) 全表扫描（按 _scan_order，命中后 move-to-front 以加速未来查询）
    for index, node_id in enumerate(index_instance.scan_order):
        if node_id in seen:
            continue
        if index_instance.validators[node_id]():
            # move-to-front（简单自适应）
            if index > 0:
                index_instance.scan_order.pop(index)
                index_instance.scan_order.insert(0, node_id)
            return node_id
    return None


def goto(index: Index, current_id: int, destination_id: int,
         path: Optional[list] = None, deprecated_path: Optional[set[tuple[int, int]]] = None) -> bool:
    if not path:
        if deprecated_path:
            # if any path is deprecated, we use runtime bfs to lookup instead of static path
            fallback_path = fallback_next_hop(index, current_id, destination_id, deprecated_path)
            if fallback_path is None:
                raise RuntimeError("No available path found (after deprecating edges).")
            return goto(index, current_id, destination_id,
                        path=fallback_path,
                        deprecated_path=deprecated_path)
        static_path = []
        tmp_id = current_id
        while tmp_id != destination_id:
            next_node_id = lookup_next_hop(index, tmp_id, destination_id)
            if next_node_id == -1:
                raise RuntimeError(f"No path exists from {index.id2name[tmp_id]} to {index.id2name[destination_id]}")
            static_path.append(next_node_id)
            tmp_id = next_node_id
        return goto(index, current_id, destination_id, path=static_path)
    print(
        f"Running path: {[index.id2name[nid] for nid in path]}, from {index.id2name[current_id]} to {index.id2name[destination_id]}")
    print(
        f"Deprecated edges: {[(index.id2name[u], index.id2name[v]) for u, v in deprecated_path]}") if deprecated_path else None
    for next_node_id in path:
        action = index.edge_handle.get((current_id, next_node_id))
        if action is None:
            return False
        try:
            action()
        except Exception as e:
            print(f"error executing action from {index.id2name[current_id]} to {index.id2name[next_node_id]}: {e}")
            if deprecated_path is None:
                deprecated_path = set()
            deprecated_path.add((current_id, next_node_id))
            return goto(index, current_id, destination_id, deprecated_path=deprecated_path)
        resolved_id = resolve_current_node(index, [next_node_id, current_id])
        if resolved_id is None:
            raise RuntimeError("cannot resolve current node during path execution")
        elif resolved_id == current_id:
            print(
                f"warning: path blocked from {index.id2name[current_id]} to {index.id2name[next_node_id]}, recalculating...")
            if deprecated_path is None:
                deprecated_path = set()
            deprecated_path.add((current_id, next_node_id))
            return goto(index, current_id, destination_id, deprecated_path=deprecated_path)
        elif resolved_id != next_node_id:
            print("warning: deviated from planned path, recalculating...")
            return goto(index, current_id, destination_id, deprecated_path=deprecated_path)
        current_id = resolved_id
    return True


def main():
    # 1) 初次构建
    specs = build_demo_specs()
    index: Index = compile_specs(specs, prev=None)

    # 运行期环境
    global CURRENT_NODE
    CURRENT_NODE = "A1"

    # 2) 从 A1 前往 B2，逐步 goto
    print("\n== goto A1 -> B2 ==")
    if not goto(index, index.name2id["A1"], index.name2id["B2"]):
        raise RuntimeError("Failed to reach B2 from A1")

    # 3) 仅修改 runnable（拓扑不变），不触发重建
    print("\n== change runnable only, topology unchanged ==")

    # 将 C1->B2 的动作替换成failed
    for s in specs:
        if s.name == "C1":
            s.actions["B2"] = partial(example_failed_action)

    # 重新编译：应复用索引，只更新行为
    index = compile_specs(specs, prev=index)

    CURRENT_NODE = "A1"
    print("\n== goto A1 -> B2 with failed runnable (fallback BFS) ==")
    if not goto(index, index.name2id["A1"], index.name2id["B2"]):
        raise RuntimeError("Failed to reach B2 from A1")

    # 5) 增量：新增边 A1->Gate，仅重建受影响的目的地列
    print("\n== incremental: add edge A1->Gate ==")
    for s in specs:
        if s.name == "A1":
            # 新增一条到 Gate 的动作（新的 runnable）
            s.actions["Gate"] = partial(example_action_goto, "Gate")
    index = compile_specs(specs, prev=index)

    # 观察路径是否更短（A1->Gate->B2）
    CURRENT_NODE = "A1"
    print("\n== goto A1 -> B2 after edge added ==")
    if not goto(index, index.name2id["A1"], index.name2id["B2"]):
        raise RuntimeError("Failed to reach B2 from A1")

    print("\nDone.")


if __name__ == "__main__":
    main()
