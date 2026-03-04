from dataclasses import dataclass


@dataclass
class StructureConfig:
    use_ids: bool = True  # 是否使用字符ID作为条件
    use_component_mask: bool = True  # 是否使用组件掩码作为条件
    use_skeleton: bool = False  # 是否使用骨架作为条件
    use_edge_map: bool = True  # 是否使用边缘图作为条件
