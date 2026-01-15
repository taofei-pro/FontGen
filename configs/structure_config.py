from dataclasses import dataclass


@dataclass
class StructureConfig:
    use_ids: bool = True
    use_component_mask: bool = True
    use_skeleton: bool = False
    use_edge_map: bool = True
