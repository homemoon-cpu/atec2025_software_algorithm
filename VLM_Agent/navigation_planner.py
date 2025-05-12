from VLM_Agent.topology_map import TopologyGraph, TopologyNode
from typing import Dict, List, Optional

class NavigationPlanner:
    @staticmethod
    def plan_next_move(current_node: TopologyNode, 
                       topology: TopologyGraph) -> Optional[int]:
        """基于当前拓扑的状态规划下一个目标节点"""
        # 优先选择未探索的连接边
        for neighbor_id, cost in current_node.connections.items():
            if not topology.nodes[neighbor_id].explored:
                return neighbor_id
        # 如果没有未探索的相邻节点，使用图搜索算法寻找最近的未探索区域
        return None