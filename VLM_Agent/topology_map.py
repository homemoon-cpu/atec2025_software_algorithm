from typing import Dict, List, Optional
import numpy as np
import faiss  # 用于特征向量快速检索

class TopologyNode:
    def __init__(self, node_id: int, 
                 semantic: str, 
                 feature: np.ndarray):
        self.id = node_id
        self.semantic = semantic          # VLM语义标签（e.g. "corridor", "door"）
        self.feature = feature            # 场景特征向量
        self.connections: Dict[int, dict] = {}  # {邻节点ID: {"explored": bool, "cost": float}}
        self.fully_explored = False      # 是否完成该节点的全方位探索
        self.landmarks: List[str] = []   # 包含的关键物体（e.g. ["fire_extinguisher"]）
        self.explored = False

class TopologyGraph:
    def __init__(self):
        self.nodes: Dict[int, TopologyNode] = {}
        self.node_index = faiss.IndexFlatL2(512)  # 特征索引
        self.current_id = 0
    
    def add_node(self, semantic: str, feature: np.ndarray) -> TopologyNode:
        new_node = TopologyNode(self.current_id, semantic, feature)
        self.nodes[self.current_id] = new_node
        self.node_index.add(feature.reshape(1, -1))
        self.current_id += 1
        return new_node
    
    def add_connection(self, from_id: int, to_id: int, cost: float):
        self.nodes[from_id].connections[to_id] = {"explored": False, "cost": cost}
        self.nodes[to_id].connections[from_id] = {"explored": False, "cost": cost}
    
    def match_node(self, feature: np.ndarray, threshold=0.85) -> Optional[int]:
        """基于特征的节点匹配"""
        distances, indices = self.node_index.search(feature.reshape(1, -1), 1)
        if distances[0][0] < (1 - threshold):
            return int(indices[0][0])
        return None
    
    def a_star_path(self, start: int, goal: int, heuristic) -> List[int]:
        """A*路径规划实现"""
        # 具体实现省略，可集成第三方路径库
        return []
    
    def find_optimal_target(self, heuristic_func) -> int:
        """基于启发式函数寻找最优目标节点"""
        # 遍历所有未完全探索的边
        candidates = []
        for node_id, node in self.nodes.items():
            if any(not conn["explored"] for conn in node.connections.values()):
                candidates.append(node_id)
        
        # 选择启发式代价最高的节点
        return max(candidates, key=lambda x: heuristic_func(x))
