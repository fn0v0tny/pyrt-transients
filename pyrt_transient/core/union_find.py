"""Union-Find (Disjoint Set) data structure, copied verbatim from transient_analyser.py."""

from collections import defaultdict
from typing import Dict, List


class UnionFind:
    """Union-Find (Disjoint Set) data structure for efficient connected components."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if components were merged."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.n_components -= 1
        return True

    def get_components(self) -> Dict[int, List[int]]:
        """Get all connected components as dict of root -> [members]."""
        components = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            components[root].append(i)
        return dict(components)
