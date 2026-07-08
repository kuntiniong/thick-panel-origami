from utils import distance3D

class SpatialHash:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}
    
    def _key(self, point):
        return (int(point[0] / self.cell_size),
                int(point[1] / self.cell_size),
                int(point[2] / self.cell_size) if len(point) > 2 else 0)
    
    def insert(self, point, idx):
        k = self._key(point)
        if k not in self.grid:
            self.grid[k] = []
        self.grid[k].append((idx, point))
    
    def find(self, point, tolerance):
        k = self._key(point)
        # 检查当前格 + 相邻格 (3x3x3 = 27 cells)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    key = (k[0]+dx, k[1]+dy, k[2]+dz)
                    if key in self.grid:
                        for idx, pt in self.grid[key]:
                            if distance3D(point, pt) < tolerance:
                                return idx
        return -1
