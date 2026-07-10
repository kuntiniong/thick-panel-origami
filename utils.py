import math
import numpy as np
from copy import deepcopy

VALLEY = 0
MOUNTAIN = 1
BORDER = 2
CUTTING = 3
HOLE = 4
HINGE_HOLE = 5
FACET = 6

UP = 1
DOWN = 0
MIDDLE = 2

FOLDING_MAXIMUM = 0.95

X = 0
Y = 1
Z = 2

START = 0
END = 1
LEFT = 0
RIGHT = 1

V = True
ACTIVE_MIURA = 1
PASSIVE_MIURA = 0

EMPTY = 0
HAVE_BORDER = 1
HAVE_CONNECTION = 2

SHOW_COLOR = 1
BLACK_COLOR = 0

LEFT_HALF = 0
RIGHT_HALF = 1
NO_HALF = 2

BOTTOM = 0
PASS = 1
TOP = 2

FREE = 0
FIXED = 1

REF_TARGET_ANGLE = 0

class Vertex:
    """
    折纸顶点类，表示折纸图案中的一个关键点（节点）。
    存储节点坐标、维度、连接关系及相关系数信息。

    Origami vertex class representing a key point (node) in the origami pattern.
    Stores node coordinates, dimension, connectivity, and coefficient information.
    """
    def __init__(self, kp):
        """
        初始化顶点。
        中文：根据给定的坐标点初始化顶点对象，设置维度、度数、连接索引等属性。

        Initialize a vertex.
        English: Initialize the vertex object with the given coordinate point,
                 setting dimension, degree, connection index, and other attributes.

        :param kp: 顶点坐标（列表，如 [x, y] 或 [x, y, z]） / Vertex coordinates (list, e.g. [x, y] or [x, y, z])
        """
        self.kp = kp
        self.dim = len(kp)
        self.dn = 0
        self.connection_index = []
        self.is_border_node = False
        self.coeff_list = []
        self.level_list = []
    
    def getKp(self):
        """
        获取顶点坐标。
        中文：返回顶点的坐标列表。

        Get vertex coordinates.
        English: Return the coordinate list of the vertex.

        :return: 顶点坐标列表 / Vertex coordinate list
        """
        return self.kp

    def __repr__(self):
        return f"Degree: {self.dn}, {0 if self.is_border_node else 1}"

class Crease:
    """
    折痕类，表示折纸图案中的一条折痕线段。
    包含折痕的起点、终点、类型（山折/谷折/边界/切割/孔等），
    以及折叠角度范围、层级等信息。

    Crease class representing a crease line segment in the origami pattern.
    Contains the crease's start/end points, type (mountain/valley/border/cutting/hole, etc.),
    folding angle bounds, level, and other information.
    """
    def __repr__(self):
        return f"S: {[round(self.points[START][i], 2) for i in range(len(self.points[START]))]}, E: {[round(self.points[END][i], 2) for i in range(len(self.points[START]))]}, {self.crease_type}"
    
    def __init__(self, start, end, crease_type, show_color_flag=True, hard=False, upper=math.pi, lower=-math.pi, height=0.0) -> None:
        """
        初始化折痕。
        中文：根据起点、终点和类型创建折痕对象，设置硬折痕、折叠角度上下界、层级等。

        Initialize a crease.
        English: Create a crease object from start/end points and type,
                 with settings for hard crease, folding angle bounds, level, etc.

        :param start: 折痕起点坐标 / Start point coordinates of the crease
        :param end: 折痕终点坐标 / End point coordinates of the crease
        :param crease_type: 折痕类型（MOUNTAIN/VALLEY/BORDER/CUTTING/HOLE等） / Crease type (MOUNTAIN/VALLEY/BORDER/CUTTING/HOLE, etc.)
        :param show_color_flag: 是否显示折痕颜色，False则强制为BORDER / Whether to show crease color; False forces BORDER type
        :param hard: 是否为硬折痕（不可折叠） / Whether the crease is hard (non-foldable)
        :param upper: 折叠角度上界（弧度） / Upper bound of folding angle (radians)
        :param lower: 折叠角度下界（弧度） / Lower bound of folding angle (radians)
        :param height: 厚板高度（用于厚板折纸） / Thick panel height (for thick-panel origami)
        """
        self.points = [start, end]
        if show_color_flag:
            self.crease_type = crease_type
        else:
            self.crease_type = BORDER
        self.hard = hard
        self.folding_angle_upper_bound = upper
        self.folding_angle_lower_bound = lower
        self.index = -1
        self.origin_index = -1

        self.start_index = 0
        self.end_index = 0
        self.level = 0
        self.coeff = 1.0

        self.visited = False
        self.undefined = False
        self.recover_level = []
        self.recover_angle = []
        
        self.thick_panel_height = height

        self.sign = "equal"
    
    def setHard(self, hard: bool):
        """
        设置折痕是否为硬折痕。
        中文：若为True，该折痕不可被折叠（刚性连接）。

        Set whether the crease is hard (rigid).
        English: If True, this crease cannot be folded (rigid connection).

        :param hard: True表示硬折痕，False表示可折叠 / True for rigid crease, False for foldable
        """
        self.hard = hard

    def getType(self):
        """
        获取折痕类型。
        中文：返回折痕的类型（MOUNTAIN/VALLEY/BORDER/CUTTING/HOLE等）。

        Get the crease type.
        English: Return the type of the crease (MOUNTAIN/VALLEY/BORDER/CUTTING/HOLE, etc.).

        :return: 折痕类型整数常量 / Integer constant representing the crease type
        """
        return self.crease_type

    def setIndex(self, index):
        """
        设置折痕的全局索引。
        中文：用于标识折痕在整体折纸结构中的位置。

        Set the global index of the crease.
        English: Used to identify the crease's position in the overall origami structure.

        :param index: 折痕索引 / Crease index
        """
        self.index = index

    def setOriginIndex(self, index):
        """
        设置折痕的原始索引（对应设计源数据中的位置）。
        中文：用于追踪折痕在原始数据中的来源。

        Set the origin index of the crease (position in the original design source data).
        English: Used to trace the crease back to its source in the original data.

        :param index: 原始索引 / Origin index
        """
        self.origin_index = index

    def getReverse(self):
        """
        获取反向折痕（起终点互换的深拷贝）。
        中文：返回一条与当前折痕方向相反的新折痕对象（深拷贝）。

        Get the reversed crease (deep copy with start/end swapped).
        English: Return a new crease object (deep copy) with start and end points swapped.

        :return: 反向折痕对象 / Reversed crease object
        """
        new = deepcopy(self)
        new.points.reverse()
        return new
    
    def getMidPoint(self):
        """
        获取折痕的中点坐标。
        中文：返回折痕起点和终点的中点。

        Get the midpoint of the crease.
        English: Return the midpoint between the start and end of the crease.

        :return: 中点坐标列表 / Midpoint coordinate list
        """
        return [(self.points[START][i] + self.points[END][i]) / 2 for i in range(len(self.points[START]))]
            
    def getDirection(self):
        """
        获取折痕方向的单位向量（2D）。
        中文：返回从起点到终点的归一化方向向量。

        Get the unit direction vector of the crease (2D).
        English: Return the normalized direction vector from start to end point.

        :return: 单位方向向量 [dx, dy] / Unit direction vector [dx, dy]
        """
        return [(self.points[END][X] - self.points[START][X]) / distance(self.points[START], self.points[END]), (self.points[END][Y] - self.points[START][Y]) / distance(self.points[START], self.points[END])]

    def getNormal(self):
        """
        获取折痕的单位法向量（2D，逆时针旋转90度）。
        中文：返回方向向量逆时针旋转90度得到的单位法向量。

        Get the unit normal vector of the crease (2D, rotated 90° counter-clockwise).
        English: Return the unit normal vector obtained by rotating the direction vector 90° CCW.

        :return: 单位法向量 [nx, ny] / Unit normal vector [nx, ny]
        """
        return [(self.points[END][Y] - self.points[START][Y]) / distance(self.points[START], self.points[END]), -(self.points[END][X] - self.points[START][X]) / distance(self.points[START], self.points[END])]
    
    def getPercentPoint(self, percent):
        """
        获取折痕上指定百分比位置的点坐标。
        中文：按给定比例在折痕上插值，0为起点，1为终点。

        Get the coordinate of a point at a given percentage along the crease.
        English: Interpolate along the crease at the given percentage (0=start, 1=end).

        :param percent: 插值比例 [0, 1] / Interpolation ratio [0, 1]
        :return: 插值点坐标 [x, y] / Interpolated point coordinates [x, y]
        """
        return [
            percent * (self.points[END][X] - self.points[START][X]) + self.points[START][X], 
            percent * (self.points[END][Y] - self.points[START][Y]) + self.points[START][Y], 
        ]

    def k(self):
        """
        计算折痕所在直线的斜率 k。
        中文：若折痕为竖线（x坐标相同），返回 math.inf。

        Calculate the slope k of the line containing the crease.
        English: Returns math.inf if the crease is vertical (same x-coordinate).

        :return: 斜率 k 或 math.inf / Slope k or math.inf
        """
        if self.points[END][X] != self.points[START][X]:
            k = (self.points[END][Y] - self.points[START][Y]) / (self.points[END][X] - self.points[START][X])
            return k
        else:
            return math.inf

    def b(self):
        """
        计算折痕所在直线的截距 b（y = kx + b）。
        中文：若折痕为竖线，返回折痕的x坐标值作为截距（特殊约定）。

        Calculate the y-intercept b of the line containing the crease (y = kx + b).
        English: For vertical creases, returns the x-coordinate as the intercept (special convention).

        :return: 截距 b 或 x 坐标（竖线情况） / Intercept b, or x-coordinate for vertical lines
        """
        if self.points[START][X] != self.points[END][X]:
            b = (self.points[END][X] * self.points[START][Y] - self.points[START][X] * self.points[END][Y]) / (self.points[END][X] - self.points[START][X])
            return b
        else:
            return self.points[END][X]

    def getPointAxis(self):
        """
        获取折痕起点和终点的坐标分量。
        中文：返回起点x、起点y、终点x、终点y四个坐标值。

        Get the axis components of the crease's start and end points.
        English: Return the x/y of start and end: (x1, y1, x2, y2).

        :return: (x1, y1, x2, y2) / (start_x, start_y, end_x, end_y)
        """
        start = self.points[START]
        end = self.points[END]
        # if start[0] <= end[0]:
        return start[X], start[Y], end[X], end[Y]
        # else:
        #     return end[0], end[1], start[0], start[1]
    
    def getLength(self):
        """
        获取折痕的长度（支持3D）。
        中文：计算并返回折痕在三维空间中的欧氏距离。

        Get the length of the crease (supports 3D).
        English: Calculate and return the Euclidean distance of the crease in 3D space.

        :return: 折痕长度（浮点数） / Crease length (float)
        """
        return distance3D(self.points[START], self.points[END])
    
    def pointIsStartAndEnd(self, p1, p2):
        """
        判断给定的两个点是否分别为折痕的起点和终点（或反向）。
        中文：检测p1、p2是否与折痕的两端点重合（允许正向和反向匹配）。

        Check if the given two points are the start and end (or reverse) of the crease.
        English: Check whether p1, p2 coincide with the two endpoints of the crease
                 (supports both forward and reverse matching).

        :param p1: 第一个点坐标 / First point coordinates
        :param p2: 第二个点坐标 / Second point coordinates
        :return: True表示匹配，False表示不匹配 / True if matched, False otherwise
        """
        if (distance(p1, self.points[START]) < 1e-5 and distance(p2, self.points[END]) < 1e-5):
            return True
        else:
            if (distance(p1, self.points[END]) < 1e-5 and distance(p2, self.points[START]) < 1e-5):
                return True
            else:
                return False

    def __getitem__(self, index):
        return self.points[index]

def calculatePercent(c1: Crease, c2: Crease):
    """
    计算两条折痕的交点在各自折痕上的位置百分比列表。
    中文：返回一个包含5组 [c1上的比例, c2上的比例] 的列表，用于分析折痕间的位置关系。
    若两折痕平行或无交点，返回 None。

    Calculate the intersection percentage list of two creases along their respective lengths.
    English: Returns a list of 5 pairs [ratio_on_c1, ratio_on_c2] to analyze relative positions.
    Returns None if the creases are parallel or have no intersection.

    :param c1: 第一条折痕 / First crease
    :param c2: 第二条折痕 / Second crease
    :return: 百分比列表（5个[p1, p2]对）或 None / List of 5 [p1, p2] pairs, or None
    """
    try:
        k1 = c1.k()
        k2 = c2.k()
        b1 = c1.b()
        b2 = c2.b()
        x11, y11, x21, y21 = c1.getPointAxis()
        x12, y12, x22, y22 = c2.getPointAxis()
        if(k1 != math.inf and k2 != math.inf):
            if(k1 == k2):
                return None
            x0 = (b2 - b1) / (k1 - k2)
            y0 = (k1 * b2 - k2 * b1) / (k1 - k2)
            cross_percent = (x0 - x11) / (x21 - x11)
            percent_list = [
                [
                    0.0, (x11 - x12) / (x22 - x12)
                ],
                [
                    1.0, (x21 - x12) / (x22 - x12)
                ],
                [
                    (x12 - x11) / (x21 - x11), 0.0
                ],
                [
                    (x22 - x11) / (x21 - x11), 1.0
                ],
                [
                    cross_percent, (x0 - x12) / (x22 - x12)
                ]
            ]
            if cross_percent >= 0.0 and cross_percent <= 1.0:
                return percent_list
            else:
                return None
        elif (k1 == math.inf and k2 == math.inf):
            return None
        elif (k1 == math.inf):
            x0 = x11
            y0 = y12 + (x0 - x12) / (x22 - x12) * (y22 - y12)
            cross_percent = (x0 - x12) / (x22 - x12)
            percent_list = [
                [
                    0.0, (x11 - x12) / (x22 - x12)
                ],
                [
                    1.0, (x21 - x12) / (x22 - x12)
                ],
                [
                    None, 0.0
                ],
                [
                    None, 1.0
                ],
                [
                    None, cross_percent
                ]
            ]
            if cross_percent >= 0.0 and cross_percent <= 1.0:
                return percent_list
            else:
                return None
        else:
            x0 = x12
            y0 = y11 + (x0 - x11) / (x21 - x11) * (y21 - y11)
            cross_percent = (x0 - x11) / (x21 - x11)
            percent_list = [
                [
                    0.0, None
                ],
                [
                    1.0, None
                ],
                [
                    (x12 - x11) / (x21 - x11), 0.0
                ],
                [
                    (x22 - x11) / (x21 - x11), 1.0
                ],
                [
                    cross_percent, None
                ]
            ]
            if cross_percent >= 0.0 and cross_percent <= 1.0:
                return percent_list
            else:
                return None
    except:
        return None

def percent_limit(x):
    """
    将值截断到 [0.0, 1.0] 范围内。
    中文：若x大于1.0返回1.0，小于0.0返回0.0，否则返回原值。

    Clamp a value to the range [0.0, 1.0].
    English: Returns 1.0 if x > 1.0, 0.0 if x < 0.0, otherwise returns x as-is.

    :param x: 输入值 / Input value
    :return: 截断后的值 / Clamped value
    """
    try:
        if x > 1.0:
            return 1.0
        elif x < 0.0:
            return 0.0
        else:
            return x
    except:
        return x

def limit(data, down, upper, matchtype="any", strict=False, tolerance_expand=0.0) -> bool:
    """
    判断数据列表中的元素是否落在指定区间内。
    中文：
        matchtype="any"：任意一个元素在 [down, upper] 内即返回 True。
        matchtype="all"（其他值）：所有元素均在 [down, upper] 内才返回 True。
        strict=True 时使用严格不等号（开区间），否则使用闭区间。
        tolerance_expand 可以扩展区间范围。

    Check whether elements of a data list fall within a specified range.
    English:
        matchtype="any": returns True if any element is within [down, upper].
        matchtype="all" (other values): returns True only if all elements are within [down, upper].
        strict=True uses strict inequality (open interval), otherwise closed interval.
        tolerance_expand expands the interval.

    :param data: 待检测的数据列表 / List of values to check
    :param down: 区间下界 / Lower bound of the interval
    :param upper: 区间上界 / Upper bound of the interval
    :param matchtype: "any"（任意满足）或其他（全部满足） / "any" (any match) or other (all match)
    :param strict: True为严格区间，False为非严格 / True for strict (open) interval
    :param tolerance_expand: 区间扩展量 / Tolerance to expand the interval
    :return: bool，是否满足条件 / bool, whether the condition is satisfied
    """
    flag = True
    truly_down = down - tolerance_expand
    truly_upper = upper + tolerance_expand
    # If any data is in the margin, funtion will return true
    if matchtype == "any":
        if strict:
            for ele in data:
                if ele > truly_down and ele < truly_upper:
                    return True
            flag = False
        else:
            for ele in data:
                if ele >= truly_down and ele <= truly_upper:
                    return True
            flag = False
    # If all data is in the margin, funtion will return true
    else:
        if strict:
            for ele in data:
                if not (ele > truly_down and ele < truly_upper):
                    return False
        else:
            for ele in data:
                if not (ele >= truly_down and ele <= truly_upper):
                    return False
    return flag

def calculateIntersectionWithinCrease(c1: Crease, c2: Crease, strict_flag = False):
    """
    计算两条折痕线段的交点（仅返回位于线段范围内的交点）。
    中文：若两折痕存在交点且交点在线段范围内，返回交点坐标；否则返回 None。
    strict_flag=True 时要求交点严格在线段内部（不含端点）。

    Calculate the intersection point of two crease segments (only if within segment bounds).
    English: Returns the intersection coordinate if it exists within both segments; else None.
    strict_flag=True requires the intersection to be strictly inside (excluding endpoints).

    :param c1: 第一条折痕 / First crease
    :param c2: 第二条折痕 / Second crease
    :param strict_flag: True时要求交点严格在线段内 / True requires intersection strictly inside segments
    :return: 交点坐标 [x, y] 或 None / Intersection [x, y] or None
    """
    try:
        k1 = c1.k()
        k2 = c2.k()
        b1 = c1.b()
        b2 = c2.b()
        x11, y11, x21, y21 = c1.getPointAxis()
        x12, y12, x22, y22 = c2.getPointAxis()
        if(k1 != math.inf and k2 != math.inf):
            if(k1 == k2):
                return None
            x0 = (b2 - b1) / (k1 - k2)
            y0 = (k1 * b2 - k2 * b1) / (k1 - k2)
            cross_percent_list = []
            index_1 = [0, -1]
            index_2 = [0, -1]
            if abs(x11 - x21) >= 1e-5:
                cross_percent_list.append((x0 - x11) / (x21 - x11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(y11 - y21) >= 1e-5:
                cross_percent_list.append((y0 - y11) / (y21 - y11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(x12 - x22) >= 1e-5:
                cross_percent_list.append((x0 - x12) / (x22 - x12))
                index_2[END] += 1
            if abs(y12 - y22) >= 1e-5:
                cross_percent_list.append((y0 - y12) / (y22 - y12))
                index_2[END] += 1
            intersection = [x0, y0]
            if strict_flag:
                if limit(cross_percent_list, 0.0, 1.0, matchtype="any", strict=True, tolerance_expand=-1e-5):
                    if limit(cross_percent_list, 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                        return intersection
            else:
                if limit(cross_percent_list, 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                    return intersection
        elif (k1 == math.inf and k2 == math.inf):
            if abs(b1 - b2) < 1e-5:
                if strict_flag:
                    y1 = c1[START][Y]
                    y2 = c1[END][Y]
                    if y1 < y2:
                        y3 = c2[START][Y]
                        y4 = c2[END][Y]
                        if (y3 > y1 and y3 < y2) or (y4 > y1 and y4 < y2):
                            return [b1, 0]
                        else:
                            return None
                    if y2 < y1:
                        y3 = c2[START][Y]
                        y4 = c2[END][Y]
                        if (y3 > y2 and y3 < y1) or (y4 > y2 and y4 < y1):
                            return [b1, 0]
                        else:
                            return None
                else:
                    return [b1, 0]
            else:
                return None
        elif (k1 == math.inf):
            x0 = x11
            y0 = y12 + (x0 - x12) / (x22 - x12) * (y22 - y12)
            cross_percent_list = []
            index_1 = [0, -1]
            index_2 = [0, -1]
            if abs(x11 - x21) >= 1e-5:
                cross_percent_list.append((x0 - x11) / (x21 - x11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(y11 - y21) >= 1e-5:
                cross_percent_list.append((y0 - y11) / (y21 - y11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(x12 - x22) >= 1e-5:
                cross_percent_list.append((x0 - x12) / (x22 - x12))
                index_2[END] += 1
            if abs(y12 - y22) >= 1e-5:
                cross_percent_list.append((y0 - y12) / (y22 - y12))
                index_2[END] += 1
            intersection = [x0, y0]
            if strict_flag:
                if limit(cross_percent_list, 0.0, 1.0, matchtype="any", strict=True, tolerance_expand=-1e-5):
                    if limit(cross_percent_list, 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                        return intersection
            else:
                if limit(cross_percent_list[index_2[START]:index_2[END]], 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                    return intersection
        else:
            x0 = x12
            y0 = y11 + (x0 - x11) / (x21 - x11) * (y21 - y11)
            cross_percent_list = []
            index_1 = [0, -1]
            index_2 = [0, -1]
            if abs(x11 - x21) >= 1e-5:
                cross_percent_list.append((x0 - x11) / (x21 - x11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(y11 - y21) >= 1e-5:
                cross_percent_list.append((y0 - y11) / (y21 - y11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(x12 - x22) >= 1e-5:
                cross_percent_list.append((x0 - x12) / (x22 - x12))
                index_2[END] += 1
            if abs(y12 - y22) >= 1e-5:
                cross_percent_list.append((y0 - y12) / (y22 - y12))
                index_2[END] += 1
            intersection = [x0, y0]
            if strict_flag:
                if limit(cross_percent_list, 0.0, 1.0, matchtype="any", strict=True, tolerance_expand=-1e-5):
                    if limit(cross_percent_list, 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                        return intersection
            else:
                if limit(cross_percent_list[index_1[START]:index_1[END]], 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                    return intersection
        return None
    except:
        return None

def distance(p1, p2):
    """
    计算两个2D点之间的欧氏距离。
    中文：仅使用 x 和 y 分量计算距离。

    Calculate the Euclidean distance between two 2D points.
    English: Uses only x and y components to compute the distance.

    :param p1: 第一个点 [x, y, ...] / First point [x, y, ...]
    :param p2: 第二个点 [x, y, ...] / Second point [x, y, ...]
    :return: 欧氏距离（浮点数） / Euclidean distance (float)
    """
    return math.sqrt((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2)

def distance3D(p1, p2):
    """
    计算两个点之间的3D欧氏距离（若维度不足则退化为2D）。
    中文：若两点均有z分量，计算三维距离；否则退回二维距离。

    Calculate the 3D Euclidean distance between two points (falls back to 2D if no z).
    English: If both points have a z component, computes 3D distance; otherwise 2D.

    :param p1: 第一个点 [x, y] 或 [x, y, z] / First point [x, y] or [x, y, z]
    :param p2: 第二个点 [x, y] 或 [x, y, z] / Second point [x, y] or [x, y, z]
    :return: 欧氏距离（浮点数） / Euclidean distance (float)
    """
    if len(p1) >= 3 and len(p2) >= 3:
        return math.sqrt((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2 + (p1[Z] - p2[Z]) ** 2)
    else:
        return math.sqrt((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2)

def pointToCrease(p, crease: Crease):
    """
    计算点到折痕所在直线的垂直距离。
    中文：使用点到直线公式计算点到折痕直线的最短距离（不限制在线段范围内）。

    Calculate the perpendicular distance from a point to the line containing a crease.
    English: Uses the point-to-line formula to compute the shortest distance
             (not restricted to the segment bounds).

    :param p: 点坐标 [x, y] / Point coordinates [x, y]
    :param crease: 折痕对象 / Crease object
    :return: 点到直线的垂直距离（浮点数） / Perpendicular distance from point to line (float)
    """
    k = crease.k()
    b = crease.b()
    if k == math.inf or k == -math.inf:
        return abs(b - p[X])
    else:
        return abs(k * p[X] - p[Y] + b) / math.sqrt(k ** 2 + 1)
    
def calculateIntersection(kb1, kb2):
    """
    根据两条直线的 (k, b) 参数计算它们的交点。
    中文：通过斜率和截距解方程求两直线交点，若平行则返回 None。

    Calculate the intersection of two lines given their (k, b) parameters.
    English: Solves the linear equations using slope and intercept to find the intersection;
             returns None if lines are parallel.

    :param kb1: 第一条直线的 [k, b] / [slope, intercept] of the first line
    :param kb2: 第二条直线的 [k, b] / [slope, intercept] of the second line
    :return: 交点坐标 [x, y] 或 None（平行）/ Intersection [x, y] or None (parallel)
    """
    if kb1[0] == kb2[0] or (kb1[0] == math.inf and kb2[0] == math.inf):
        return None
    elif kb1[0] == math.inf:
        return [
            kb1[1],
            kb2[0] * kb1[1] + kb2[1]
        ]
    elif kb2[0] == math.inf:
        return [
            kb2[1],
            kb1[0] * kb2[1] + kb1[1]
        ]
    else:
        return [
            (kb1[1] - kb2[1]) / (kb2[0] - kb1[0]),
            (kb1[1] * kb2[0] - kb2[1] * kb1[0]) / (kb2[0] - kb1[0]),
        ]

def norm(vec: list):
    """
    计算向量的欧氏范数（模长）。
    中文：对向量各分量的平方和开方。

    Calculate the Euclidean norm (magnitude) of a vector.
    English: Square root of the sum of squares of all components.

    :param vec: 向量（列表） / Vector (list)
    :return: 向量的模长（浮点数） / Magnitude of the vector (float)
    """
    norm = 0
    for ele in vec:
        norm += ele ** 2
    return math.sqrt(norm)

def R(theta):
    """
    生成2D旋转矩阵。
    中文：返回绕原点逆时针旋转 theta 弧度的2x2旋转矩阵。

    Generate a 2D rotation matrix.
    English: Returns a 2x2 rotation matrix for counter-clockwise rotation by theta radians.

    :param theta: 旋转角度（弧度） / Rotation angle (radians)
    :return: 2x2 旋转矩阵（numpy array） / 2x2 rotation matrix (numpy array)
    """
    return np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])

def calculateCenterPoint(points):
    """
    计算2D点集的几何中心（重心）。
    中文：对所有点的x、y坐标求平均值。

    Calculate the geometric centroid of a set of 2D points.
    English: Average the x and y coordinates of all points.

    :param points: 点列表，每个点为 [x, y, ...] / List of points, each as [x, y, ...]
    :return: 中心点坐标 [x, y] / Centroid coordinates [x, y]
    """
    x = 0.0
    y = 0.0
    for p in points:
        x += p[X]
        y += p[Y]
    kp_len = len(points)
    x /= kp_len
    y /= kp_len
    return [x, y]

def calculateCenterPoint3D(points):
    """
    计算3D点集的几何中心（重心）。
    中文：对所有点的x、y、z坐标求平均值。

    Calculate the geometric centroid of a set of 3D points.
    English: Average the x, y, and z coordinates of all points.

    :param points: 点列表，每个点为 [x, y, z] / List of points, each as [x, y, z]
    :return: 中心点坐标 [x, y, z] / Centroid coordinates [x, y, z]
    """
    x = 0.0
    y = 0.0
    z = 0.0
    for p in points:
        x += p[X]
        y += p[Y]
        z += p[Z]
    kp_len = len(points)
    x /= kp_len
    y /= kp_len
    z /= kp_len
    return [x, y, z]

def pointOnCrease(point, c: Crease, tolerance=1e-5):
    start = c[START]
    end = c[END]
    k = c.k()
    if abs(k) > 1e5:
        percent_x = 0.5
        percent_y = (point[Y] - start[Y]) / (end[Y] - start[Y])
    else:
        percent_x = (point[X] - start[X]) / (end[X] - start[X])
        if abs(k) < 1e-5:
            percent_y = 0.5
        else:
            percent_y = (point[Y] - start[Y]) / (end[Y] - start[Y])
    if pointToCrease(point, c) < tolerance and (percent_limit(percent_x) == percent_x and percent_limit(percent_y) == percent_y):
        return True
    else:
        return False
    
    start = c[START]
    end = c[END]
    if abs(end[X] - start[X]) >= 1e-5 and abs(end[Y] - start[Y]) >= 1e-5:
        p_x = (point[X] - start[X]) / (end[X] - start[X])
        p_y = (point[Y] - start[Y]) / (end[Y] - start[Y])
        if abs(p_x - p_y) < tolerance and p_x >= 0.0 and p_x <= 1.0:
            return True
        else:
            return False
    else:
        tolerance *= 10
        if abs(end[X] - start[X]) < tolerance:
            p_y = (point[Y] - start[Y]) / (end[Y] - start[Y])
            if abs(point[X] - start[X]) < tolerance and p_y >= 0.0 and p_y <= 1.0:
                return True
            else:
                return False
        else:
            p_x = (point[X] - start[X]) / (end[X] - start[X])
            if abs(point[Y] - start[Y]) < tolerance and p_x >= 0.0 and p_x <= 1.0:
                return True
            else:
                return False

def packTrajectory(trajectory, P_number):
    """
    将穿绳轨迹打包为字典格式，记录类型、编号和方向信息。
    中文：
        将轨迹中的每个点分类为 "A"（边界点）或 "B"（折纸点），
        并推断绳子在每段的穿绕方向（reverse字段）。
        P_number 是 A 类点的数量，大于此值的索引归为 B 类。

    Pack the threading trajectory into a dictionary with type, id, and direction info.
    English:
        Classifies each point in the trajectory as "A" (border point) or "B" (origami point),
        and infers the threading direction for each segment (reverse field).
        P_number is the count of type-A points; indices >= P_number are type-B.

    :param trajectory: 轨迹列表，每项为 [index, side] / Trajectory list, each item is [index, side]
    :param P_number: A类点的数量阈值 / Count threshold for type-A points
    :return: 包含type、id、reverse三个键的字典 / Dict with keys: type, id, reverse
    """
    dict = {
        "type": [],
        "id": [],
        "reverse": []
    }
    for i in range(len(trajectory)):
        if trajectory[i][0] < P_number:
            dict["type"].append("A")
            dict["id"].append(trajectory[i][0])
        else:
            dict["type"].append("B")
            dict["id"].append(trajectory[i][0] - P_number)
    exist_non_zero_side = 0
    for i in range(len(trajectory)):
        if trajectory[i][1] != 0:
            exist_non_zero_side = trajectory[i][1]
            break
    if exist_non_zero_side:
        for j in range(len(trajectory)):
            if (j - i) % 2 == 0:
                dict["reverse"].append(trajectory[i][1])
            else:
                dict["reverse"].append(-trajectory[i][1])
        dict["reverse"][0] = dict["reverse"][1]
    else:
        for j in range(len(trajectory)):
            if j % 2 == 0:
                dict["reverse"].append(-1)
            else:
                dict["reverse"].append(1)
        dict["reverse"][0] = 1
    return dict

def point_in_segment(point, seg_start, seg_end):
    """
    判断点是否在某条线段上（边界情况处理）
    :param point: 目标点 (x, y)
    :param seg_start: 线段起点 (x, y)
    :param seg_end: 线段终点 (x, y)
    :return: bool，点是否在线段上
    """
    # 1. 点在线段的包围盒内（x、y范围）
    min_x = min(seg_start[X], seg_end[X])
    max_x = max(seg_start[X], seg_end[X])
    min_y = min(seg_start[Y], seg_end[Y])
    max_y = max(seg_start[Y], seg_end[Y])
    if not (min_x - 1e-8 <= point[X] <= max_x + 1e-8 and min_y - 1e-8 <= point[Y] <= max_y + 1e-8):
        return False
    
    # 2. 向量叉积为0（点与线段共线）
    vec1 = [seg_start[X] - point[X], seg_start[Y] - point[Y]]
    vec2 = [seg_end[X] - point[X], seg_end[Y] - point[Y]]
    cross = vec1[X] * vec2[Y] - vec1[Y] * vec2[X]
    if abs(cross) > 1e-8:
        return False
    
    return True

def distance_point_to_segment(point, seg_start, seg_end):
    """
    计算点到线段的最短距离（核心：投影法）
    :param point: 目标点 (x, y)
    :param seg_start: 线段起点 (x, y)
    :param seg_end: 线段终点 (x, y)
    :return: 点到线段的最短距离
    """
    # 向量定义
    vec_seg = [seg_end[X] - seg_start[X], seg_end[Y] - seg_start[Y]]  # 线段向量
    vec_point = [point[X] - seg_start[X], point[Y] - seg_start[Y]]     # 起点到点的向量
    
    # 线段长度的平方（避免开方，提升效率）
    seg_len_sq = vec_seg[X] ** 2 + vec_seg[Y] ** 2
    if seg_len_sq < 1e-16:  # 线段退化为点
        return math.sqrt(vec_point[X] ** 2 + vec_point[Y] ** 2)
    
    # 计算点在seg上的投影系数t（t∈[0,1]表示投影在线段内）
    t = max(0.0, min(1.0, (vec_point[X] * vec_seg[X] + vec_point[Y] * vec_seg[Y]) / seg_len_sq))
    
    # 投影点坐标
    proj_x = seg_start[X] + t * vec_seg[X]
    proj_y = seg_start[Y] + t * vec_seg[Y]
    
    # 点到投影点的距离（最短距离）
    dx = point[X] - proj_x
    dy = point[Y] - proj_y
    return math.sqrt(dx ** 2 + dy ** 2)

def polygon_area(points):
    """计算多边形面积（判断顶点顺序：正=逆时针，负=顺时针）"""
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    return area / 2.0

def normalize_polygon_order(points):
    """统一多边形顶点为逆时针顺序"""
    area = polygon_area(points)
    if area < 0:  # 顺时针转为逆时针
        return points[::-1]
    return points.copy()

def edge_inner_normal(edge_start, edge_end, is_counter_clockwise=True):
    """
    计算边的向内单位法向量（核心：适配凸/凹多边形）
    :param edge_start: 边起点 (x,y)
    :param edge_end: 边终点 (x,y)
    :param is_counter_clockwise: 多边形是否逆时针
    :return: 向内法向量 (nx, ny)
    """
    dx = edge_end[0] - edge_start[0]
    dy = edge_end[1] - edge_start[1]
    # 边的方向向量的垂直向量（两个方向）
    normal1 = (-dy, dx)  # 左法向量（逆时针多边形的向内方向）
    normal2 = (dy, -dx)  # 右法向量
    # 归一化
    norm = math.hypot(normal1[0], normal1[1])
    if norm < 1e-8:
        return (0.0, 0.0)
    normal1 = (normal1[0]/norm, normal1[1]/norm)
    normal2 = (normal2[0]/norm, normal2[1]/norm)
    # 确定向内方向
    if is_counter_clockwise:
        return normal1
    else:
        return normal2
    
def translate_edge(edge_start, edge_end, distance, normal):
    """沿法向量平移边，返回平移后的起点和终点"""
    tx = normal[0] * distance
    ty = normal[1] * distance
    new_start = (edge_start[0] + tx, edge_start[1] + ty)
    new_end = (edge_end[0] + tx, edge_end[1] + ty)
    return new_start, new_end

def line_intersection(p1, p2, p3, p4):
    """
    计算两条直线（p1-p2, p3-p4）的交点
    :return: 交点 (x,y) 或 None（平行）
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None  # 平行无交点

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    t = t_num / denom
    u = u_num / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (x, y)

def segment_intersection(p1, p2, p3, p4):
    """
    计算两条线段的交点（仅返回线段上的交点）
    :return: 交点 (x,y) 或 None
    """
    intersect = line_intersection(p1, p2, p3, p4)
    if intersect is None:
        return None

    # 判断交点是否在两条线段上
    def is_between(a, b, c):
        return min(a, b) - 1e-2 <= c <= max(a, b) + 1e-2

    x, y = intersect
    if (is_between(p1[0], p2[0], x) and is_between(p1[1], p2[1], y) and
        is_between(p3[0], p4[0], x) and is_between(p3[1], p4[1], y)):
        return intersect
    return None

def detect_self_intersection(points, offset=0):
    """
    检测多边形是否自相交，返回自相交的边对和交点
    :param points: 多边形顶点列表 [(x1,y1), ...]
    :return: list of (intersection_point, edge1_idx, edge2_idx)
    """
    intersections = []
    n = len(points)
    for i in range(offset, n + offset):
        p1 = points[i % n]
        p2 = points[(i + 1) % n]
        for j in range(i + 2, n + offset):
            if (j + 1) % n == i:  # 相邻边跳过
                continue
            p3 = points[j % n]
            p4 = points[(j + 1) % n]
            intersect = segment_intersection(p1, p2, p3, p4)
            if intersect is not None:
                if distance(p1, intersect) > distance(p2, intersect):
                    intersections.append((intersect, i % n, j % n))
                else:
                    intersections.append((intersect, j % n, i % n))
    return intersections
    
def pointInPolygon(point, polygon: list, return_min_distance=False, upper_x_bound=math.inf, lower_x_bound=-math.inf):
    """
    判断2D点是否在多边形内（适配凸/凹多边形，边不交叉）
    :param point: 目标点 (x, y)
    :param polygon: 多边形顶点列表，按顺序存储 [(x1,y1), (x2,y2), ..., (xn,yn)]
    :param return_min_distance: 若点在内部，是否返回点到多边形的最短距离（否则返回bool）
    :param upper_x_bound/lower_x_bound: 原逻辑保留的x边界（距离计算时参与比较）
    :return: bool（是否在内部） 或 float（最短距离，仅return_min_distance=True且点在内部时）
    """
    # 边界1：空多边形直接返回False
    if len(polygon) < 3:
        return False if not return_min_distance else 0.0
    
    inside = False
    polygon_len = len(polygon)
    
    # ========== 核心：射线法判断点是否在多边形内 ==========
    for i in range(polygon_len):
        # 取当前边的起点和终点（闭合多边形，最后一个点连回第一个点）
        p1 = polygon[i]
        p2 = polygon[(i + 1) % polygon_len]
        
        # 先判断点是否在当前边上（边界情况，直接视为在内部）
        if point_in_segment(point, p1, p2):
            if return_min_distance:
                return 0.0  # 点在边上，距离为0
            else:
                return True
        
        # 射线法核心逻辑：判断水平射线（向右）是否与当前边相交
        # 1. 过滤掉不相交的边（y坐标不在点的y区间内）
        if ((p1[Y] > point[Y]) != (p2[Y] > point[Y])):
            # 2. 计算射线与边的交点x坐标
            x_intersect = ( (point[Y] - p1[Y]) * (p2[X] - p1[X]) ) / (p2[Y] - p1[Y]) + p1[X]
            # 3. 交点在射线右侧（x > 点的x），则计数+1，翻转inside状态
            if point[X] < x_intersect - 1e-8:  # 1e-8避免浮点精度问题
                inside = not inside
    
    # ========== 距离计算逻辑 ==========
    if inside and return_min_distance:
        # 计算点到所有边的最短距离
        min_dist = math.inf
        for i in range(polygon_len):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % polygon_len]
            dist = distance_point_to_segment(point, p1, p2)
            if dist < min_dist:
                min_dist = dist
        # 原逻辑保留：与x边界的距离比较（若有需要）
        delta_x_lower = point[X] - lower_x_bound
        delta_x_upper = upper_x_bound - point[X]
        min_dist = min(min_dist, delta_x_lower, delta_x_upper)
        return min_dist
    else:
        # 不在内部 或 不需要返回距离：返回布尔值
        return inside

def is_point_equal(p1, p2):
    """判断两个点（x,y,z）是否相等（忽略z，仅比较x,y）"""
    return abs(p1[0] - p2[0]) < 1e-2 and abs(p1[1] - p2[1]) < 1e-2

def cross_product(p1, p2, p3):
    """计算向量 (p2-p1) 和 (p3-p1) 的叉积（仅x,y）"""
    vec1_x = p2[0] - p1[0]
    vec1_y = p2[1] - p1[1]
    vec2_x = p3[0] - p1[0]
    vec2_y = p3[1] - p1[1]
    return vec1_x * vec2_y - vec1_y * vec2_x

def is_collinear(p1, p2, p3):
    """判断三个点是否共线（仅x,y）"""
    return abs(cross_product(p1, p2, p3)) < 1e-2

def is_valid_triangle(tri):
    """
    判断是否为有效三角形：
    1. 包含且仅包含3个点
    2. 3个点不全重合
    3. 3个点不共线
    """
    # 条件1：必须是3个点
    if len(tri) != 3:
        return False
    
    p1, p2, p3 = tri
    # 条件2：3个点不全重合
    if is_point_equal(p1, p2) and is_point_equal(p2, p3):
        return False
    
    # 条件3：3个点不共线
    if is_collinear(p1, p2, p3):
        return False
    
    return True

def segment_intersection_bool(s1_p1, s1_p2, s2_p1, s2_p2):
    """
    判断两条线段是否相交（仅x,y）
    :param s1_p1/s1_p2: 第一条线段的两个端点
    :param s2_p1/s2_p2: 第二条线段的两个端点
    :return: True=相交，False=不相交
    """
    # 计算四个点的叉积，判断线段相对位置
    def ccw(a, b, c):
        return cross_product(a, b, c) > 1e-2
    
    def ccw_eps(a, b, c):
        return cross_product(a, b, c) > -1e-2

    A, B = s1_p1, s1_p2
    C, D = s2_p1, s2_p2

    # 快速排斥实验（先判断包围盒是否相交）
    def rect_overlap(a1, a2, b1, b2):
        min_x1 = min(a1[0], a2[0])
        max_x1 = max(a1[0], a2[0])
        min_y1 = min(a1[1], a2[1])
        max_y1 = max(a1[1], a2[1])
        
        min_x2 = min(b1[0], b2[0])
        max_x2 = max(b1[0], b2[0])
        min_y2 = min(b1[1], b2[1])
        max_y2 = max(b1[1], b2[1])
        
        return not (max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1)

    if not rect_overlap(A, B, C, D):
        return False

    # 跨立实验
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D)) or \
           (abs(cross_product(A, B, C)) < 1e-2 and rect_overlap(A, B, A, C)) or \
           (abs(cross_product(A, B, D)) < 1e-2 and rect_overlap(A, B, A, D)) or \
           (abs(cross_product(C, D, A)) < 1e-2 and rect_overlap(C, D, C, A)) or \
           (abs(cross_product(C, D, B)) < 1e-2 and rect_overlap(C, D, C, B))

def point_in_triangle(point, tri):
    """
    判断点是否在三角形内（包括边）
    :param point: 待判断点 [x,y,z]
    :param tri: 三角形 [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
    :return: True=在内部/边上，False=外部
    """
    p1, p2, p3 = tri
    # 计算点与三角形三边的叉积
    cp1 = cross_product(p1, p2, point)
    cp2 = cross_product(p2, p3, point)
    cp3 = cross_product(p3, p1, point)
    
    # 所有叉积同号（都正/都负），或其中一个为0（在边上）
    is_all_pos = (cp1 >= -1e-2) and (cp2 >= -1e-2) and (cp3 >= -1e-2)
    is_all_neg = (cp1 <= 1e-2) and (cp2 <= 1e-2) and (cp3 <= 1e-2)
    return is_all_pos or is_all_neg

def triangles_intersect(tri1, tri2):
    """
    判断两个有效三角形是否相交
    :param tri1/tri2: 有效三角形（3个点）
    :return: True=相交，False=不相交
    """
    # 提取两个三角形的所有边（每条边是两个端点）
    tri1_edges = [
        (tri1[0], tri1[1]),
        (tri1[1], tri1[2]),
        (tri1[2], tri1[0])
    ]
    tri2_edges = [
        (tri2[0], tri2[1]),
        (tri2[1], tri2[2]),
        (tri2[2], tri2[0])
    ]
    
    # 1. 判断边是否相交
    for e1 in tri1_edges:
        for e2 in tri2_edges:
            if segment_intersection_bool(e1[0], e1[1], e2[0], e2[1]):
                return True
    
    # 2. 判断一个三角形的顶点是否在另一个三角形内
    for p in tri1:
        if point_in_triangle(p, tri2):
            return True
    for p in tri2:
        if point_in_triangle(p, tri1):
            return True
    
    return False

def _triangle_aabb_overlap(tri1, tri2, eps=1e-9):
    for axis in range(3):
        mins1 = min(p[axis] for p in tri1)
        maxs1 = max(p[axis] for p in tri1)
        mins2 = min(p[axis] for p in tri2)
        maxs2 = max(p[axis] for p in tri2)
        if maxs1 < mins2 - eps or maxs2 < mins1 - eps:
            return False
    return True

def _project_tri_to_2d(tri, drop_axis):
    a = (drop_axis + 1) % 3
    b = (drop_axis + 2) % 3
    return [[p[a], p[b]] for p in tri]

def _triangles_coplanar(tri1, tri2, eps=1e-4):
    a = np.array(tri1[0], dtype=float)
    b = np.array(tri1[1], dtype=float)
    c = np.array(tri1[2], dtype=float)
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return True
    n = n / norm
    for p in tri2:
        if abs(np.dot(np.array(p, dtype=float) - a, n)) > eps:
            return False
    return True

def _point_in_triangle_3d(point, tri, eps=1e-4):
    a = np.array(tri[0], dtype=float)
    b = np.array(tri[1], dtype=float)
    c = np.array(tri[2], dtype=float)
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return False
    n = n / norm
    p = np.array(point, dtype=float)
    dist = np.dot(p - a, n)
    if abs(dist) > eps:
        return False
    p_proj = p - dist * n
    axis = int(np.argmax(np.abs(n)))
    p2d = [p_proj[(axis + 1) % 3], p_proj[(axis + 2) % 3]]
    t2d = _project_tri_to_2d(tri, axis)
    return point_in_triangle(p2d, t2d)

def _segment_triangle_intersect_3d(seg_start, seg_end, tri, eps=1e-4):
    return _segment_triangle_intersect_point_3d(seg_start, seg_end, tri, eps) is not None

def _segment_triangle_intersect_point_3d(seg_start, seg_end, tri, eps=1e-4):
    """Return 3D hit point if segment pierces triangle, else None."""
    a = np.array(tri[0], dtype=float)
    b = np.array(tri[1], dtype=float)
    c = np.array(tri[2], dtype=float)
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    n = n / norm
    s0 = np.array(seg_start, dtype=float)
    s1 = np.array(seg_end, dtype=float)
    d = s1 - s0
    denom = np.dot(n, d)
    if abs(denom) < eps:
        return None
    t = np.dot(n, a - s0) / denom
    if t < -eps or t > 1.0 + eps:
        return None
    p = s0 + t * d
    if not _point_in_triangle_3d(p.tolist(), tri, eps=eps * 10):
        return None
    return p.tolist()

def _dedupe_points_3d(points, eps=1e-5):
    unique = []
    for p in points:
        pa = np.asarray(p, dtype=float)
        if any(np.linalg.norm(pa - np.asarray(q, dtype=float)) <= eps for q in unique):
            continue
        unique.append(pa.tolist())
    return unique

def triangle_intersection_contacts_3d(tri1, tri2, eps=1e-4):
    """
    Contact geometry for two non-coplanar triangles.
    Returns a list of unique 3D points on the intersection
    (typically 1–2 points = contact point or contact segment endpoints).
    Empty list if no intersection or coplanar (caller may skip coplanar).
    """
    if not _triangle_aabb_overlap(tri1, tri2, eps):
        return []
    if _triangles_coplanar(tri1, tri2, eps):
        return []

    pts = []
    edges1 = [(tri1[0], tri1[1]), (tri1[1], tri1[2]), (tri1[2], tri1[0])]
    edges2 = [(tri2[0], tri2[1]), (tri2[1], tri2[2]), (tri2[2], tri2[0])]
    for e in edges1:
        hit = _segment_triangle_intersect_point_3d(e[0], e[1], tri2, eps)
        if hit is not None:
            pts.append(hit)
    for e in edges2:
        hit = _segment_triangle_intersect_point_3d(e[0], e[1], tri1, eps)
        if hit is not None:
            pts.append(hit)
    for p in tri1:
        if _point_in_triangle_3d(p, tri2, eps):
            pts.append(list(p))
    for p in tri2:
        if _point_in_triangle_3d(p, tri1, eps):
            pts.append(list(p))
    return _dedupe_points_3d(pts, eps=max(eps * 10, 1e-6))

def triangles_intersect_3d(tri1, tri2, eps=1e-4):
    """
    Determine whether two 3D triangles intersect (including edge/vertex contact).
    """
    if not _triangle_aabb_overlap(tri1, tri2, eps):
        return False

    if _triangles_coplanar(tri1, tri2, eps):
        a = np.array(tri1[0], dtype=float)
        b = np.array(tri1[1], dtype=float)
        c = np.array(tri1[2], dtype=float)
        n = np.cross(b - a, c - a)
        axis = int(np.argmax(np.abs(n)))
        return triangles_intersect(_project_tri_to_2d(tri1, axis), _project_tri_to_2d(tri2, axis))

    return len(triangle_intersection_contacts_3d(tri1, tri2, eps)) > 0

def process_triangles(input_triangles):
    """
    处理三角形列表：
    1. 过滤无效三角形
    2. 删除所有相交的有效三角形
    :param input_triangles: 单个三角形 [p1,p2,p3] 或多个三角形 [[p1,p2,p3], ...]
    :return: 剩余的不相交有效三角形列表
    """
    # ========== 步骤1：统一输入格式为列表的列表 ==========
    if isinstance(input_triangles[0], list) and len(input_triangles[0]) == 3:
        # 输入是多个三角形
        raw_triangles = input_triangles.copy()
    else:
        # 输入是单个三角形
        raw_triangles = [input_triangles.copy()]
    
    # ========== 步骤2：过滤无效三角形 ==========
    valid_triangles = []
    for tri in raw_triangles:
        if is_valid_triangle(tri):
            valid_triangles.append(tri)
    if len(valid_triangles) <= 1:
        return valid_triangles  # 只有0/1个有效三角形，无相交可能
    
    # ========== 步骤3：找出所有相交的三角形 ==========
    intersect_indices = set()  # 存储所有相交的三角形索引
    n = len(valid_triangles)
    
    # 遍历所有三角形对（i < j，避免重复判断）
    for i in range(n):
        if i in intersect_indices:
            continue  # 已标记为相交，跳过
        for j in range(i+1, n):
            if triangles_intersect(valid_triangles[i], valid_triangles[j]):
                intersect_indices.add(i)
                intersect_indices.add(j)
    
    # ========== 步骤4：删除相交的三角形，返回剩余 ==========
    result = []
    for idx, tri in enumerate(valid_triangles):
        if idx not in intersect_indices:
            result.append(tri)
    
    return result

def is_valid_polygon(poly):
    """
    判断是否为正常多边形（简单多边形）：
    1. 至少包含3个点
    2. 所有点不全重合
    3. 相邻边不共线
    4. 边不自相交
    """
    n = len(poly)
    # 条件1：至少3个点
    if n < 3:
        return False
    
    # 条件2：所有点不全重合
    all_same = True
    first_p = poly[0]
    for p in poly[1:]:
        if not is_point_equal(first_p, p):
            all_same = False
            break
    if all_same:
        return False
    
    # 条件3：相邻边不共线
    i = 0
    while i < n:
        p1 = poly[i]
        p2 = poly[(i+1)%n]
        p3 = poly[(i+2)%n]
        if is_collinear(p1, p2, p3):
            del(poly[(i+1)%n])
            n -= 1
        else:
            i += 1
        
    if n < 3:
        return False
    
    # 条件4：边不自相交（仅检查非相邻边）
    for i in range(n):
        # 当前边：i -> i+1
        e1_p1 = poly[i]
        e1_p2 = poly[(i+1)%n]
        for j in range(i+2, n):
            # 跳过相邻边
            if (j+1)%n == i:
                continue
            # 另一条边：j -> j+1
            e2_p1 = poly[j]
            e2_p2 = poly[(j+1)%n]
            if segment_intersection_bool(e1_p1, e1_p2, e2_p1, e2_p2):
                return False
    
    return poly

def generatePolygonByCenter(center, size, resolution):
    """
    以指定中心点生成正多边形的顶点列表（3D，z由center[2]决定）。
    中文：按指定分辨率（边数）和大小（半径），生成圆内接多边形的顶点坐标列表。

    Generate a regular polygon's vertex list centered at a given point (3D, z from center[2]).
    English: Generates vertices of a regular polygon inscribed in a circle
             with given center, radius (size), and number of sides (resolution).

    :param center: 中心点坐标 [x, y, z] / Center point [x, y, z]
    :param size: 外接圆半径 / Circumscribed circle radius
    :param resolution: 多边形边数（即顶点数） / Number of sides/vertices
    :return: 顶点坐标列表 [[x, y, z], ...] / List of vertex coordinates [[x, y, z], ...]
    """
    points = []
    step = math.pi * 2 / resolution
    for i in range(0, resolution):
        points.append(
            [
                center[0] + math.cos(i * step) * size, 
                center[1] + math.sin(i * step) * size,
                center[2]
            ]
        )
    return points

def calculateAngle(kp, kp_child1, kp_child2):
    """
    计算以 kp 为顶点，kp_child1 和 kp_child2 为两侧点构成的夹角（弧度）。
    中文：用点积公式计算两个向量间的夹角，返回范围为 [0, π]。

    Calculate the angle (radians) at vertex kp, formed by vectors to kp_child1 and kp_child2.
    English: Uses the dot product formula to compute the angle between two vectors,
             result in [0, π].

    :param kp: 顶点坐标 [x, y] / Vertex coordinates [x, y]
    :param kp_child1: 第一侧点坐标 [x, y] / First side point coordinates [x, y]
    :param kp_child2: 第二侧点坐标 [x, y] / Second side point coordinates [x, y]
    :return: 夹角（弧度） / Angle (radians)
    """
    vec1 = [kp_child1[0] - kp[0], kp_child1[1] - kp[1]]
    vec2 = [kp_child2[0] - kp[0], kp_child2[1] - kp[1]]
    len1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    len2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
    dot_multiple_result = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    cos_result = dot_multiple_result / (len1 * len2)
    return math.acos(np.clip(cos_result, -1.0, 1.0))

def samePoint(p1, p2, dim, tol=1e-2):
    """
    判断两个点是否在容差范围内相同（基于2D距离）。
    中文：若两点之间的欧氏距离小于容差 tol，则认为相同。

    Check if two points are the same within a tolerance (based on 2D distance).
    English: Returns True if the Euclidean distance between p1 and p2 is less than tol.

    :param p1: 第一个点 / First point
    :param p2: 第二个点 / Second point
    :param dim: 维度（未使用，保留参数） / Dimension (unused, reserved parameter)
    :param tol: 容差 / Tolerance
    :return: bool，是否相同 / bool, whether the points are the same
    """
    if distance(p1, p2) > tol:
        return False
    return True

def sameCrease(c1: Crease, c2: Crease, tol=1e-2):
    """
    判断两条折痕在容差范围内是否完全重合（起点和终点均相同）。
    中文：逐坐标比较起点和终点，均在容差内则视为相同折痕。

    Check if two creases coincide within tolerance (same start and end points).
    English: Compares start and end coordinates component-wise; returns True if all within tol.

    :param c1: 第一条折痕 / First crease
    :param c2: 第二条折痕 / Second crease
    :param tol: 容差 / Tolerance
    :return: bool，是否重合 / bool, whether the creases coincide
    """
    if abs(c1[END][Y] - c2[END][Y]) < tol and abs(c1[START][X] - c2[START][X]) < tol \
          and abs(c1[END][X] - c2[END][X]) < tol and abs(c1[START][Y] - c2[START][Y]) < tol:
        return True
    else:
        return False

def crossProduct(c1, c2):
    """
    计算两条折痕方向向量的2D叉积。
    中文：将c1和c2的方向向量（从起点到终点）做叉积运算，结果为标量。

    Compute the 2D cross product of the direction vectors of two creases.
    English: Cross-multiplies the direction vectors (from start to end) of c1 and c2;
             result is a scalar.

    :param c1: 第一条折痕 / First crease
    :param c2: 第二条折痕 / Second crease
    :return: 叉积标量（v1.x*v2.y - v1.y*v2.x） / Cross product scalar (v1.x*v2.y - v1.y*v2.x)
    """
    v1 = [c1[END][X] - c1[START][X], c1[END][Y] - c1[START][Y]] 
    v2 = [c2[END][X] - c2[START][X], c2[END][Y] - c2[START][Y]]
    return v1[X] * v2[Y] - v1[Y] * v2[X]

def angleBetweenCreases(c1, c2, from_same_start=True):
    """
    计算两条折痕之间的有向夹角（弧度）。
    中文：
        若 from_same_start=True，自动调整使两条折痕从同一方向出发（起点共享），
        再用叉积确定正负，返回有符号角度（正=逆时针，负=顺时针）。

    Calculate the directed angle between two creases (radians).
    English:
        If from_same_start=True, automatically adjusts so both creases share a common start direction,
        then uses cross product to determine sign; returns signed angle (positive=CCW, negative=CW).

    :param c1: 第一条折痕 / First crease
    :param c2: 第二条折痕 / Second crease
    :param from_same_start: True时自动对齐起点方向 / True to auto-align start directions
    :return: 有向夹角（弧度） / Signed angle (radians)
    """
    if from_same_start:
        if distance(c1[END], c2[START]) < 1e-5:
            c1 = c1.getReverse()
        elif distance(c1[START], c2[END]) < 1e-5:
            c2 = c2.getReverse()
    v1 = [c1[END][X] - c1[START][X], c1[END][Y] - c1[START][Y]] 
    v2 = [c2[END][X] - c2[START][X], c2[END][Y] - c2[START][Y]]
    sign = v1[X] * v2[Y] - v1[Y] * v2[X]
    angle = (v1[X] * v2[X] + v1[Y] * v2[Y]) / (norm(v1) * norm(v2))
    if angle > 1.0:
        angle = 1.0
    elif angle < -1.0:
        angle = -1.0
    if sign >= 0:
        return math.acos(angle)
    else:
        return -math.acos(angle)

def methodToTotalInformation(method, P_points, O_points):
    """
    将穿绳方法字典转换为包含 TSAPoint 对象的完整信息列表。
    中文：
        遍历 method 字典中的每个字符串序列，为每个穿绳路径点创建 TSAPoint 对象，
        并根据类型（"A"或"B"）分配坐标（分别来自 P_points 或 O_points）。

    Convert a threading method dictionary into a full list of TSAPoint objects.
    English:
        Iterates over each string sequence in the method dict, creates TSAPoint objects
        for each threading path point, and assigns coordinates based on type
        ("A" from P_points, "B" from O_points).

    :param method: 穿绳方法字典（含type、id、reverse键） / Threading method dict (keys: type, id, reverse)
    :param P_points: A类点（边界点）坐标列表 / Type-A (border) point coordinate list
    :param O_points: B类点（折纸点）坐标列表 / Type-B (origami) point coordinate list
    :return: TSAPoint对象的嵌套列表 / Nested list of TSAPoint objects
    """
    total_information = []
    for i in range(len(method["type"])):
        point_list = []
        for j in range(len(method["type"][i])):
            tsa_point = TSAPoint()
            tsa_point.point_type = method["type"][i][j]
            tsa_point.id = method["id"][i][j]
            tsa_point.dir = method["reverse"][i][j]
            if tsa_point.point_type == 'A':
                tsa_point.point = P_points[tsa_point.id]
            else:
                tsa_point.point = O_points[tsa_point.id]
            point_list.append(tsa_point)
        total_information.append(point_list)
    return total_information

def getMaxDistance(kps):
    """
    计算点集的最大包围尺寸及x、y方向的跨度。
    中文：返回点集在x和y方向的最大差值中的较大者，以及x方向和y方向的单独跨度。

    Calculate the maximum bounding size and span in x/y of a point set.
    English: Returns the larger of x-span and y-span, along with the individual x-span and y-span.

    :param kps: 点坐标列表 / List of point coordinates
    :return: (最大跨度, x跨度, y跨度) / (max_span, x_span, y_span)
    """
    max_x = max([kps[i][X] for i in range(len(kps))])
    min_x = min([kps[i][X] for i in range(len(kps))])
    max_y = max([kps[i][Y] for i in range(len(kps))])
    min_y = min([kps[i][Y] for i in range(len(kps))])
    return max(max_x - min_x, max_y - min_y), max_x - min_x, max_y - min_y
    
def getTotalBias(units):
    """
    计算所有折纸单元顶点的几何重心（偏移量）。
    中文：遍历所有单元的顶点坐标，求所有点的x、y均值，用于图形居中等操作。

    Calculate the geometric centroid (bias) of all vertices across all origami units.
    English: Iterates over all unit vertices and computes the mean x/y, used for centering graphics.

    :param units: 折纸单元列表（每个单元实现了 getSeqPoint() 方法） / List of origami units (each has getSeqPoint())
    :return: 重心坐标 [mean_x, mean_y] / Centroid coordinates [mean_x, mean_y]
    """
    total_x = 0.0
    total_y = 0.0
    count = 0
    for unit in units:
        seq_points = unit.getSeqPoint()
        for p in seq_points:
            total_x += p[X]
            total_y += p[Y]
            count += 1
    return [total_x / count, total_y / count]

def coplanar(v1_start, v1_end, v2_start, v2_end):
    """
    判断两条3D线段是否共面（即在同一平面内）。
    中文：通过混合积（三重积）判断两向量是否共面，
    若混合积接近0则共面。

    Check whether two 3D line segments are coplanar.
    English: Uses the scalar triple product to determine coplanarity;
             coplanar if the product is close to zero.

    :param v1_start: 第一条线段的起点 [x, y, z] / Start of first segment [x, y, z]
    :param v1_end: 第一条线段的终点 [x, y, z] / End of first segment [x, y, z]
    :param v2_start: 第二条线段的起点 [x, y, z] / Start of second segment [x, y, z]
    :param v2_end: 第二条线段的终点 [x, y, z] / End of second segment [x, y, z]
    :return: bool，是否共面 / bool, whether the segments are coplanar
    """
    x1 = np.array(v1_start)
    x2 = np.array(v1_end)
    y1 = np.array(v2_start)
    y2 = np.array(v2_end)
    v1 = x2 - x1
    v2 = y2 - y1
    v1_start_to_v2_end = y2 - x1
    n = np.cross(v1, v1_start_to_v2_end)
    if abs(np.dot(n, v2)) < 1e-4:
        return True
    else:
        return False

def rapidRepel(v1_start, v1_end, v2_start, v2_end):
    """
    快速排斥测试：判断两条3D线段的包围盒是否重叠。
    中文：若两条线段在x、y、z三个轴向的投影均有重叠，则返回True（包围盒相交）。

    Rapid rejection test: check if the bounding boxes of two 3D segments overlap.
    English: Returns True if the segments' projections on x, y, z all overlap.

    :param v1_start: 第一条线段起点 / Start of first segment
    :param v1_end: 第一条线段终点 / End of first segment
    :param v2_start: 第二条线段起点 / Start of second segment
    :param v2_end: 第二条线段终点 / End of second segment
    :return: bool，包围盒是否重叠 / bool, whether bounding boxes overlap
    """
    Vo = [v1_start, v1_end]
    Vp = [v2_start, v2_end]
    if (    max(Vo[START][0], Vo[END][0]) >= min(Vp[START][0], Vp[END][0])
        and min(Vo[START][0], Vo[END][0]) <= max(Vp[START][0], Vp[END][0])
        and max(Vo[START][1], Vo[END][1]) >= min(Vp[START][1], Vp[END][1])
        and min(Vo[START][1], Vo[END][1]) <= max(Vp[START][1], Vp[END][1])
        and max(Vo[START][2], Vo[END][2]) >= min(Vp[START][2], Vp[END][2])
        and min(Vo[START][2], Vo[END][2]) <= max(Vp[START][2], Vp[END][2])
    ):
        return True
    else:
        return False

def straddle(v1_start, v1_end, v2_start, v2_end):
    """
    跨立测试：判断两条3D线段是否互相跨立（用于3D相交判断）。
    中文：通过计算两对叉积和点积，判断两线段是否满足跨立条件。

    Straddle test: check whether two 3D line segments straddle each other (for 3D intersection).
    English: Computes two pairs of cross products and dot products to verify straddle condition.

    :param v1_start: 第一条线段起点 / Start of first segment
    :param v1_end: 第一条线段终点 / End of first segment
    :param v2_start: 第二条线段起点 / Start of second segment
    :param v2_end: 第二条线段终点 / End of second segment
    :return: bool，是否满足跨立条件 / bool, whether straddle condition is satisfied
    """
    V1 = np.array([v1_start, v1_end])
    V2 = np.array([v2_start, v2_end])
    tempV1 = np.array([v1_start, v2_end])
    tempV2 = np.array([v1_start, v2_start])
    N1 = np.cross(tempV1, V1)
    N2 = np.cross(V1, tempV2)
    res1 = np.dot(N1, N2)

    tempV3 = np.array([v2_start, v1_end])
    tempV4 = np.array([v2_start, v1_start])
    N3 = np.cross(tempV3, V2)
    N4 = np.cross(V2, tempV4)
    res2 = np.dot(N3, N4)

    if res1 > 0 and res2 > 0:
        return True
    else:
        return False

def intersection3D(v1_start, v1_end, v2_start, v2_end):
    """
    判断两条3D线段是否相交（依次通过共面、快速排斥、跨立三重测试）。
    中文：先判断共面，再做快速包围盒排斥，最后做跨立测试，全部通过则相交。

    Determine if two 3D line segments intersect (coplanar + rapid repulsion + straddle test).
    English: First checks coplanarity, then rapid bounding-box repulsion, then straddle test;
             returns True only if all three tests pass.

    :param v1_start: 第一条线段起点 / Start of first segment
    :param v1_end: 第一条线段终点 / End of first segment
    :param v2_start: 第二条线段起点 / Start of second segment
    :param v2_end: 第二条线段终点 / End of second segment
    :return: bool，是否相交 / bool, whether segments intersect
    """
    if coplanar(v1_start, v1_end, v2_start, v2_end):
        if rapidRepel(v1_start, v1_end, v2_start, v2_end):
            if straddle(v1_start, v1_end, v2_start, v2_end):
                return True
    return False

def compute_3d_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    核心函数：通过SVD求解3D刚性变换矩阵（旋转+平移，无缩放/剪切）
    变换公式：目标点 = 旋转矩阵 @ 源点 + 平移向量
    输出：4x4齐次变换矩阵（方便3D图形学直接使用）
    :param source_points: 源点集 (N, 3)  numpy数组
    :param target_points: 目标点集 (N, 3) numpy数组
    :return: 4x4 齐次变换矩阵
    """
    # 校验点集维度和数量
    if source_points.shape != target_points.shape:
        raise ValueError("源点集和目标点集的点数/维度必须完全一致！")
    if source_points.shape[0] < 3:
        raise ValueError("至少需要3个不共线的3D点才能计算3D变换！")

    # 1. 计算点集质心
    mu_src = source_points.mean(axis=0)
    mu_tgt = target_points.mean(axis=0)

    # 2. 去中心化（去除平移影响）
    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    # 3. 计算协方差矩阵
    H = src_centered.T @ tgt_centered

    # 4. SVD奇异值分解求解旋转矩阵
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 处理反射情况（保证旋转矩阵行列式为1，右手坐标系）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 5. 计算平移向量
    t = mu_tgt - R @ mu_src

    # 6. 组合为4x4齐次变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R  # 左上角3x3为旋转矩阵
    transform_matrix[:3, 3] = t  # 最后一列前3行为平移向量

    return transform_matrix

class Unit:
    """
    折纸面片单元类，表示折纸图案中由折痕围成的一个多边形面片。
    包含折痕列表、质量分布、连接关系等属性，并提供面积计算、质量分配等方法。

    Origami panel unit class representing a polygonal facet bounded by creases in the origami pattern.
    Contains a list of creases, mass distribution, connectivity info, and methods for
    area calculation, mass assignment, etc.
    """
    def __init__(self) -> None:
        """
        初始化面片单元，清空所有属性。

        Initialize the panel unit with empty attributes.
        """
        self.crease = []
        self.seq_point = []
        self.connection = None
        self.connection_number = None
        self.mass = []
        self.consistent_mass = []
        self.special = False
        self.special_mass = 0.0
        self.repaired = False

    def reset(self):
        """
        重置面片单元，清空折痕列表。
        中文：清除当前单元的所有折痕，以便重新添加。

        Reset the unit by clearing all creases.
        English: Clears all creases in the unit for re-initialization.
        """
        self.crease.clear()

    def addCrease(self, c: Crease):
        """
        向面片单元中添加一条折痕。
        中文：将折痕对象追加到单元的折痕列表中。

        Add a crease to the panel unit.
        English: Appends a crease object to the unit's crease list.

        :param c: 要添加的折痕对象 / Crease object to add
        """
        self.crease.append(c)

    def repair(self, tolerance = 1.0):
        """
        修复面片单元中过短的折痕。
        中文：
            遍历折痕列表，将长度小于容差的折痕用其中点替换相邻折痕的端点，
            并删除过短折痕，直到所有折痕长度均大于容差为止。

        Repair degenerate (too-short) creases in the panel unit.
        English:
            Iterates the crease list, replaces adjacent endpoints with the midpoint
            of any crease shorter than tolerance, and removes the short crease,
            until all creases are longer than tolerance.

        :param tolerance: 最短折痕长度容差（默认2.0mm） / Minimum crease length tolerance (default 2.0mm)
        """
        while 1:
            problem = False
            i = 0
            while i < len(self.crease):
                crease = self.crease[i]
                length = crease.getLength()
                if length < tolerance:
                    problem = True
                    self.repaired = True
                    mid_point = crease.getMidPoint()
                    self.crease[i - 1].points[END] = mid_point
                    self.crease[(i + 1) % len(self.crease)].points[START] = mid_point
                    del(self.crease[i])
                i += 1
            if not problem:
                break

    def setupMassDirectly(self, mass):
        """
        直接按总质量设置面片各顶点的质量分配（集中质量矩阵和一致质量矩阵）。
        中文：
            将总质量 mass 按面积加权分配到各顶点，同时构建一致质量矩阵。
            适用于已知总质量的情况（而非面密度）。

        Set up vertex mass from a given total mass (lumped and consistent mass matrices).
        English:
            Distributes total mass proportionally to vertex areas and builds consistent mass matrix.
            Used when total mass (not area density) is known.

        :param mass: 面片总质量 / Total mass of the panel
        """
        kp_number = len(self.getSeqPoint())
        self.mass = [0. for _ in range(kp_number)]
        self.consistent_mass = np.zeros((kp_number, kp_number))
        new_rho = mass / self.calculateArea() / float(kp_number - 2)
        for i in range(kp_number - 2):
            for j in range(i + 1, i + 1 + kp_number - 2):
                current = j % kp_number
                next = (j + 1) % kp_number
                area = self.calculateTriArea(i, current, next)
                self.mass[i] += new_rho * area / 3.0
                self.mass[current] += new_rho * area / 3.0
                self.mass[next] += new_rho * area / 3.0
                self.consistent_mass[i][i] += new_rho * area / 6.0
                self.consistent_mass[current][current] += new_rho * area / 6.0
                self.consistent_mass[next][next] += new_rho * area / 6.0
                self.consistent_mass[i][current] += new_rho * area / 12.0
                self.consistent_mass[current][i] += new_rho * area / 12.0
                self.consistent_mass[i][next] += new_rho * area / 12.0
                self.consistent_mass[next][i] += new_rho * area / 12.0
                self.consistent_mass[current][next] += new_rho * area / 12.0
                self.consistent_mass[next][current] += new_rho * area / 12.0

    def setupMass(self, rho):
        """
        按面密度设置面片各顶点的质量分配（集中质量矩阵和一致质量矩阵）。
        中文：
            将面密度 rho 乘以三角剖分的面积，分配质量到各顶点，
            同时构建一致质量矩阵（用于有限元分析）。

        Set up vertex mass from area density (lumped and consistent mass matrices).
        English:
            Multiplies density rho by triangle areas to assign mass to each vertex,
            and builds the consistent mass matrix (for FEM analysis).

        :param rho: 面密度（单位面积质量） / Area density (mass per unit area)
        """
        kp_number = len(self.getSeqPoint())
        self.mass = [0. for _ in range(kp_number)]
        self.consistent_mass = np.zeros((kp_number, kp_number))
        new_rho = rho / float(kp_number - 2)
        for i in range(kp_number - 2):
            for j in range(i + 1, i + 1 + kp_number - 2):
                current = j % kp_number
                next = (j + 1) % kp_number
                area = self.calculateTriArea(i, current, next)
                self.mass[i] += new_rho * area / 3.0
                self.mass[current] += new_rho * area / 3.0
                self.mass[next] += new_rho * area / 3.0
                self.consistent_mass[i][i] = new_rho * area / 6.0
                self.consistent_mass[current][current] = new_rho * area / 6.0
                self.consistent_mass[next][next] = new_rho * area / 6.0
                self.consistent_mass[i][current] = self.consistent_mass[current][i] = new_rho * area / 12.0
                self.consistent_mass[i][next] = self.consistent_mass[next][i] = new_rho * area / 12.0
                self.consistent_mass[current][next] = self.consistent_mass[next][current] = new_rho * area / 12.0
    
    def setConnection(self, con):
        """
        设置面片的连接标识（与其他面片的连接关系）。
        中文：用于标记该面片连接到哪个结构或边界。

        Set the connection identifier for the panel (connectivity to other panels).
        English: Marks which structure or boundary this panel is connected to.

        :param con: 连接标识对象 / Connection identifier object
        """
        self.connection = con

    def setConnectionNumber(self, num):
        """
        设置面片的连接编号。
        中文：用于标记该面片属于哪个连接组。

        Set the connection number for the panel.
        English: Marks which connection group this panel belongs to.

        :param num: 连接编号 / Connection number
        """
        self.connection_number = num

    def calculateTriArea(self, i0, i1, i2):
        """
        计算由面片上三个顶点索引围成的三角形面积。
        中文：通过叉积公式计算三角形的面积（结果取绝对值）。

        Calculate the area of a triangle formed by three vertex indices of the panel.
        English: Uses the cross product formula to compute the triangle area (absolute value).

        :param i0: 第一个顶点索引 / Index of first vertex
        :param i1: 第二个顶点索引 / Index of second vertex
        :param i2: 第三个顶点索引 / Index of third vertex
        :return: 三角形面积（浮点数） / Triangle area (float)
        """
        kps = self.getSeqPoint()
        x0 = kps[i0]
        x1 = kps[i1]
        x2 = kps[i2]
        vec1 = [x1[X] - x0[X], x1[Y] - x0[Y]]
        vec2 = [x2[X] - x0[X], x2[Y] - x0[Y]]
        return abs(0.5 * (vec1[X] * vec2[Y] - vec1[Y] * vec2[X]))
    
    def calculateArea(self):
        """
        计算面片单元的总面积。
        中文：以第一个顶点为基准，将多边形分割为若干三角形，累加各三角形面积。

        Calculate the total area of the panel unit.
        English: Uses the first vertex as base, divides the polygon into triangles,
                 and sums up their areas.

        :return: 面片总面积（浮点数） / Total panel area (float)
        """
        kps = self.getSeqPoint()
        start_point = kps[0]
        area = 0.0
        for i in range(1, len(kps) - 1):
            cur_index = i
            next_index = i + 1
            vec1 = [kps[cur_index][X] - start_point[X], kps[cur_index][Y] - start_point[Y]]
            vec2 = [kps[next_index][X] - start_point[X], kps[next_index][Y] - start_point[Y]]
            area += abs(0.5 * (vec1[X] * vec2[Y] - vec1[Y] * vec2[X]))
        return area

    def isBorder(self):
        """
        判断该面片是否包含边界折痕（即是否为折纸的边缘面片）。
        中文：遍历折痕列表，若有任意一条折痕类型为 BORDER，则返回 True。

        Check if the panel contains any border crease (i.e., is on the edge of the origami).
        English: Returns True if any crease in the list is of type BORDER.

        :return: bool，是否含有边界折痕 / bool, whether the panel has a border crease
        """
        is_border = False
        for crease in self.crease:
            if crease.getType() == BORDER:
                is_border = True
                break
        return is_border
    
    def getCrease(self):
        """
        获取面片的所有折痕列表。
        中文：返回当前面片包含的所有折痕对象的列表（引用）。

        Get the list of all creases of the panel.
        English: Returns a reference to the list of all crease objects in the panel.

        :return: 折痕对象列表 / List of crease objects
        """
        return self.crease
    
    def getBorderCrease(self):
        """
        获取面片中所有边界折痕的深拷贝列表。
        中文：遍历折痕列表，筛选出类型为 BORDER 的折痕并返回深拷贝列表。

        Get a deep-copy list of all border creases in the panel.
        English: Filters creases of type BORDER and returns a deep-copied list.

        :return: 边界折痕的深拷贝列表 / Deep-copied list of border creases
        """
        creases = []
        for crease in self.crease:
            if crease.getType() == BORDER:
                creases.append(deepcopy(crease))
        return creases
    
    def getMaxX(self):
        """
        获取面片顶点中x坐标的最大值。

        Get the maximum x-coordinate among all vertices of the panel.

        :return: 最大x坐标（浮点数） / Maximum x-coordinate (float)
        """
        max_x = max([ele[X] for ele in self.seq_point])
        return max_x

    def getMaxY(self):
        """
        获取面片顶点中y坐标的最大值。

        Get the maximum y-coordinate among all vertices of the panel.

        :return: 最大y坐标（浮点数） / Maximum y-coordinate (float)
        """
        max_y = max([ele[Y] for ele in self.seq_point])
        return max_y

    def getMinX(self):
        """
        获取面片顶点中x坐标的最小值。

        Get the minimum x-coordinate among all vertices of the panel.

        :return: 最小x坐标（浮点数） / Minimum x-coordinate (float)
        """
        min_x = min([ele[X] for ele in self.seq_point])
        return min_x

    def getMinY(self):
        """
        获取面片顶点中y坐标的最小值。

        Get the minimum y-coordinate among all vertices of the panel.

        :return: 最小y坐标（浮点数） / Minimum y-coordinate (float)
        """
        min_y = min([ele[Y] for ele in self.seq_point])
        return min_y
    
    def getAABB(self):
        """
        获取面片的轴对齐包围盒（AABB）。
        中文：返回包围盒的左下角和右上角坐标。

        Get the axis-aligned bounding box (AABB) of the panel.
        English: Returns the bottom-left and top-right corners of the bounding box.

        :return: [[min_x, min_y], [max_x, max_y]]
        """
        min_x = self.getMinX()
        min_y = self.getMinY()
        max_x = self.getMaxX()
        max_y = self.getMaxY()
        return [[min_x, min_y], [max_x, max_y]]

    def getSeqPoint(self):
        """
        获取面片所有顶点的有序坐标列表（按折痕起点顺序）。
        中文：
            遍历折痕列表，提取每条折痕的起点作为顶点，
            返回一个按顺序排列的 [x, y, z] 坐标列表（z若无则为0）。

        Get the ordered list of all vertex coordinates of the panel (in crease start-point order).
        English:
            Iterates the crease list, extracts each crease's start point as a vertex,
            returns an ordered list of [x, y, z] coordinates (z=0 if not present).

        :return: 顶点坐标的有序列表 [[x, y, z], ...] / Ordered vertex coordinate list [[x, y, z], ...]
        """
        self.seq_point.clear()
        for ele in self.crease:
            if len(ele[START]) == 2:
                height = 0.0
            else:
                height = ele[START][Z]
            self.seq_point.append([ele[START][X], ele[START][Y], height])
        return self.seq_point
    
    def getCenter(self):
        """
        计算面片的几何重心（加权中心）。
        中文：用各顶点的面积贡献系数加权求和，得到面片的质心坐标。

        Calculate the geometric centroid of the panel (area-weighted center).
        English: Weighted sum of vertex positions using area contribution coefficients.

        :return: 质心坐标（numpy array） / Centroid coordinates (numpy array)
        """
        seq_point = np.array(self.getSeqPoint())
        c = self.getContribution()
        center = sum([c[i] * seq_point[i] for i in range(seq_point.shape[0])])
        return center
    
    def getCenterUsingContribution(self, c):
        """
        使用外部传入的贡献系数计算面片的加权重心。
        中文：用给定系数 c 对顶点坐标加权求和，适用于需要使用预计算系数的场景。

        Calculate the weighted centroid using an externally provided contribution coefficient list.
        English: Weighted sum of vertex positions using the given coefficient array c.

        :param c: 各顶点的贡献系数列表（长度等于顶点数） / Contribution coefficient list for each vertex
        :return: 加权重心坐标（numpy array） / Weighted centroid (numpy array)
        """
        seq_point = np.array(self.getSeqPoint())
        center = sum([c[i] * seq_point[i] for i in range(seq_point.shape[0])])
        return center
    
    def getContribution(self):
        """
        计算各顶点对面片重心的面积贡献系数。
        中文：
            对于三角形面片，每个顶点系数均为 1/3；
            对于多边形面片，将其三角剖分后按面积加权计算各顶点系数，
            系数之和为1（类似重心坐标）。

        Calculate the area contribution coefficients for each vertex toward the panel centroid.
        English:
            For triangular panels, each vertex coefficient is 1/3;
            for polygons, triangulates and computes area-weighted coefficients per vertex;
            coefficients sum to 1 (similar to barycentric coordinates).

        :return: 各顶点贡献系数列表（长度等于顶点数） / List of contribution coefficients (length = vertex count)
        """
        seq_point = np.array(self.getSeqPoint())
        kp_len = len(seq_point)
        if kp_len == 3:
            return [1/3, 1/3, 1/3]
        else:
            area_tri = [
                abs(0.5 * ((seq_point[i][X] - seq_point[0][X]) * (seq_point[i + 1][Y] - seq_point[0][Y]) - (seq_point[i][Y] - seq_point[0][Y]) * (seq_point[i + 1][X] - seq_point[0][X]))) for i in range(1, kp_len - 1)
            ]

            c = np.zeros(shape=kp_len)
            for k in range(kp_len - 2):
                c[0] += 1/3 * area_tri[k]
                c[k+1] += 1/3 * area_tri[k]
                c[k+2] += 1/3 * area_tri[k]
            
            c /= sum(area_tri)
            return c.tolist()
        
    def __repr__(self) -> str:
        return f"kp_num: {len(self.crease)}"

class TSAPoint:
    """
    TSA（Twist-String Actuator，双绞线驱动器）绳子路径点类。
    中文：表示穿绳路径中的一个关键点，包含点坐标、点类型（边界A或折纸B）、编号和穿绕方向。

    TSA (Twist-Spring Actuator) threading path point class.
    English: Represents a key point in the string threading path,
             with coordinates, point type (border A or origami B), ID, and threading direction.
    """
    def __init__(self) -> None:
        """
        初始化TSA路径点，默认坐标为原点，类型为A（边界），方向为1（从上到下）。

        Initialize a TSA path point with default origin coordinates, type A (border), direction 1 (top-to-bottom).
        """
        self.point = np.array([0.0, 0.0, 0.0])
        self.point_type = 'A' # A: border, B: origami
        self.id = 0
        self.dir = 1 # 1 means up to down, -1 means down to up

class TSAString:
    """
    TSA绳子类，表示穿过折纸孔洞的一段驱动绳子（绳索段）。
    中文：
        存储绳子的宽度、高度、起终点、类型（底部/中间/顶部穿越）、
        固定/自由端属性以及ID等信息，并提供3D圆柱体点生成方法。

    TSA string class representing a segment of the actuating string threading through origami holes.
    English:
        Stores string width, height, start/end points, type (bottom/pass/top),
        free/fixed end properties, IDs, and provides a method to generate 3D cylinder points.
    """
    def __init__(self) -> None:
        """
        初始化TSA绳子，设置默认的宽度、高度、类型和端点属性。

        Initialize a TSA string with default width, height, type, and endpoint properties.
        """
        self.width = 1.0
        self.height = 1.2
        self.start_point = np.array([0.0, 0.0, 0.0])
        self.end_point = np.array([0.0, 0.0, 0.0])
        self.type = BOTTOM
        self.start_type = FREE
        self.end_type = FREE
        self.ids = [] # start, end
        self.id_types = [] # start, end
        self.id = -1
    
    def setStringWidth(self, width: float):
        """
        设置绳子的宽度（直径）。

        Set the width (diameter) of the string.

        :param width: 绳子宽度（mm） / String width (mm)
        """
        self.width = width

    def setStringHeight(self, height: float):
        """
        设置绳子段的高度。

        Set the height of the string segment.

        :param height: 绳子高度（mm） / String height (mm)
        """
        self.height = height

    def setStringKeyPoint(self, start, end):
        """
        设置绳子的起点和终点坐标。

        Set the start and end key point coordinates of the string.

        :param start: 起点坐标 [x, y, z] / Start point [x, y, z]
        :param end: 终点坐标 [x, y, z] / End point [x, y, z]
        """
        self.start_point = np.array(start)
        self.end_point = np.array(end)

    def setStartType(self, type):
        """
        设置绳子起点的端部类型（FREE=自由端, FIXED=固定端）。

        Set the end type of the string start point (FREE or FIXED).

        :param type: 端部类型常量（FREE或FIXED） / End type constant (FREE or FIXED)
        """
        self.start_type = type

    def setEndType(self, type):
        """
        设置绳子终点的端部类型（FREE=自由端, FIXED=固定端）。

        Set the end type of the string end point (FREE or FIXED).

        :param type: 端部类型常量（FREE或FIXED） / End type constant (FREE or FIXED)
        """
        self.end_type = type

    def getDirectionVector(self):
        """
        获取绳子的方向向量（从起点到终点）。

        Get the direction vector of the string (from start to end).

        :return: 方向向量（numpy array） / Direction vector (numpy array)
        """
        return self.end_point - self.start_point

    def generatePointWithResolution(self, resolution):
        """
        生成绳子两端圆截面上的点列表（用于STL 3D建模）。
        中文：
            根据绳子的类型（BOTTOM/TOP或PASS）生成起点端和终点端各 resolution 个圆截面点，
            用于后续生成圆柱体STL模型。

        Generate point lists for the circular cross-sections at both ends of the string (for STL 3D modeling).
        English:
            Based on string type (BOTTOM/TOP or PASS), generates 'resolution' points
            at both the start and end circular cross-sections for STL cylinder generation.

        :param resolution: 圆截面的采样点数 / Number of sampling points on the circular cross-section
        :return: (start圆截面点列表, end圆截面点列表) / (start circle point list, end circle point list)
        """
        vec = self.getDirectionVector()
        vec_length = np.linalg.norm(vec)
        if self.type != PASS:
            angle = np.arctan(vec[Y] / vec[X])
            if vec[X] < 0.0:
                angle += math.pi
            face_angle = angle - math.pi / 2.0
        start_point = self.start_point
        point_list = []
        # if end_point[Y] == start_point[Y]:
        #     new_first_point = np.array([start_point[X], start_point[Y] + self.width / 2.0, start_point[Z]])
        # else:
        #     K = (end_point[Z] - start_point[Z]) / (end_point[Y] - start_point[Y])
        #     new_first_point = np.array([start_point[X], start_point[Y] + self.width / 2.0 * K/(np.sqrt(K^2 + 1)), start_point[Z] + self.width / 2.0 /np.sqrt(K^2 + 1)])
        # point_list.append(new_first_point)
        if self.type != PASS: # K=0
            for i in range(resolution):
                point_list.append([start_point[X] + self.width / 2.0 * np.cos(2.0 * i / resolution * np.pi) * np.cos(face_angle), 
                                            start_point[Y] + self.width / 2.0 * np.cos(2.0 * i / resolution * np.pi) * np.sin(face_angle),
                                            start_point[Z] + self.width / 2.0 * np.sin(2.0 * i / resolution * np.pi)])
        else:
            for i in range(resolution):
                point_list.append([start_point[X] + self.width / 2.0 * np.cos(2.0 * i / resolution * np.pi), 
                                            start_point[Y] + self.width / 2.0 * np.sin(2.0 * i / resolution * np.pi),
                                            start_point[Z]])
        upper_point_list = []
        for ele in point_list:
            upper_point_list.append(ele + vec)

        return point_list, upper_point_list

class Node:
    """
    MCTS（蒙特卡洛树搜索）节点类。
    中文：
        表示搜索树中的一个节点，包含当前状态（穿绳状态编码）、
        访问次数、累计奖励、子节点列表及UCB1选择方法。

    MCTS (Monte Carlo Tree Search) node class.
    English:
        Represents a node in the search tree, containing the current state (string threading encoding),
        visit count, cumulative reward, children list, and UCB1 selection method.
    """
    def __init__(self, state, maximum_child, done, action_id, string_id, parent=None) -> None:
        """
        初始化MCTS节点。
        中文：设置节点状态、最大子节点数、完成标志、动作编号和父节点等信息。

        Initialize an MCTS node.
        English: Sets node state, max children count, done flag, action ID, and parent reference.

        :param state: 当前穿绳状态（numpy uint8数组） / Current string state (numpy uint8 array)
        :param maximum_child: 最大子节点数（可能的动作数） / Max children count (number of possible actions)
        :param done: 是否为终止节点 / Whether this is a terminal node
        :param action_id: 到达该节点的动作编号 / Action ID that leads to this node
        :param string_id: 对应的绳子编号 / Corresponding string ID
        :param parent: 父节点（根节点为None） / Parent node (None for root)
        """
        # xn + sqrt(2ln(N) / v)  [UCB1 formula]
        self.state = np.array(state, dtype=np.uint8)
        self.visits = 1
        self.reward = 0.0
        # self.state = state
        self.maximum_child = maximum_child
        self.string_id = string_id

        self.action_id = action_id
        self.done = done
        self.children = []
        self.parent = parent

    def initializeTree(self, standard):
        """
        重置搜索树（重置访问次数和奖励），保留树结构。
        中文：将所有节点的访问次数重置为1，非终止节点的奖励清零，递归处理所有子节点。

        Reset the search tree (visit counts and rewards), keeping the tree structure.
        English: Resets all nodes' visits to 1 and clears rewards of non-terminal nodes,
                 recursively processing all children.

        :param standard: 奖励阈值（保留参数，此版本未使用） / Reward threshold (reserved, unused here)
        """
        self.visits = 1
        
        if not self.done:
            self.reward = 0.0
            for c in self.children:
                c.initializeTree(standard)
                
    def initializeTree2(self, standard):
        """
        重置搜索树并剪枝：删除平均奖励低于阈值的子树。
        中文：
            重置访问次数，若当前节点不存在满足阈值的优质子节点，
            则删除所有子节点（剪枝），标记为终止节点。
            返回被删除的子节点总数。

        Reset and prune the search tree: removes subtrees with average reward below threshold.
        English:
            Resets visit counts; if no high-quality child (above threshold) exists,
            deletes all children (pruning) and marks node as done.
            Returns total number of pruned children.

        :param standard: 奖励阈值，低于此值的子节点将被剪掉 / Reward threshold; children below this are pruned
        :return: 被删除的子节点总数 / Total number of pruned children
        """
        self.visits = 1
        cut_num = 0
        
        if not self.done:
            self.reward = 0.0
            if not self.existBestChild(standard):
                cut_num += self.maximum_child
                while len(self.children):
                    del(self.children[0])
                self.children = None
                # print(f"Node with level {level} is cut due to average reward {node.reward / (node.visits - 1)} < {self.best_reward}, cut_number: {cut_num}")
                self.done = 1
            else:
                for c in self.children:
                    cut_num += c.initializeTree2(standard)

        return cut_num

    # def cutChild(self, standard):
    #     cut_num = 0
    #     if not self.existBestChild(standard):
    #         cut_num += self.maximum_child
    #         while len(self.children):
    #             del(self.children[0])
    #         self.children = None
    #         self.done = 1
            
    #     return cut_num

    def addChild(self, child_state, maximum_child, done, action_id, string_id):
        """
        为当前节点添加一个子节点。
        中文：创建新的 Node 对象并追加到子节点列表，父节点为当前节点。

        Add a child node to the current node.
        English: Creates a new Node and appends it to the children list with this node as parent.

        :param child_state: 子节点的状态 / State of the child node
        :param maximum_child: 子节点的最大子节点数 / Max children of the child
        :param done: 子节点是否为终止节点 / Whether child is a terminal node
        :param action_id: 到达子节点的动作编号 / Action ID leading to child
        :param string_id: 子节点对应的绳子编号 / String ID corresponding to child
        """
        self.children.append(Node(child_state, maximum_child, done, action_id, string_id, self))
    
    def update(self, reward):
        """
        更新节点的累计奖励和访问次数（反向传播用）。
        中文：将本次仿真的奖励值加到节点奖励上，并将访问次数+1。

        Update the node's cumulative reward and visit count (for backpropagation).
        English: Adds the simulation reward to the node's total reward, and increments visit count.

        :param reward: 本次仿真获得的奖励值 / Reward from current simulation
        """
        self.reward += reward
        self.visits += 1

    def fullyExpanded(self):
        """
        判断当前节点是否已完全展开（所有可能的子节点均已被探索）。
        中文：若子节点数等于最大子节点数，则说明已完全展开。

        Check if the current node is fully expanded (all possible children have been explored).
        English: Returns True if the number of children equals the maximum allowed.

        :return: bool，是否完全展开 / bool, whether fully expanded
        """
        return len(self.children) == self.maximum_child

    def existOnlyOneValid(self, standard):
        """
        统计当前节点中有效子节点的数量（包括未展开的槽位和有效的已展开子节点）。
        中文：
            有效子节点包括：尚未创建的子节点槽位，以及平均奖励超过阈值且未被剪枝的子节点。

        Count the number of valid children (including unexplored slots and valid expanded children).
        English:
            Valid children include: unfilled child slots, and expanded children
            with average reward above threshold that haven't been pruned.

        :param standard: 奖励阈值 / Reward threshold
        :return: 有效子节点数量（整数） / Number of valid children (int)
        """
        valid = self.maximum_child - len(self.children)
        for c in self.children:
            if c.done and c.reward <= standard:
                continue
            # elif c.done and c.visits == 1:
            #     continue
            elif c.done and c.visits > 1 and c.reward / (c.visits - 1) <= standard:
                continue
            elif c.children == None:
                continue
            else:
                valid += 1
        return valid

    def existBestChild(self, standard):
        """
        判断当前节点是否存在满足阈值的优质子节点（或仍有未展开槽位）。
        中文：若尚未完全展开，或存在平均奖励超过阈值且未被剪枝的子节点，则返回True。

        Check if there exists any high-quality child (or unfilled slot) above the threshold.
        English: Returns True if not fully expanded, or if there's a child with average reward
                 above threshold that hasn't been pruned.

        :param standard: 奖励阈值 / Reward threshold
        :return: bool，是否存在优质子节点 / bool, whether a good child exists
        """
        exist = False
        if len(self.children) < self.maximum_child:
            exist = True
        else:
            for c in self.children:
                if c.done and c.reward <= standard:
                    continue
                # elif c.done and c.visits == 1:
                #     continue
                elif c.done and c.visits > 1 and c.reward / (c.visits - 1) <= standard:
                    continue
                elif c.children == None:
                    continue
                else:
                    exist = True
                    break
        return exist

    def existChildWithAction(self, action_id):
        """
        检查当前节点是否已存在对应动作编号的子节点。
        中文：遍历子节点列表，若找到对应 action_id 的子节点，则返回 (True, 该子节点)。

        Check if a child with the given action ID already exists.
        English: Searches the children list; returns (True, child) if found, (False, None) otherwise.

        :param action_id: 要查找的动作编号 / Action ID to search for
        :return: (bool是否存在, 子节点或None) / (bool exists, child node or None)
        """
        exist = False
        child = None
        for c in self.children:
            if c.action_id == action_id:
                exist = True
                child = c
                break
        return exist, child
    
    def bestChild(self, scalar, standard):
        """
        使用UCB1公式选择最优子节点。
        中文：
            对所有有效子节点计算UCB1分数（利用率 + scalar * 探索率），
            返回分数最高的子节点（若有并列则随机选一个）。

        Select the best child node using the UCB1 formula.
        English:
            Computes UCB1 score (exploitation + scalar * exploration) for all valid children,
            returns the child with the highest score (random among ties).

        :param scalar: UCB1 探索系数（通常为 sqrt(2)） / UCB1 exploration scalar (typically sqrt(2))
        :param standard: 奖励阈值（低于此值的子节点被跳过） / Reward threshold (children below are skipped)
        :return: (最优子节点, 对应动作编号) / (best child node, corresponding action ID)
        """
        best_score = -math.inf
        best_children = []
        for c in self.children:
            if c.done and c.reward <= standard:
                continue
            # elif c.done and c.visits == 1:
            #     continue
            elif c.done and c.visits > 1 and c.reward / (c.visits - 1) <= standard:
                continue
            elif c.children == None:
                continue
            if c.visits != 1:
                exploit = c.reward / (c.visits - 1)
            else:
                exploit = 0.
            explore = math.sqrt(2. * math.log(self.visits) / float(c.visits))
            score = exploit + scalar * explore
            if score == best_score:
                best_children.append(c)
            elif score > best_score:
                best_children = [c]
                best_score = score
        if len(best_children) == 0:
            return self.children[0], self.children[0].action_id
        best_child = np.random.choice(best_children)
        action_id = best_child.action_id
        return best_child, action_id

    def __repr__(self) -> str:
        # return f"Visits: {self.visits}, Reward: {self.reward}, Done: {self.done}"
        if self.parent == None:
            return f"{self.state}"
        else:
            if self.visits == 1:
                return f"{self.state}, -1"
            else:
                return f"{self.state}, UCB: {self.reward / (self.visits - 1) + math.sqrt(2. * math.log(self.parent.visits) / float(self.visits))}, VIS: {self.visits}, EXPLOIT: {self.reward / (self.visits - 1)}"

class HalfEdge:
    """
    半边数据结构类，用于多边形识别时构建平面图。
    中文：
        每条有向边记录起点、终点、原始边索引、配对半边、下一条半边及遍历标记，
        支持高效的平面图面追踪操作。

    Half-edge data structure class, used for building planar graphs during polygon identification.
    English:
        Each directed edge stores start, end, original edge index, twin half-edge,
        next half-edge, and visited flag, supporting efficient planar face traversal.
    """
    def __init__(self, start, end, line_idx):
        """
        初始化半边对象。
        中文：设置起终点、原始边索引，twin和next初始为None，visited为False。

        Initialize a half-edge object.
        English: Sets start/end, original line index; twin and next default to None, visited to False.

        :param start: 起点坐标或节点ID / Start point coordinate or node ID
        :param end: 终点坐标或节点ID / End point coordinate or node ID
        :param line_idx: 对应的原始线段索引 / Index of the original line segment
        """
        self.start = start      # 起点
        self.end = end          # 终点
        self.line_idx = line_idx# 原始边索引
        self.twin = None        # 配对半边
        self.next = None        # 下一条半边
        self.visited = False    # 遍历标记

import bisect
from collections import deque

class PolygonIdentifier:
    """
    从线段集合中识别封闭多边形面单元的类。
    中文：
        基于平面图的面追踪算法（最小化左转策略），
        从给定的线段列表中识别出所有独立闭合多边形（面片），
        并返回各多边形所使用的线段ID列表。

    Class for identifying closed polygon units from a collection of line segments.
    English:
        Uses a planar graph face-tracing algorithm (minimum left-turn strategy)
        to identify all independent closed polygons (facets) from a list of segments,
        returning the segment ID lists for each polygon.
    """
    def __init__(self, lines):
        """
        初始化多边形识别器。
        中文：存储待分析的线段列表。

        Initialize the polygon identifier.
        English: Stores the list of line segments to analyze.

        :param lines: 待分析的线段列表（每个线段需有 START/END 属性） / List of line segments (each with START/END attributes)
        """
        self.new_lines = lines

    def identify_polygons(self):
        """
        从线段集合中识别所有独立闭合多边形单元。
        中文：
            构建平面图，删除悬挂边，利用最小左转面追踪算法识别所有内部面，
            筛选不包含其他多边形顶点的最小面，返回各面的线段ID列表。

        Identify all independent closed polygon units from the line segment collection.
        English:
            Builds a planar graph, removes dangling edges, uses minimum left-turn face tracing
            to identify all interior faces, filters minimal faces not containing other polygon vertices,
            and returns the segment ID lists for each face.

        :return: List[List[int]]，每个子列表为一个多边形所使用的线段ID列表 / List of segment ID lists per polygon
        """
        # 构建初始图结构
        nodes = {}          # 坐标 -> 节点索引
        node_coords = []    # 节点索引 -> (x, y)
        edges = []          # (node_u, node_v, edge_id)
        
        for id, line in enumerate(self.new_lines):
            p1 = line[START]
            p2 = line[END]
            # 规范化坐标（确保元组）
            key1 = (p1.x, p1.y) if hasattr(p1, 'x') else tuple(p1)
            key2 = (p2.x, p2.y) if hasattr(p2, 'x') else tuple(p2)
            if key1 == key2:
                continue  # 忽略退化线段
            if key1 not in nodes:
                nodes[key1] = len(node_coords)
                node_coords.append(key1)
            if key2 not in nodes:
                nodes[key2] = len(node_coords)
                node_coords.append(key2)
            u = nodes[key1]
            v = nodes[key2]
            edges.append((u, v, id))
        
        # 构建邻接表（无向）
        adj = [[] for _ in range(len(node_coords))]
        for u, v, eid in edges:
            adj[u].append((v, eid))
            adj[v].append((u, eid))
        
        # 预处理：删除悬挂边（度为1的节点）
        degree = [len(adj[i]) for i in range(len(node_coords))]
        active_edge = [True] * len(edges)
        active_node = [True] * len(node_coords)
        q = deque([i for i, d in enumerate(degree) if d == 1])
        
        while q:
            u = q.popleft()
            if not active_node[u] or degree[u] != 1:
                continue
            # 找到唯一未删除的邻边
            for v, eid in adj[u]:
                if active_edge[eid]:
                    # 删除该边
                    active_edge[eid] = False
                    degree[u] -= 1
                    degree[v] -= 1
                    active_node[u] = False
                    if degree[v] == 1:
                        q.append(v)
                    break
        
        # 收集剩余的有效边和节点
        new_edges = []
        node_map = {}       # 旧节点索引 -> 新节点索引
        new_coords = []
        for i, (u, v, eid) in enumerate(edges):
            if active_edge[i] and active_node[u] and active_node[v]:
                if u not in node_map:
                    node_map[u] = len(new_coords)
                    new_coords.append(node_coords[u])
                if v not in node_map:
                    node_map[v] = len(new_coords)
                    new_coords.append(node_coords[v])
                new_edges.append((node_map[u], node_map[v], eid))
        
        if not new_edges:
            return []
        
        # 重新构建邻接表（带角度）
        n = len(new_coords)
        adj2 = [[] for _ in range(n)]
        for u, v, eid in new_edges:
            dx = new_coords[v][0] - new_coords[u][0]
            dy = new_coords[v][1] - new_coords[u][1]
            ang = math.atan2(dy, dx)
            adj2[u].append((v, eid, ang))
            # 反向边
            dx = new_coords[u][0] - new_coords[v][0]
            dy = new_coords[u][1] - new_coords[v][1]
            ang_rev = math.atan2(dy, dx)
            adj2[v].append((u, eid, ang_rev))
        
        # 对每个节点的邻接表按角度排序
        for i in range(n):
            adj2[i].sort(key=lambda x: x[2])
        
        # 面追踪
        visited = {}
        faces = []  # 每个元素: (edge_ids, vertices, area)
        
        for u in range(n):
            for v, eid, ang in adj2[u]:
                if (u, v) in visited:
                    continue
                # 开始追踪一个新的有向环
                edge_ids = []
                vertices = []
                cur_u, cur_v = u, v
                while (cur_u, cur_v) not in visited:
                    visited[(cur_u, cur_v)] = True
                    # 记录当前边的ID和起点坐标
                    edge_ids.append(self._get_edge_id(cur_u, cur_v, adj2, new_coords))  # 需实现辅助函数
                    vertices.append(new_coords[cur_u])
                    # 在 cur_v 处找下一条边
                    # 计算进入方向 (cur_v -> cur_u) 的角度
                    dx_in = new_coords[cur_u][0] - new_coords[cur_v][0]
                    dy_in = new_coords[cur_u][1] - new_coords[cur_v][1]
                    ang_in = math.atan2(dy_in, dx_in)
                    # 获取 cur_v 的邻接表（已排序）
                    neighs = adj2[cur_v]
                    angles = [a for (_, _, a) in neighs]
                    # 二分查找第一个大于 ang_in 的角度
                    i = bisect.bisect_right(angles, ang_in)
                    if i == len(angles):
                        i = 0
                    next_neighbor, _, _ = neighs[i]
                    cur_u, cur_v = cur_v, next_neighbor
                    if (cur_u, cur_v) == (u, v):
                        break
                # 计算有符号面积
                area = self._polygon_area(vertices)
                if abs(area) < 1e-2:
                    continue  # 退化环
                faces.append((edge_ids, vertices, area))
        
        # 筛选内部不包含其他顶点的环
        result = []
        n_faces = len(faces)
        # 预先计算每个面的包围盒
        bboxes = []
        for _, verts, _ in faces:
            xs = [p[0] for p in verts]
            ys = [p[1] for p in verts]
            bboxes.append((min(xs), max(xs), min(ys), max(ys)))
        
        for i in range(n_faces):
            edge_ids_i, verts_i, area_i = faces[i]
            # 检查是否存在其他面的顶点位于该面内部
            contained = False
            for j in range(n_faces):
                if i == j:
                    continue
                # 快速包围盒排除
                bxmin, bxmax, bymin, bymax = bboxes[j]
                ixmin, ixmax, iymin, iymax = bboxes[i]
                if not (ixmin <= bxmin <= ixmax and ixmin <= bxmax <= ixmax and
                        iymin <= bymin <= iymax and iymin <= bymax <= iymax):
                    # 如果 j 的包围盒不完全在 i 的包围盒内，则 j 的顶点不可能都在 i 内部（但可能有部分顶点？）
                    # 为了严格，我们仍需要检查顶点，但这里跳过以提高性能（如果包围盒不相交则不可能包含）
                    # 注意：包含要求所有顶点在内部，所以 j 的包围盒必须完全在 i 的包围盒内
                    continue
                # 检查 j 的每个顶点是否在 i 内部
                _, verts_j, _ = faces[j]
                for p in verts_j:
                    if self._point_in_polygon(p[0], p[1], verts_i):
                        contained = True
                        break
                if contained:
                    break
            if not contained:
                result.append(edge_ids_i)
        
        return result
    
    def _get_edge_id(self, u, v, adj, coords):
        """
        根据有向边 (u->v) 获取对应的线段ID。
        中文：在邻接表 adj[u] 中查找目标节点 v，返回对应的线段ID。

        Get the segment ID for the directed edge (u->v).
        English: Searches adj[u] for neighbor v and returns the corresponding edge ID.

        :param u: 起点节点索引 / Start node index
        :param v: 终点节点索引 / End node index
        :param adj: 邻接表（含角度） / Adjacency list (with angles)
        :param coords: 节点坐标列表（未使用，保留） / Node coordinate list (unused, reserved)
        :return: 线段ID（整数） / Segment ID (int)
        :raises ValueError: 若边不存在则抛出异常 / Raises ValueError if edge not found
        """
        for nb, eid, _ in adj[u]:
            if nb == v:
                return eid
        raise ValueError("Edge not found")
    
    def _polygon_area(self, vertices):
        """
        计算多边形的有符号面积（正=逆时针，负=顺时针）。
        中文：使用鞋带公式（Shoelace formula）计算有符号面积。

        Calculate the signed area of a polygon (positive=CCW, negative=CW).
        English: Uses the Shoelace formula to compute the signed area.

        :param vertices: 多边形顶点列表 [(x1,y1), (x2,y2), ...] / Polygon vertex list [(x1,y1), ...]
        :return: 有符号面积（浮点数） / Signed area (float)
        """
        area = 0.0
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i+1)%n]
            area += x1*y2 - x2*y1
        return area * 0.5
    
    def _point_in_polygon(self, px, py, poly):
        """
        用射线法判断点是否在多边形内部（不含边界）。
        中文：若点在边界上，返回False；否则用射线法判断内部。

        Check if a point is strictly inside a polygon using ray casting (excluding boundary).
        English: Returns False if on boundary; uses ray casting for interior test.

        :param px: 点的x坐标 / Point x-coordinate
        :param py: 点的y坐标 / Point y-coordinate
        :param poly: 多边形顶点列表 [(x1,y1), ...] / Polygon vertex list [(x1,y1), ...]
        :return: bool，是否在内部（不含边界） / bool, whether strictly inside (excluding boundary)
        """
        # 首先检查是否在边界上
        if self._point_on_polygon_boundary(px, py, poly):
            return False
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            # 射线从 (px,py) 向右水平
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside
    
    def _point_on_polygon_boundary(self, px, py, poly):
        """
        判断点是否在多边形的边界上（含顶点和边）。
        中文：遍历多边形所有边，若点在任意边上则返回True。

        Check if a point lies on the polygon boundary (including vertices and edges).
        English: Iterates all polygon edges; returns True if the point lies on any edge.

        :param px: 点的x坐标 / Point x-coordinate
        :param py: 点的y坐标 / Point y-coordinate
        :param poly: 多边形顶点列表 [(x1,y1), ...] / Polygon vertex list [(x1,y1), ...]
        :return: bool，是否在边界上 / bool, whether on the boundary
        """
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            if self._point_on_segment(px, py, x1, y1, x2, y2):
                return True
        return False
    
    def _point_on_segment(self, px, py, x1, y1, x2, y2):
        """
        判断点 (px, py) 是否在线段 (x1,y1)-(x2,y2) 上（含端点）。
        中文：先用叉积判断共线，再用点积判断投影范围是否在线段内。

        Check if point (px, py) lies on segment (x1,y1)-(x2,y2) (including endpoints).
        English: Uses cross product for collinearity, dot product for projection range check.

        :return: bool，是否在线段上 / bool, whether on the segment
        """
        # 叉积判断共线
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) > 1e-2:
            return False
        # 点积判断投影范围
        dot = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
        if dot < -1e-2:
            return False
        sqlen = (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1)
        if dot > sqlen + 1e-2:
            return False
        return True
    
if __name__ == "__main__":
    # tsa = TSAString()
    # tsa.setStringKeyPoint([0.0, 0.0, 0.0], [10.0, 10.0, 0])
    # a, b = tsa.generatePointWithResolution(4)
    # c = 1
    unit = Unit()
    unit.addCrease(Crease([0., 0.], [10., 0.], BORDER))
    unit.addCrease(Crease([10., 0.], [0., 10.], BORDER))
    unit.addCrease(Crease([0., 10.], [0., 0.], BORDER))
    center = unit.getCenter()
    unit.setupMass(1)
    a = 1