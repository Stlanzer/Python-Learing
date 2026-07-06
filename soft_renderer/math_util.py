"""
数学工具模块：向量/矩阵运算 & MVP 变换矩阵

不依赖任何数学库，纯手写 4x4 矩阵和向量运算。
"""

import math


class Vec3:
    """三维向量"""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s):
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __truediv__(self, s):
        return Vec3(self.x / s, self.y / s, self.z / s)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        length = self.length()
        if length > 1e-8:
            return Vec3(self.x / length, self.y / length, self.z / length)
        return Vec3(0, 0, 0)

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


class Vec4:
    """四维向量（齐次坐标）"""
    __slots__ = ('x', 'y', 'z', 'w')

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def perspective_divide(self):
        """透视除法：从齐次坐标变换到三维空间"""
        if abs(self.w) > 1e-8:
            return Vec3(self.x / self.w, self.y / self.w, self.z / self.w)
        return Vec3(self.x, self.y, self.z)


class Mat4:
    """
    4x4 矩阵（列主序存储，与 OpenGL 一致）
    索引：m[row][col] 或通过 __getitem__ 访问
    """
    def __init__(self):
        # 初始化为单位矩阵
        self.m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def look_at(cls, eye: Vec3, center: Vec3, up: Vec3):
        """
        视图矩阵 (View Matrix)
        将世界坐标变换到相机空间
        """
        f = (center - eye).normalize()          # 前方向
        s = f.cross(up.normalize()).normalize() # 右方向
        u = s.cross(f)                          # 上方向

        m = cls()
        m.m = [
            [s.x,  u.x, -f.x, 0.0],
            [s.y,  u.y, -f.y, 0.0],
            [s.z,  u.z, -f.z, 0.0],
            [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
        ]
        return m

    @classmethod
    def perspective(cls, fov_y: float, aspect: float, near: float, far: float):
        """
        透视投影矩阵
        fov_y: 垂直视场角（度）
        aspect: 宽高比
        """
        f = 1.0 / math.tan(math.radians(fov_y) / 2.0)
        m = cls()
        m.m = [
            [f / aspect, 0.0,  0.0,                             0.0],
            [0.0,        f,    0.0,                             0.0],
            [0.0,        0.0,  (far + near) / (near - far),    -1.0],
            [0.0,        0.0,  (2.0 * far * near) / (near - far), 0.0],
        ]
        return m

    @classmethod
    def rotate_y(cls, angle: float):
        """绕 Y 轴旋转矩阵"""
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        m = cls()
        m.m = [
            [ c, 0.0,  s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0,  c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return m

    @classmethod
    def rotate_x(cls, angle: float):
        """绕 X 轴旋转矩阵"""
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        m = cls()
        m.m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0,  c,  -s, 0.0],
            [0.0,  s,   c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return m

    @classmethod
    def rotate_z(cls, angle: float):
        """绕 Z 轴旋转矩阵"""
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        m = cls()
        m.m = [
            [ c, -s, 0.0, 0.0],
            [ s,  c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return m

    @classmethod
    def translate(cls, tx: float, ty: float, tz: float):
        """平移矩阵"""
        m = cls()
        m.m[3][0] = tx
        m.m[3][1] = ty
        m.m[3][2] = tz
        return m

    @classmethod
    def scale(cls, sx: float, sy: float, sz: float):
        """缩放矩阵"""
        m = cls()
        m.m[0][0] = sx
        m.m[1][1] = sy
        m.m[2][2] = sz
        return m

    def __mul__(self, other):
        """矩阵乘法"""
        if isinstance(other, Mat4):
            result = Mat4()
            for i in range(4):
                for j in range(4):
                    result.m[i][j] = (
                        self.m[i][0] * other.m[0][j]
                        + self.m[i][1] * other.m[1][j]
                        + self.m[i][2] * other.m[2][j]
                        + self.m[i][3] * other.m[3][j]
                    )
            return result
        elif isinstance(other, Vec4):
            x = self.m[0][0] * other.x + self.m[0][1] * other.y + self.m[0][2] * other.z + self.m[0][3] * other.w
            y = self.m[1][0] * other.x + self.m[1][1] * other.y + self.m[1][2] * other.z + self.m[1][3] * other.w
            z = self.m[2][0] * other.x + self.m[2][1] * other.y + self.m[2][2] * other.z + self.m[2][3] * other.w
            w = self.m[3][0] * other.x + self.m[3][1] * other.y + self.m[3][2] * other.z + self.m[3][3] * other.w
            return Vec4(x, y, z, w)
        raise TypeError(f"不支持 Mat4 与 {type(other)} 相乘")

    def __repr__(self):
        lines = []
        for i in range(4):
            lines.append(f"[{self.m[i][0]:8.4f} {self.m[i][1]:8.4f} {self.m[i][2]:8.4f} {self.m[i][3]:8.4f}]")
        return "\n".join(lines)


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def lerp(a, b, t):
    return a + (b - a) * t
