"""
OBJ 模型加载器

支持解析 Wavefront .obj 文件中的：
  - v  顶点位置
  - vt 纹理坐标
  - vn 法线
  - f  面（三角形）
"""

from soft_renderer.math_util import Vec3


class Model:
    """存储从 OBJ 文件解析的模型数据"""

    def __init__(self):
        self.positions: list[Vec3] = []   # 顶点位置
        self.texcoords: list[tuple[float, float]] = []  # 纹理坐标
        self.normals: list[Vec3] = []     # 法线
        self.faces: list[tuple[tuple[int, int, int],
                               tuple[int, int, int],
                               tuple[int, int, int]]] = []
        # 每个面是 3 个顶点的索引元组: (pos_idx, tex_idx, norm_idx)

    @classmethod
    def from_obj(cls, filepath: str):
        """从 .obj 文件加载模型"""
        model = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if parts[0] == 'v':  # 顶点位置
                    model.positions.append(Vec3(
                        float(parts[1]), float(parts[2]), float(parts[3])
                    ))
                elif parts[0] == 'vt':  # 纹理坐标
                    model.texcoords.append((
                        float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0
                    ))
                elif parts[0] == 'vn':  # 法线
                    model.normals.append(Vec3(
                        float(parts[1]), float(parts[2]), float(parts[3])
                    ))
                elif parts[0] == 'f':  # 面
                    face_verts = []
                    for v in parts[1:]:
                        indices = v.split('/')
                        pos_idx = int(indices[0]) - 1
                        tex_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else 0
                        norm_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else 0
                        face_verts.append((pos_idx, tex_idx, norm_idx))

                    # 三角剖分（处理四边形面）
                    if len(face_verts) == 3:
                        model.faces.append(tuple(face_verts))
                    elif len(face_verts) == 4:
                        model.faces.append((face_verts[0], face_verts[1], face_verts[2]))
                        model.faces.append((face_verts[0], face_verts[2], face_verts[3]))

        return model

    def compute_normals(self):
        """如果模型没有法线，自动计算面法线"""
        if self.normals:
            return

        # 为每个顶点初始化为零向量
        self.normals = [Vec3(0, 0, 0) for _ in self.positions]
        vertex_face_count = [0] * len(self.positions)

        for face in self.faces:
            v0_idx, v1_idx, v2_idx = face[0][0], face[1][0], face[2][0]
            p0 = self.positions[v0_idx]
            p1 = self.positions[v1_idx]
            p2 = self.positions[v2_idx]

            # 计算面法线
            edge1 = p1 - p0
            edge2 = p2 - p0
            face_normal = edge1.cross(edge2).normalize()

            for idx in (v0_idx, v1_idx, v2_idx):
                self.normals[idx] = self.normals[idx] + face_normal
                vertex_face_count[idx] += 1

        # 平均法线
        for i in range(len(self.normals)):
            if vertex_face_count[i] > 0:
                self.normals[i] = self.normals[i] / vertex_face_count[i]
                self.normals[i] = self.normals[i].normalize()

    @classmethod
    def create_cube(cls):
        """程序化生成立方体（无需 OBJ 文件）"""
        model = cls()

        # 8 个顶点
        s = 0.5  # 半边长
        model.positions = [
            Vec3(-s, -s, -s), Vec3( s, -s, -s), Vec3( s,  s, -s), Vec3(-s,  s, -s),  # 后
            Vec3(-s, -s,  s), Vec3( s, -s,  s), Vec3( s,  s,  s), Vec3(-s,  s,  s),  # 前
        ]
        model.texcoords = [(0, 0), (1, 0), (1, 1), (0, 1)] * 2
        model.normals = []  # 将自动计算

        # 12 个三角形（6 个面）
        indices = [
            (0, 1, 2), (0, 2, 3),  # 后面
            (4, 6, 5), (4, 7, 6),  # 前面
            (0, 3, 7), (0, 7, 4),  # 左面
            (1, 5, 6), (1, 6, 2),  # 右面
            (0, 4, 5), (0, 5, 1),  # 底面
            (3, 2, 6), (3, 6, 7),  # 顶面
        ]
        for a, b, c in indices:
            model.faces.append((
                (a, 0, 0), (b, 0, 0), (c, 0, 0)
            ))

        model.compute_normals()
        return model

    @classmethod
    def create_sphere(cls, segments=16, rings=16):
        """程序化生成球体"""
        model = cls()
        r = 1.0

        # 生成顶点
        for j in range(rings + 1):
            phi = math.pi * j / rings
            for i in range(segments + 1):
                theta = 2.0 * math.pi * i / segments
                x = r * math.sin(phi) * math.cos(theta)
                y = r * math.cos(phi)
                z = r * math.sin(phi) * math.sin(theta)
                model.positions.append(Vec3(x, y, z))
                model.texcoords.append((i / segments, j / rings))

        # 生成三角形索引
        for j in range(rings):
            for i in range(segments):
                a = j * (segments + 1) + i
                b = a + segments + 1
                c = a + 1
                d = b + 1
                model.faces.append(((a, a, a), (b, b, b), (c, c, c)))
                model.faces.append(((c, c, c), (b, b, b), (d, d, d)))

        model.compute_normals()
        return model

    @classmethod
    def create_plane(cls):
        """程序化生成地面平面"""
        model = cls()
        s = 3.0
        model.positions = [
            Vec3(-s, 0, -s), Vec3( s, 0, -s), Vec3( s, 0,  s), Vec3(-s, 0,  s),
        ]
        model.texcoords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        model.faces = [
            ((0, 0, 0), (1, 1, 1), (2, 2, 2)),
            ((0, 0, 0), (2, 2, 2), (3, 3, 3)),
        ]
        model.compute_normals()
        return model


import math
