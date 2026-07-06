"""
光栅化模块：三角形扫描线光栅化 + 重心坐标插值

核心功能：
  1. 顶点变换（模型 → 世界 → 相机 → 裁剪 → 屏幕）
  2. 背面剔除
  3. Z-Buffer 深度测试
  4. Blinn-Phong 光照着色
  5. 透视校正插值
"""

import math
from soft_renderer.math_util import Vec3, Vec4, Mat4, clamp


class Rasterizer:
    """软光栅渲染器核心"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # 帧缓冲：存储每个像素的颜色
        self.frame_buffer = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

        # Z-Buffer（深度缓冲）：存储每个像素的深度值，初始为无穷远
        self.z_buffer = [[float('inf') for _ in range(width)] for _ in range(height)]

        # MVP 矩阵
        self.model_matrix = Mat4.identity()
        self.view_matrix = Mat4.identity()
        self.projection_matrix = Mat4.identity()

        # 视口变换矩阵（NDC → 屏幕坐标）
        self.viewport_matrix = Mat4.identity()
        self._build_viewport()

        # 光照参数
        self.light_dir = Vec3(0.5, -1.0, 0.3).normalize()  # 平行光方向
        self.light_color = (1.0, 1.0, 1.0)                   # 白光
        self.ambient = 0.15                                  # 环境光强度

        # 模型顶点数据（变换后的屏幕空间顶点）
        self.screen_verts: list[Vec3] = []  # 屏幕坐标
        self.world_verts: list[Vec3] = []   # 世界坐标（用于光照）
        self.world_normals: list[Vec3] = [] # 世界空间法线

    def _build_viewport(self):
        """构建视口变换矩阵：NDC [-1,1] → 屏幕坐标 [0, W]x[0, H]"""
        w2 = self.width / 2.0
        h2 = self.height / 2.0
        self.viewport_matrix = Mat4.identity()
        self.viewport_matrix.m = [
            [w2,  0.0, 0.0, 0.0],
            [0.0, -h2, 0.0, 0.0],  # Y 轴翻转（屏幕 Y 轴向下）
            [0.0, 0.0, 0.5, 0.0],  # 深度映射到 [0, 1]
            [w2,  h2, 0.5, 1.0],
        ]

    def set_view(self, eye: Vec3, center: Vec3, up: Vec3):
        """设置相机"""
        self.view_matrix = Mat4.look_at(eye, center, up)

    def set_projection(self, fov_y: float, near: float, far: float):
        """设置透视投影"""
        aspect = self.width / self.height
        self.projection_matrix = Mat4.perspective(fov_y, aspect, near, far)

    def set_model_transform(self, model_mat: Mat4):
        """设置模型变换"""
        self.model_matrix = model_mat

    def clear(self, color=(0, 0, 0)):
        """清空帧缓冲和深度缓冲"""
        for y in range(self.height):
            for x in range(self.width):
                self.frame_buffer[y][x] = color
                self.z_buffer[y][x] = float('inf')

    def vertex_shader(self, positions: list[Vec3], normals: list[Vec3]) -> None:
        """
        顶点着色器：对每个顶点执行 MVP 变换
        管线：模型空间 → 世界空间 → 相机空间 → 裁剪空间 → NDC → 屏幕空间
        """
        mvp = self.model_matrix * self.view_matrix * self.projection_matrix

        self.screen_verts = []
        self.world_verts = []
        self.world_normals = []

        for i, pos in enumerate(positions):
            # 世界空间坐标（用于光照计算）
            world_pos = self.model_matrix * Vec4(pos.x, pos.y, pos.z, 1.0)
            world_pos3 = Vec3(world_pos.x, world_pos.y, world_pos.z)
            self.world_verts.append(world_pos3)

            # 世界空间法线
            if i < len(normals):
                n = normals[i]
                world_n = self.model_matrix * Vec4(n.x, n.y, n.z, 0.0)
                self.world_normals.append(Vec3(world_n.x, world_n.y, world_n.z).normalize())
            else:
                self.world_normals.append(Vec3(0, 1, 0))

            # 裁剪空间
            clip = mvp * Vec4(pos.x, pos.y, pos.z, 1.0)

            # 透视除法 → NDC
            ndc = clip.perspective_divide()

            # 视口变换 → 屏幕坐标
            screen = self.viewport_matrix * Vec4(ndc.x, ndc.y, ndc.z, 1.0)
            self.screen_verts.append(Vec3(screen.x, screen.y, screen.z))

    def _barycentric(self, p: Vec3, a: Vec3, b: Vec3, c: Vec3):
        """计算重心坐标"""
        v0 = Vec3(b.x - a.x, b.y - a.y, 0)
        v1 = Vec3(c.x - a.x, c.y - a.y, 0)
        v2 = Vec3(p.x - a.x, p.y - a.y, 0)

        d00 = v0.x * v0.x + v0.y * v0.y
        d01 = v0.x * v1.x + v0.y * v1.y
        d11 = v1.x * v1.x + v1.y * v1.y
        d20 = v2.x * v0.x + v2.y * v0.y
        d21 = v2.x * v1.x + v2.y * v1.y

        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            return -1, -1, -1

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    def _blinn_phong(self, normal: Vec3, view_dir: Vec3) -> tuple[float, float, float]:
        """
        Blinn-Phong 光照模型
        返回 (r, g, b) 颜色
        """
        # 漫反射 (Lambert)
        n_dot_l = normal.dot(self.light_dir)
        diffuse = max(0.0, n_dot_l)

        # 镜面反射 (Blinn-Phong)
        half_vec = (self.light_dir + view_dir).normalize()
        spec = max(0.0, normal.dot(half_vec))
        specular = spec ** 32  # 高光指数

        # 组合
        ambient = self.ambient
        r = clamp(ambient + 0.7 * diffuse + 0.3 * specular, 0, 1)
        g = clamp(ambient + 0.7 * diffuse + 0.3 * specular, 0, 1)
        b = clamp(ambient + 0.7 * diffuse + 0.3 * specular, 0, 1)

        return (r, g, b)

    def fragment_shader(self, u: float, v: float, w: float,
                        v0_idx: int, v1_idx: int, v2_idx: int,
                        obj_color=(0.7, 0.7, 0.8)):
        """
        片元着色器：计算单个像素的最终颜色
        使用透视校正插值法线，然后计算光照
        """
        # 插值世界空间法线
        n0 = self.world_normals[v0_idx]
        n1 = self.world_normals[v1_idx]
        n2 = self.world_normals[v2_idx]
        normal = Vec3(
            u * n0.x + v * n1.x + w * n2.x,
            u * n0.y + v * n1.y + w * n2.y,
            u * n0.z + v * n1.z + w * n2.z,
        ).normalize()

        # 插值世界空间位置
        p0 = self.world_verts[v0_idx]
        p1 = self.world_verts[v1_idx]
        p2 = self.world_verts[v2_idx]
        world_pos = Vec3(
            u * p0.x + v * p1.x + w * p2.x,
            u * p0.y + v * p1.y + w * p2.y,
            u * p0.z + v * p1.z + w * p2.z,
        )

        # 视线方向（从顶点指向相机）
        view_dir = Vec3(0, 0, 1)  # 简化：相机在 +Z 方向

        # Blinn-Phong 光照
        light = self._blinn_phong(normal, view_dir)

        return (
            int(clamp(light[0] * obj_color[0] * 255, 0, 255)),
            int(clamp(light[1] * obj_color[1] * 255, 0, 255)),
            int(clamp(light[2] * obj_color[2] * 255, 0, 255)),
        )

    def draw_triangle(self, v0_idx: int, v1_idx: int, v2_idx: int,
                      color=(180, 180, 200)):
        """
        光栅化单个三角形
        包含：背面剔除、扫描线遍历、Z-Buffer 深度测试、片元着色
        """
        a = self.screen_verts[v0_idx]
        b = self.screen_verts[v1_idx]
        c = self.screen_verts[v2_idx]

        # --- 背面剔除 (Backface Culling) ---
        # 计算屏幕空间三角形法线的 Z 分量，剔除背对相机的面
        edge1 = Vec3(b.x - a.x, b.y - a.y, 0)
        edge2 = Vec3(c.x - a.x, c.y - a.y, 0)
        cross_z = edge1.x * edge2.y - edge1.y * edge2.x
        if cross_z <= 0:  # 逆时针 → 背面，剔除
            return

        # --- 包围盒 ---
        min_x = int(max(min(a.x, b.x, c.x), 0))
        min_y = int(max(min(a.y, b.y, c.y), 0))
        max_x = int(min(max(a.x, b.x, c.x), self.width - 1))
        max_y = int(min(max(a.y, b.y, c.y), self.height - 1))

        # --- 扫描线遍历 ---
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = Vec3(x + 0.5, y + 0.5, 0)
                u, v, w = self._barycentric(p, a, b, c)

                if u < 0 or v < 0 or w < 0:
                    continue

                # 插值深度
                z = u * a.z + v * b.z + w * c.z

                # --- Z-Buffer 深度测试 ---
                if z >= self.z_buffer[y][x]:
                    continue
                self.z_buffer[y][x] = z

                # --- 片元着色 ---
                final_color = self.fragment_shader(u, v, w, v0_idx, v1_idx, v2_idx, color)
                self.frame_buffer[y][x] = final_color

    def draw_model(self, positions: list[Vec3], faces: list,
                   normals: list[Vec3], color=(180, 180, 200)):
        """
        绘制整个模型
        步骤：顶点着色 → 遍历所有三角形 → 光栅化
        """
        # 顶点着色阶段
        self.vertex_shader(positions, normals)

        # 遍历所有三角形面
        for face in faces:
            v0_idx = face[0][0]
            v1_idx = face[1][0]
            v2_idx = face[2][0]
            self.draw_triangle(v0_idx, v1_idx, v2_idx, color)

    def draw_wireframe(self, positions: list[Vec3], faces: list,
                       normals: list[Vec3], color=(0, 255, 0)):
        """线框模式绘制"""
        self.vertex_shader(positions, normals)

        for face in faces:
            v0_idx, v1_idx, v2_idx = face[0][0], face[1][0], face[2][0]
            self._draw_line_screen(self.screen_verts[v0_idx], self.screen_verts[v1_idx], color)
            self._draw_line_screen(self.screen_verts[v1_idx], self.screen_verts[v2_idx], color)
            self._draw_line_screen(self.screen_verts[v2_idx], self.screen_verts[v0_idx], color)

    def _draw_line_screen(self, a: Vec3, b: Vec3, color):
        """Bresenham 画线（屏幕坐标）"""
        x0, y0 = int(a.x), int(a.y)
        x1, y1 = int(b.x), int(b.y)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.frame_buffer[y][x] = color
            if x == x1 and y == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
