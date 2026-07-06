"""
软光栅渲染器 - 主程序入口

展示：
  - 旋转的立方体（Blinn-Phong 光照）
  - Z-Buffer 深度测试
  - 线框/实体切换
  - 实时 FPS 显示
"""

import pygame
import math
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soft_renderer.math_util import Vec3, Mat4
from soft_renderer.rasterizer import Rasterizer
from soft_renderer.model import Model

# ─── 窗口设置 ───
WIDTH, HEIGHT = 800, 600
TITLE = "Soft Rasterizer | Blinn-Phong + Z-Buffer"


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)

        # 渲染器
        self.rasterizer = Rasterizer(WIDTH, HEIGHT)

        # 相机设置
        self.rasterizer.set_view(
            eye=Vec3(0, 1.5, 4),
            center=Vec3(0, 0, 0),
            up=Vec3(0, 1, 0),
        )
        self.rasterizer.set_projection(fov_y=60, near=0.1, far=100)

        # 模型
        self.cube = Model.create_cube()
        self.sphere = Model.create_sphere(segments=24, rings=24)

        # 状态
        self.angle = 0.0
        self.wireframe = False
        self.show_sphere = True
        self.running = True

        # FPS 计数器
        self.frame_count = 0
        self.fps_timer = 0
        self.fps = 0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.wireframe = not self.wireframe
                elif event.key == pygame.K_TAB:
                    self.show_sphere = not self.show_sphere
                elif event.key == pygame.K_1:
                    self.rasterizer.light_dir = Vec3(0.5, -1.0, 0.3).normalize()
                elif event.key == pygame.K_2:
                    self.rasterizer.light_dir = Vec3(-0.5, -0.5, 1.0).normalize()
                elif event.key == pygame.K_3:
                    self.rasterizer.light_dir = Vec3(0, -1, 0).normalize()

    def update(self, dt: float):
        self.angle += dt * 45  # 每秒旋转 45 度

        # FPS 计算
        self.frame_count += 1
        self.fps_timer += dt
        if self.fps_timer >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_timer = 0

    def render(self):
        # 清屏
        self.rasterizer.clear((25, 25, 35))

        # 旋转立方体
        rot = Mat4.rotate_y(self.angle) * Mat4.rotate_x(self.angle * 0.7)
        self.rasterizer.set_model_transform(rot)
        model = self.sphere if self.show_sphere else self.cube

        if self.wireframe:
            self.rasterizer.draw_wireframe(
                model.positions, model.faces, model.normals, (100, 255, 100)
            )
        else:
            color = (0.8, 0.6, 0.4) if self.show_sphere else (0.5, 0.6, 0.8)
            self.rasterizer.draw_model(
                model.positions, model.faces, model.normals, color
            )

        # 将帧缓冲绘制到 Pygame 窗口
        self._blit_framebuffer()
        self._draw_hud()

        pygame.display.flip()

    def _blit_framebuffer(self):
        """将帧缓冲直接写入 Pygame surface"""
        fb = self.rasterizer.frame_buffer
        for y in range(HEIGHT):
            for x in range(WIDTH):
                self.screen.set_at((x, y), fb[y][x])

    def _draw_hud(self):
        """绘制界面信息"""
        lines = [
            f"FPS: {self.fps}",
            f"Model: {'Sphere' if self.show_sphere else 'Cube'} | Mode: {'Wireframe' if self.wireframe else 'Solid'}",
            f"Triangles: {len(self.sphere.faces) if self.show_sphere else len(self.cube.faces)}",
            f"Rotation: {self.angle % 360:.0f} deg",
            "",
            "[Space] Toggle Wireframe  [Tab] Switch Model  [1/2/3] Light Dir  [Esc] Quit",
        ]
        for i, text in enumerate(lines):
            surf = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(surf, (10, 10 + i * 18))

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # delta time in seconds
            self.handle_events()
            self.update(dt)
            self.render()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    App().run()
