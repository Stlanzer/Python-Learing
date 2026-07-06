"""
软光栅渲染器 - 第一课：画线算法 & 三角形光栅化

原理：
  软光栅渲染 = 在 CPU 上模拟 GPU 的渲染管线
  核心步骤：顶点变换 → 图元组装 → 光栅化 → 片元着色

本课内容：
  1. Bresenham 画线算法
  2. 三角形扫描线光栅化
  3. 重心坐标插值

依赖安装：pip install pygame numpy
"""

import pygame
import numpy as np

# ============================================================
# 窗口初始化
# ============================================================
WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("软光栅渲染器 - 三角形光栅化")
clock = pygame.time.Clock()

# ============================================================
# Part 1: 像素绘制工具
# ============================================================

def put_pixel(x: int, y: int, color: tuple[int, int, int]):
    """绘制单个像素"""
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        screen.set_at((x, y), color)


# ============================================================
# Part 2: Bresenham 画线算法
# ============================================================

def draw_line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]):
    """
    Bresenham 直线算法
    核心思想：用整数运算决定每个像素是往 x 方向走还是 y 方向走，
    从而画出最接近理想直线的像素集合。
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1   # x 步进方向
    sy = 1 if y0 < y1 else -1   # y 步进方向
    err = dx - dy               # 决策参数

    x, y = x0, y0
    while True:
        put_pixel(x, y, color)
        if x == x1 and y == y1:
            break
        e2 = err * 2
        if e2 > -dy:            # 往 x 方向走
            err -= dy
            x += sx
        if e2 < dx:             # 往 y 方向走
            err += dx
            y += sy


# ============================================================
# Part 3: 三角形光栅化（扫描线 + 重心坐标）
# ============================================================

def barycentric_coords(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """
    计算点 p 在三角形 abc 中的重心坐标 (u, v, w)
    u + v + w = 1，且当 u,v,w 都在 [0,1] 时 p 在三角形内部
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return -1, -1, -1  # 退化三角形

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w


def draw_triangle(
    v0: tuple[float, float],
    v1: tuple[float, float],
    v2: tuple[float, float],
    color: tuple[int, int, int],
    wireframe: bool = False,
):
    """
    三角形光栅化
    步骤：
      1. 计算包围盒 (bounding box)
      2. 遍历包围盒内每个像素
      3. 用重心坐标判断像素是否在三角形内
      4. 在三角形内的像素进行着色
    """
    a = np.array(v0, dtype=float)
    b = np.array(v1, dtype=float)
    c = np.array(v2, dtype=float)

    if wireframe:
        draw_line(int(a[0]), int(a[1]), int(b[0]), int(b[1]), color)
        draw_line(int(b[0]), int(b[1]), int(c[0]), int(c[1]), color)
        draw_line(int(c[0]), int(c[1]), int(a[0]), int(a[1]), color)
        return

    # 计算包围盒
    min_x = max(int(min(a[0], b[0], c[0])), 0)
    min_y = max(int(min(a[1], b[1], c[1])), 0)
    max_x = min(int(max(a[0], b[0], c[0])), WIDTH - 1)
    max_y = min(int(max(a[1], b[1], c[1])), HEIGHT - 1)

    # 遍历包围盒内每个像素
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = np.array([x + 0.5, y + 0.5])  # 采样点在像素中心
            u, v, w = barycentric_coords(p, a, b, c)

            # 如果重心坐标都在 [0,1]，说明像素在三角形内部
            if u >= 0 and v >= 0 and w >= 0:
                put_pixel(x, y, color)


# ============================================================
# Part 4: 彩色三角形（重心坐标插值）
# ============================================================

def draw_triangle_interpolated(
    v0: tuple[float, float],
    v1: tuple[float, float],
    v2: tuple[float, float],
    c0: tuple[int, int, int],
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
):
    """
    带颜色插值的三角形光栅化
    每个顶点的颜色不同，内部像素颜色 = 重心坐标加权平均
    """
    a = np.array(v0, dtype=float)
    b = np.array(v1, dtype=float)
    c = np.array(v2, dtype=float)

    min_x = max(int(min(a[0], b[0], c[0])), 0)
    min_y = max(int(min(a[1], b[1], c[1])), 0)
    max_x = min(int(max(a[0], b[0], c[0])), WIDTH - 1)
    max_y = min(int(max(a[1], b[1], c[1])), HEIGHT - 1)

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = np.array([x + 0.5, y + 0.5])
            u, v, w = barycentric_coords(p, a, b, c)

            if u >= 0 and v >= 0 and w >= 0:
                # 用重心坐标插值颜色
                r = int(u * c0[0] + v * c1[0] + w * c2[0])
                g = int(u * c0[1] + v * c1[1] + w * c2[1])
                b_val = int(u * c0[2] + v * c1[2] + w * c2[2])
                put_pixel(x, y, (r, g, b_val))


# ============================================================
# Part 5: 主循环 - 演示
# ============================================================

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 清屏
        screen.fill((30, 30, 30))

        # 演示 1: 画线 - 画坐标轴
        draw_line(WIDTH // 2, 0, WIDTH // 2, HEIGHT, (50, 50, 50))  # 垂直中线
        draw_line(0, HEIGHT // 2, WIDTH, HEIGHT // 2, (50, 50, 50))  # 水平中线

        # 演示 2: 线框三角形
        draw_triangle(
            (200, 100), (400, 400), (100, 500),
            (0, 255, 0),
            wireframe=True,
        )

        # 演示 3: 实心三角形（单色填充）
        draw_triangle(
            (600, 150), (750, 450), (450, 500),
            (100, 100, 255),
            wireframe=False,
        )

        # 演示 4: 彩色插值三角形
        draw_triangle_interpolated(
            (350, 150), (500, 350), (250, 400),
            (255, 0, 0),    # 红
            (0, 255, 0),    # 绿
            (0, 0, 255),    # 蓝
        )

        # 绘制文字提示
        font = pygame.font.SysFont("Consolas", 16)
        texts = [
            "Lesson 02: 软光栅渲染 - 三角形光栅化",
            "绿色: 线框三角形 (wireframe)",
            "蓝色: 单色填充三角形",
            "红绿蓝: 重心坐标颜色插值三角形",
            "",
            "核心概念:",
            "1. Bresenham 画线算法 - 整数运算画直线",
            "2. 重心坐标 (Barycentric) - 判断像素是否在三角形内",
            "3. 颜色插值 - 用重心坐标在三角形内平滑过渡颜色",
            "",
            "按 ESC 退出",
        ]
        for i, t in enumerate(texts):
            text_surf = font.render(t, True, (220, 220, 220))
            screen.blit(text_surf, (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
