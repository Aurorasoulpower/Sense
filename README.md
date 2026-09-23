# 基于 OpenCV 的灯带识别与目标位移解算

> 对应《传感器组考核题》**题目一**。机器人搭载固定 RGB 相机，识别目标灯带，并根据灯带在图像中的位置变化，解算目标相对相机的位姿与位移。

## 1. 题目要求

**任务要求：**

1. **灯带识别** —— 阈值分割、轮廓提取、几何筛选，准确识别两条细长灯带。
2. **矩形拟合与角点提取** —— 对每条灯带拟合旋转矩形框，提取 4 个角点。
3. **建立目标三维模型** —— 按灯带实际尺寸与相对位置建立三维角点，与图像角点一一对应。
4. **目标位姿解算** —— `solvePnP` 求目标相对相机位姿（rvec / tvec）。
5. **目标位移计算** —— 连续位置差得到位移。

**输出要求：** 实时显示 ① 原始图像 ② 检测灯带 ③ 旋转矩形框 ④ 4 个角点 ⑤ 当前目标位置 X/Y/Z ⑥ 相对初始位置的位移 dX/dY/dZ/Distance。

## 2. 方案总览

```text
RealSense RGB
    ↓
相机内参（pyrealsense2）
    ↓
实时帧
    ↓
白芯掩膜  min(R,G,B) ≥ 120
    ↓
形态学闭运算
    ↓
轮廓 → minAreaRect → 长宽比筛选
    ↓
每根灯带：旋转矩形框 + 4 角点
    ↓
两条灯带几何匹配（长度/宽度/方向/垂直）
    ↓
4 个稳定中心点（灯带顶/底中心）
    ↓
SOLVEPNP_IPPE + 重投影误差
    ↓
位置 X/Y/Z + 位移 dX/dY/dZ/Distance
```

## 3. 文件结构

```text
project/
├── main.py             # 主程序：编排 + 可视化 + 信息面板
├── camera.py           # 相机封装：初始化 / 读取 / 内参 / 释放
├── detector.py         # 单灯带检测：白芯掩膜 → 矩形拟合 → 4 角点
├── geometry.py         # 目标匹配：几何校验 + 四点提取 + 结构校验
├── pose.py             # 位姿解算：PnP + 重投影误差 + 欧拉角
├── exam.md / exam.pdf  # 考核题目（作业要求原文）
├── map.md              # 学习路线（个人备查，非交付物）
├── README.md           # 本文档
└── test/               # 调试工具（不参与主流程）
    ├── hsv_tuner.py        # 红色 HSV 阈值调参工具
    ├── white_tuner.py      # 白芯阈值调参工具
    ├── legacy_pipeline.py  # 旧版单体流水线（历史版本，保留备查）
    └── test.jpg            # 测试样张
```

主目录只保留核心源码与文档，所有调试工具集中在 `test/`。

## 4. 模块说明

| 模块 | 职责 |
|---|---|
| `main.py` | 编排 + 可视化。三个窗口：Camera（原图+灯带框+角点+PnP状态）、Mask（白芯掩膜）、Info（位置/位移）。按 `r` 记录初始位，`Esc` 退出。 |
| `camera.py` | RealSense 封装（640×480 @30fps BGR8），固定曝光/增益/白平衡，`camera_matrix`/`dist_coeffs` 由 `video_stream_profile` 获取。 |
| `detector.py` | `detect_light_bars(img, mode="white")`：白芯掩膜 `min(R,G,B)≥120` → 闭运算 → `findContours` → `minAreaRect` → 长宽比筛选 → `LightBar`。`mode="red"` 保留 HSV 掩膜做 A/B 对比。 |
| `geometry.py` | `match_target(candidates)`：长度/宽度/方向相似 + 中心连线⊥灯带方向；提取每根灯带顶/底中心组成 4 个稳定图像点。 |
| `pose.py` | `OBJECT_POINTS` 为 4 个三维点（灯带中心线端点，mm）；`solve_pose` 用 `SOLVEPNP_IPPE` + 重投影误差；输出 `Pose`。 |

## 5. 作业要求 ↔ 实现

**任务要求：**

| 题目要求 | 实现 | 关键代码 |
|---|---|---|
| (1) 灯带识别 | 白芯掩膜 + 形态学 + 轮廓 + 长宽比筛选 | `detector.detect_light_bars` |
| (2) 矩形拟合与角点提取 | 每根灯带 `minAreaRect` + `boxPoints`，4 角点（共 8） | `detector.LightBar.corners` |
| (3) 建立目标三维模型 | 4 个三维角点与图像点一一对应 | `pose.OBJECT_POINTS` |
| (4) 目标位姿解算 | `solvePnPGeneric(SOLVEPNP_IPPE)` + 重投影误差 | `pose.solve_pose` |
| (5) 目标位移计算 | 当前位姿 − 初始位姿 | `main.py`（`r` 记录初始位） |

**输出要求：**

| 输出要求 | 实现 | 窗口 |
|---|---|---|
| 1. 原始相机图像 | 原图 | Camera |
| 2. 检测得到的灯带 | 白芯掩膜 | Mask |
| 3. 旋转矩形框 | 每根灯带红色旋转矩形 | Camera |
| 4. 灯带对应 4 角点 | 每根灯带 4 角点（c0~c3） | Camera |
| 5. 当前目标位置 X/Y/Z | `pose.position` | Info |
| 6. 位移 dX/dY/dZ/Distance | `pose.position − init_position` | Info |

> **关于「角点」的说明：** 检测阶段每根灯带单独拟合旋转矩形并输出 4 角点（共 8 个）用于显示；PnP 位姿解算使用每根灯带的顶/底中心（4 个更稳定的点），避免细长灯带角点近似共线导致的数值不稳定。

## 6. 调试重大节点

1. **灯带过曝饱和 → 掩膜黑洞 → 白芯方案**
   红色灯带本体过曝饱和成近白，HSV 红色掩膜在本体处形成「黑洞」，矩形拟合不稳。改用白芯掩膜 `min(R,G,B)≥120` 直接框出灯带本体。关键结论：灯芯是「暖白」（R 先饱和、G/B 拖后），判据应为「三通道都够亮」而非「R/G/B 接近白色」。

2. **PnP 解算版本差异**
   `cv2.solvePnP(SOLVEPNP_IPPE)` 在不同 OpenCV 版本返回值不一致（单解 `(3,)` 或双解 `(2,3)`）。改用 `solvePnPGeneric` 显式取回全部解，再选 Z>0 的解。

3. **角点语义调整**
   最初把两条灯带连成一个目标矩形（4 点），不符合「每根灯带独立矩形 + 4 角点」。改为每根灯带单独绘制旋转矩形框 + 4 角点（共 8）；PnP 改用 4 个稳定中心线端点。

4. **输出方式调整**
   位置/位移最初打印在终端，改为独立 `Info` 窗口实时刷新，符合输出要求。

## 7. 运行方法

**依赖：**

```bash
pip install opencv-python numpy pyrealsense2
```

（需正确安装 Intel RealSense SDK）

**运行：**

```bash
python main.py
```

**窗口：** Camera（原图 + 检测叠加）、Mask（白芯掩膜）、Info（位置/位移）。

**按键：** `r` 记录当前位姿为初始位，`Esc` 退出。

## 8. 提交清单

按题目要求，最终提交：

- [ ] 完整程序（本仓库）
- [ ] `README.md`（本文档）
- [ ] 代码运行效果展示视频（`mp4`）
- [ ] 代码实现与文件组织的逻辑性说明（见第 3、4 节）
