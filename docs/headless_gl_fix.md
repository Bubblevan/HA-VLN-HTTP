# 无头服务器上 GL::Context: cannot retrieve OpenGL version 的解决办法

## 原因说明

README 里安装的 apt 包：

```bash
sudo apt-get install -y --no-install-recommends \
  libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
```

提供的是 **Mesa** 的 OpenGL/EGL 库（通用或软件渲染）。这些包是**必要依赖**（头文件、部分符号），但在**无显示器 + NVIDIA GPU** 的服务器上：

- 运行时可能先加载到 **Mesa** 的 `libEGL`/`libGL`，而不是 **NVIDIA** 的；
- Mesa 在无显示/无 X 时往往无法创建有效的 GL 上下文；
- 就会出现：`GL::Context: cannot retrieve OpenGL version` / `InvalidValue` / `Aborted`。

所以：**仅靠上述 apt 包不能解决无头 + NVIDIA 的 GL 问题**，需要让进程使用 **NVIDIA** 的 GL 库。

## 解决办法（NVIDIA GPU 服务器）

在运行 habitat-sim / 验证脚本**之前**设置环境变量，强制预加载 NVIDIA 的库（参见 [habitat-sim#2576](https://github.com/facebookresearch/habitat-sim/issues/2576)）：

```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0
```

然后再运行你的 Python 脚本，例如：

```bash
cd /root/backup/HA-VLN
export PYTHONPATH=/root/backup/HA-VLN/agent/VLN-CE:/root/backup/HA-VLN:$PYTHONPATH
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0
python scripts/verify_havlnce_env.py
```

若 NVIDIA 库不在上述路径，可先查找：

```bash
ldconfig -p | grep -E 'libGLX_nvidia|libGLdispatch'
# 或
ls /usr/lib/x86_64-linux-gnu/libGL*nvidia* /lib/x86_64-linux-gnu/libGL*nvidia* 2>/dev/null
```

把实际路径填进 `LD_PRELOAD` 即可。

## 小结

| 项目 | 说明 |
|------|------|
| README 的 apt 包 | 需要装，提供编译/运行依赖，但无头+NVIDIA 时不足以保证 GL 正常 |
| 无头 + NVIDIA | 需额外设置 `LD_PRELOAD` 使用 NVIDIA 的 libGLX_nvidia / libGLdispatch |
| 无 GPU / 纯 Mesa | 可尝试 `xvfb-run` 提供虚拟显示，或使用 habitat-sim 的 headless 构建 |
