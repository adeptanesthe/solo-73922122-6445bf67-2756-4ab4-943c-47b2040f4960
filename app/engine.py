"""卷积转换核心引擎（纯 NumPy，float64）。

设计要点
========
1. 必读坐标按 tap 级精确裁剪：
   仅当 a[c] != 0 且 K[o,c,u,v] != 0（某个输出通道下该 tap 非零）、
   源像素在图内、且该 tap 服务的输出点在请求窗口内时，该图内像素才必读。
   权重为零的 tap、a=0 的通道、以及只落在补零圈的位置一律不读。

2. 窗口求值只接收“必读像素”的原始值，不构造标准化像素块，也不需要整图数组：
       Z[p,q,o] = bias[o]
                + Σ 图内tap A[o,c,u,v] * X[源r,源c,c]
                + Σ 图内tap t[o,c,u,v]
   常量 t 的求和用有效位掩码按输出点逐 tap 修正，边角/单行单列自动正确。

3. reference_full_image 按原模型定义逐像素标准化、补零、互相关，
   仅供误差核对使用，不参与部署路径。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

# (u, v) 为 3x3 卷积在补零图上的偏移：输出 (p,q) 读取 (p+u, q+v)，u,v ∈ {-1,0,1}
TAP_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


class EngineError(ValueError):
    """引擎输入/求值错误。"""


# --------------------------------------------------------------------------- #
# 基础校验
# --------------------------------------------------------------------------- #
def check_image_shape(H: int, W: int) -> Tuple[int, int]:
    if not isinstance(H, (int, np.integer)) or not isinstance(W, (int, np.integer)):
        raise EngineError("H、W 必须为整数")
    if H <= 0 or W <= 0:
        raise EngineError("整图尺寸 H、W 必须为正整数")
    return int(H), int(W)


def check_window(
    r0: int, c0: int, r1: int, c1: int, H: int, W: int
) -> Tuple[int, int, int, int]:
    """窗口为半开区间 [r0, r1) x [c0, c1)，输出坐标，含边角。"""
    check_image_shape(H, W)
    vals = (r0, c0, r1, c1)
    if any(not isinstance(v, (int, np.integer)) for v in vals):
        raise EngineError("窗口边界必须为整数")
    if not (0 <= r0 < r1 <= H and 0 <= c0 < c1 <= W):
        raise EngineError(
            f"窗口需满足 0<=r0<r1<={H}, 0<=c0<c1<={W}，收到 "
            f"r0={r0}, c0={c0}, r1={r1}, c1={c1}"
        )
    return int(r0), int(c0), int(r1), int(c1)


def _active_channels(pack: Mapping[str, np.ndarray]) -> np.ndarray:
    """a != 0 且至少有一个非零权重 tap 的通道（bool 数组，长度 C）。"""
    a = np.asarray(pack["a"], dtype=np.float64)
    kernel = np.asarray(pack["K"], dtype=np.float64)
    return (a != 0.0) & np.any(kernel != 0.0, axis=(0, 2, 3))


# --------------------------------------------------------------------------- #
# 必读坐标
# --------------------------------------------------------------------------- #
def required_coordinates(
    pack: Mapping[str, np.ndarray], H: int, W: int,
    r0: int, c0: int, r1: int, c1: int,
) -> Dict[str, object]:
    """列出窗口求值所需的全部图内必读坐标。

    Returns
    -------
    dict:
      coords:  [(r, c), ...] 按行优先排序的空间坐标（所有活跃通道共用）
      by_channel: {c: {"index": 通道号, "coords": [...]}}
      active_channels: [c, ...]（a 可为零：a=0 的通道不在其中）
      note: 裁剪规则说明
    """
    r0, c0, r1, c1 = check_window(r0, c0, r1, c1, H, W)
    a = np.asarray(pack["a"], dtype=np.float64)
    kernel = np.asarray(pack["K"], dtype=np.float64)
    C = int(pack["C"])

    active = _active_channels(pack)
    # 每个 (c, u, v) 是否存在某个输出通道的非零权重 tap
    tap_live = np.any(kernel != 0.0, axis=0)  # shape (C, 3, 3)

    per_channel: Dict[int, set] = {c: set() for c in range(C) if active[c]}
    for c in range(C):
        if not active[c]:
            continue
        for ui, (u, v) in enumerate(TAP_OFFSETS):
            if not tap_live[c, ui // 3, ui % 3]:
                continue
            # 该 tap 服务的输出行/列范围（与窗口取交）
            p_lo = max(r0, -u)
            p_hi = min(r1, H - u)
            q_lo = max(c0, -v)
            q_hi = min(c1, W - v)
            if p_lo >= p_hi or q_lo >= q_hi:
                continue
            rs = np.arange(p_lo + u, p_hi + u)
            cs = np.arange(q_lo + v, q_hi + v)
            # 网格去重后入集合
            grid = np.array(np.meshgrid(rs, cs, indexing="ij")).reshape(2, -1).T
            per_channel[c].update(map(tuple, grid.tolist()))

    union_sorted = sorted(set().union(*per_channel.values()) if per_channel else set())
    by_channel = {
        str(c): {"index": c, "coords": sorted(per_channel[c])}
        for c in sorted(per_channel)
    }
    return {
        "coords": union_sorted,
        "by_channel": by_channel,
        "active_channels": [c for c in range(C) if active[c]],
        "count": len(union_sorted),
        "note": (
            "仅包含 a!=0、对应 3x3 tap 权重非零、源点在图内且对应输出点"
            "落在窗口内的图内像素；补零圈与零权重 tap 不读取。"
        ),
    }


# --------------------------------------------------------------------------- #
# 窗口求值（稀疏原始像素，无标准化像素块）
# --------------------------------------------------------------------------- #
def evaluate_window(
    pack: Mapping[str, np.ndarray],
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    pixels: Mapping[Sequence[int], Sequence[float]],
) -> np.ndarray:
    """用原始像素计算窗口响应，形状 (r1-r0, c1-c0, O)，float64。

    pixels: {(r, c): [x_c0, x_c1, ...]}，只需提供 required_coordinates
    返回的坐标；缺少任一必读坐标（含非活跃通道无需提供）即抛 EngineError。

    内存复杂度：临时数组只随**窗口面积** ``(ph+2)*(pw+2)``（tap 源点的
    局部包围盒，ph=r1-r0, pw=c1-c0）与**实收像素数** P 增长，通道数为
    1..4 的常数；绝不分配 H*W 的整图索引表，因此 12000x12000 图上求
    单个右下角响应也只占用 KB 级临时内存。远在窗口外的合法（图内）
    额外像素照常接收，仅不进入局部索引图。
    """
    r0, c0, r1, c1 = check_window(r0, c0, r1, c1, H, W)
    A = np.asarray(pack["A"], dtype=np.float64)        # (O,C,3,3)
    t = np.asarray(pack["t"], dtype=np.float64)        # (O,C,3,3)
    bias = np.asarray(pack["bias"], dtype=np.float64)  # (O,)
    C, O = int(pack["C"]), int(pack["O"])

    req = required_coordinates(pack, H, W, r0, c0, r1, c1)
    req_set = {tuple(rc) for rc in req["coords"]}

    # -- 解析实收像素 -> 排序数组（O(P) 内存），缺失/非法即报错 -------------
    n_px = len(pixels)
    if n_px == 0:
        if req_set:
            raise EngineError(
                f"缺少必读像素坐标，例如 {sorted(req_set)[:5]}，共 {len(req_set)} 个"
            )
        keys = np.zeros((0, 2), dtype=np.int64)
        svals = np.zeros((0, C), dtype=np.float64)
    else:
        try:
            keys = np.asarray([tuple(k) for k in pixels.keys()], dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise EngineError("pixels 的键必须是 [r, c] 二元组") from exc
        if keys.ndim != 2 or keys.shape != (n_px, 2):
            raise EngineError("pixels 的键必须是 [r, c] 二元组")
        svals = np.asarray(list(pixels.values()), dtype=np.float64)
        if svals.shape != (n_px, C):
            raise EngineError(f"每个像素点需要 {C} 个通道值，收到形状 {svals.shape}")
        if np.any(np.abs(svals) > 16.0):
            raise EngineError("像素值绝对值超过 16")

    order = np.lexsort((keys[:, 1], keys[:, 0])) if n_px else np.empty(0, dtype=np.int64)
    skeys = keys[order]
    svals = svals[order]
    # 行优先线性编号；按 (row, col) 排序后必然单调递增
    slinear = skeys[:, 0] * W + skeys[:, 1] if n_px else np.empty(0, dtype=np.int64)

    if n_px:
        # 排序后逐行与上一行比较（np.diff 对多行整数数组会逐元素比较，不能直接用）
        if np.any(np.all(skeys[1:] == skeys[:-1], axis=1)):
            raise EngineError("pixels 中存在重复坐标")
        if np.any((skeys[:, 0] < 0) | (skeys[:, 0] >= H) |
                  (skeys[:, 1] < 0) | (skeys[:, 1] >= W)):
            raise EngineError("pixels 中存在图外坐标")

    req_arr = np.array(sorted(req_set), dtype=np.int64)
    if len(req_arr):
        pos = np.searchsorted(slinear, req_arr[:, 0] * W + req_arr[:, 1])
        found = pos < n_px
        found[found] = (
            skeys[pos[found], 0] == req_arr[found, 0]
        ) & (skeys[pos[found], 1] == req_arr[found, 1])
        if not np.all(found):
            missing = req_arr[~found].tolist()
            raise EngineError(f"缺少必读像素坐标，例如 {missing[:5]}，共 {len(missing)} 个")

    ph, pw = r1 - r0, c1 - c0

    # -- 窗口局部索引图：只覆盖 9 个 tap 可能读到的源点包围盒 --------------
    # 输出行 p∈[r0,r1) 配 u∈{-1,0,1}，源行范围恰为 [r0-1, r1]；列同理。
    # 裁到图内后尺寸 <= (ph+2)*(pw+2)，与整图 H*W 完全无关。
    r_lo, r_hi = max(0, r0 - 1), min(H, r1 + 1)
    c_lo, c_hi = max(0, c0 - 1), min(W, c1 + 1)
    local_idx = np.full((r_hi - r_lo, c_hi - c_lo), -1, dtype=np.int64)
    if n_px:
        inside = (
            (skeys[:, 0] >= r_lo) & (skeys[:, 0] < r_hi)
            & (skeys[:, 1] >= c_lo) & (skeys[:, 1] < c_hi)
        )
        hit_keys = skeys[inside]
        # 写入排序后的像素行号；窗口外的合法额外像素被 inside 滤掉，不占索引图
        local_idx[hit_keys[:, 0] - r_lo, hit_keys[:, 1] - c_lo] = np.nonzero(inside)[0]

    out = np.broadcast_to(bias, (ph, pw, O)).copy()
    rows = np.arange(r0, r1)[:, None]
    cols = np.arange(c0, c1)[None, :]

    for ui, (u, v) in enumerate(TAP_OFFSETS):
        sr = rows + u  # (ph,1)
        sc = cols + v  # (1,pw)
        valid = (sr >= 0) & (sr < H) & (sc >= 0) & (sc < W)  # (ph,pw)
        if not np.any(valid):
            continue  # 整个 tap 只落在补零圈：无权重贡献，也无常量贡献
        vmask = valid.astype(np.float64)
        cst = t[:, :, ui // 3, ui % 3]   # (O,C)
        # 图内 tap 的常量修正：仅图内源点计入 Σ_c t[o,c]。
        # a=0（零缩放）通道的 t=K*b 同样在此计入 —— 其图内偏移不丢。
        out += np.einsum("hw,oc->hwo", vmask, cst)

        wgt = A[:, :, ui // 3, ui % 3]   # (O,C)
        if n_px == 0 or not np.any(wgt):
            # A 整 tap 为 0（零权重或 a=0 通道）：无原始像素项
            continue
        # 越界位置裁剪到局部图内以便取数，其结果由 vmask 精确清零；
        # 合法位置对非零 A tap 必被上面的必读校验保证命中（idx>=0）。
        ridx = np.clip(sr - r_lo, 0, r_hi - r_lo - 1)
        cidx = np.clip(sc - c_lo, 0, c_hi - c_lo - 1)
        hit = local_idx[ridx, cidx]              # (ph,pw)，广播高级索引
        gathered = svals[np.maximum(hit, 0)]     # (ph,pw,C)，-1 借第 0 行占位
        # 图内 tap 的原始像素贡献：Σ_c A[o,c] * X[src,c]，掩码位置零
        out += np.einsum("hwc,oc,hw->hwo", gathered, wgt, vmask)

    # 注：a=0 的通道在 A=K*a 中整列为零，数学上自动无像素贡献，无需特判
    return out


# --------------------------------------------------------------------------- #
# 原模型参考实现（仅校验用：逐像素标准化 + 补零 + 互相关）
# --------------------------------------------------------------------------- #
def reference_full_image(
    pack: Mapping[str, np.ndarray],
    image: np.ndarray,
) -> np.ndarray:
    """按原模型定义在整图上计算，输出 (H, W, O)。

    会显式生成标准化像素块 Y_pad —— 仅用于误差核对，部署路径不调用。
    """
    img = np.asarray(image, dtype=np.float64)
    H, W, C = img.shape
    if C != int(pack["C"]):
        raise EngineError(f"参考图通道数应为 {int(pack['C'])}，收到 {C}")
    if np.any(np.abs(img) > 16.0):
        raise EngineError("像素值绝对值超过 16")

    a = np.asarray(pack["a"], dtype=np.float64)
    b = np.asarray(pack["b"], dtype=np.float64)
    kernel = np.asarray(pack["K"], dtype=np.float64)
    bias = np.asarray(pack["bias"], dtype=np.float64)

    y = img * a[None, None, :] + b[None, None, :]          # 逐像素标准化
    y_pad = np.zeros((H + 2, W + 2, C), dtype=np.float64)  # 外圈补零
    y_pad[1:-1, 1:-1, :] = y
    # 步长 1 的互相关（等价于 K 与 Y_pad 的滑动乘加）
    windows = np.lib.stride_tricks.sliding_window_view(y_pad, (3, 3), axis=(0, 1))
    # windows: (H, W, C, 3, 3) -> 与 K (O,C,3,3) 缩并
    return np.einsum("hwcuv,ocuv->hwo", windows, kernel) + bias[None, None, :]
