# -*- coding: utf-8 -*-
"""
梯度校验基准脚本 (Numerical Gradient Check Benchmark)
采用高精度中心差分法验证 RNNCell, TimeRNN, TimeAffine, TimeSoftmaxWithLoss 的解析梯度正确性。
"""
import sys
import io
from pathlib import Path

# 强制在 Windows 控制台下使用 UTF-8 编码，杜绝 gbk 编码错误
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.rnn_cells import RNNCell
from src.time_rnn import TimeRNN
from src.gated_rnn import GRUCell, TimeGRU
from src.layers import TimeAffine, TimeSoftmaxWithLoss

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
    console = Console(force_terminal=True)
except ImportError:
    HAS_RICH = False
    console = None


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """双侧中心差分数值梯度计算"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]

        x[idx] = orig_val + eps
        fxh1 = f(x)

        x[idx] = orig_val - eps
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * eps)
        x[idx] = orig_val
        it.iternext()
    return grad


def compute_rel_error(analytic: np.ndarray, numerical: np.ndarray) -> float:
    """计算相对误差: ||a - n|| / max(||a|| + ||n||, eps)"""
    diff = np.linalg.norm(analytic - numerical)
    norm_sum = np.linalg.norm(analytic) + np.linalg.norm(numerical)
    return float(diff / max(norm_sum, 1e-12))


def check_rnn_cell():
    print("=== 正在检验单步 RNNCell 梯度 ===")
    np.random.seed(42)
    N, D, H = 3, 4, 5
    x = np.random.randn(N, D).astype(np.float64)
    h_prev = np.random.randn(N, H).astype(np.float64)
    Wx = np.random.randn(D, H).astype(np.float64)
    Wh = np.random.randn(H, H).astype(np.float64)
    b = np.random.randn(H).astype(np.float64)
    dh_next = np.random.randn(N, H).astype(np.float64)

    cell = RNNCell(Wx.copy(), Wh.copy(), b.copy())
    h = cell.forward(x, h_prev)
    dx_ana, dh_prev_ana = cell.backward(dh_next)
    dWx_ana = cell.dWx.copy()
    dWh_ana = cell.dWh.copy()
    db_ana = cell.db.copy()

    # 损失函数: L = sum(h * dh_next)
    def f_x(x_val):
        c = RNNCell(Wx, Wh, b)
        out = c.forward(x_val, h_prev)
        return np.sum(out * dh_next)

    def f_h(h_val):
        c = RNNCell(Wx, Wh, b)
        out = c.forward(x, h_val)
        return np.sum(out * dh_next)

    def f_Wx(Wx_val):
        c = RNNCell(Wx_val, Wh, b)
        out = c.forward(x, h_prev)
        return np.sum(out * dh_next)

    def f_Wh(Wh_val):
        c = RNNCell(Wx, Wh_val, b)
        out = c.forward(x, h_prev)
        return np.sum(out * dh_next)

    def f_b(b_val):
        c = RNNCell(Wx, Wh, b_val)
        out = c.forward(x, h_prev)
        return np.sum(out * dh_next)

    err_x = compute_rel_error(dx_ana, numerical_gradient(f_x, x))
    err_h = compute_rel_error(dh_prev_ana, numerical_gradient(f_h, h_prev))
    err_Wx = compute_rel_error(dWx_ana, numerical_gradient(f_Wx, Wx))
    err_Wh = compute_rel_error(dWh_ana, numerical_gradient(f_Wh, Wh))
    err_b = compute_rel_error(db_ana, numerical_gradient(f_b, b))

    results = [
        ("RNNCell.dx", err_x),
        ("RNNCell.dh_prev", err_h),
        ("RNNCell.dWx", err_Wx),
        ("RNNCell.dWh", err_Wh),
        ("RNNCell.db", err_b),
    ]
    return results


def check_time_rnn():
    print("=== 正在检验时序 TimeRNN 梯度 ===")
    np.random.seed(42)
    N, T, D, H = 2, 4, 3, 4
    xs = np.random.randn(N, T, D).astype(np.float64)
    Wx = np.random.randn(D, H).astype(np.float64)
    Wh = np.random.randn(H, H).astype(np.float64)
    b = np.random.randn(H).astype(np.float64)
    dhs = np.random.randn(N, T, H).astype(np.float64)

    time_rnn = TimeRNN(Wx.copy(), Wh.copy(), b.copy(), stateful=False)
    hs = time_rnn.forward(xs)
    dxs_ana, dh_prev_chunk = time_rnn.backward(dhs)
    dWx_ana = time_rnn.grads[0].copy()
    dWh_ana = time_rnn.grads[1].copy()
    db_ana = time_rnn.grads[2].copy()

    def f_xs(xs_val):
        layer = TimeRNN(Wx, Wh, b)
        out = layer.forward(xs_val)
        return np.sum(out * dhs)

    def f_Wx(Wx_val):
        layer = TimeRNN(Wx_val, Wh, b)
        out = layer.forward(xs)
        return np.sum(out * dhs)

    def f_Wh(Wh_val):
        layer = TimeRNN(Wx, Wh_val, b)
        out = layer.forward(xs)
        return np.sum(out * dhs)

    def f_b(b_val):
        layer = TimeRNN(Wx, Wh, b_val)
        out = layer.forward(xs)
        return np.sum(out * dhs)

    err_xs = compute_rel_error(dxs_ana, numerical_gradient(f_xs, xs))
    err_Wx = compute_rel_error(dWx_ana, numerical_gradient(f_Wx, Wx))
    err_Wh = compute_rel_error(dWh_ana, numerical_gradient(f_Wh, Wh))
    err_b = compute_rel_error(db_ana, numerical_gradient(f_b, b))

    results = [
        ("TimeRNN.dxs", err_xs),
        ("TimeRNN.dWx", err_Wx),
        ("TimeRNN.dWh", err_Wh),
        ("TimeRNN.db", err_b),
    ]
    return results


def check_time_softmax_with_loss():
    print("=== 正在检验 TimeSoftmaxWithLoss 梯度 ===")
    np.random.seed(42)
    N, T, V = 2, 3, 5
    xs = np.random.randn(N, T, V).astype(np.float64)
    ts = np.random.randint(0, V, size=(N, T))

    loss_layer = TimeSoftmaxWithLoss()
    loss = loss_layer.forward(xs, ts)
    dx_ana = loss_layer.backward()

    def f_xs(xs_val):
        layer = TimeSoftmaxWithLoss()
        return layer.forward(xs_val, ts)

    err_dx = compute_rel_error(dx_ana, numerical_gradient(f_xs, xs))
    return [("TimeSoftmaxWithLoss.dx", err_dx)]


def check_gru_cell():
    print("=== 正在检验门控 Gated RNN (GRUCell) 梯度 ===")
    np.random.seed(42)
    N, D, H = 2, 3, 4
    x = np.random.randn(N, D).astype(np.float64)
    h_prev = np.random.randn(N, H).astype(np.float64)
    Wx = np.random.randn(D, 3 * H).astype(np.float64)
    Wh = np.random.randn(H, 3 * H).astype(np.float64)
    b = np.random.randn(3 * H).astype(np.float64)
    dh_next = np.random.randn(N, H).astype(np.float64)

    cell = GRUCell(Wx.copy(), Wh.copy(), b.copy())
    h = cell.forward(x, h_prev)
    dx_ana, dh_prev_ana = cell.backward(dh_next)
    dWx_ana = cell.grads[0].copy()
    dWh_ana = cell.grads[1].copy()
    db_ana = cell.grads[2].copy()

    def f_x(x_val):
        c = GRUCell(Wx, Wh, b)
        out = c.forward(x_val, h_prev)
        return np.sum(out * dh_next)

    def f_h(h_val):
        c = GRUCell(Wx, Wh, b)
        out = c.forward(x, h_val)
        return np.sum(out * dh_next)

    def f_Wx(Wx_val):
        c = GRUCell(Wx_val, Wh, b)
        out = c.forward(x, h_prev)
        return np.sum(out * dh_next)

    def f_Wh(Wh_val):
        c = GRUCell(Wx, Wh_val, b)
        out = c.forward(x, h_prev)
        return np.sum(out * dh_next)

    def f_b(b_val):
        c = GRUCell(Wx, Wh, b_val)
        out = c.forward(x, h_prev)
        return np.sum(out * dh_next)

    err_x = compute_rel_error(dx_ana, numerical_gradient(f_x, x))
    err_h = compute_rel_error(dh_prev_ana, numerical_gradient(f_h, h_prev))
    err_Wx = compute_rel_error(dWx_ana, numerical_gradient(f_Wx, Wx))
    err_Wh = compute_rel_error(dWh_ana, numerical_gradient(f_Wh, Wh))
    err_b = compute_rel_error(db_ana, numerical_gradient(f_b, b))

    return [
        ("GRUCell.dx", err_x),
        ("GRUCell.dh_prev", err_h),
        ("GRUCell.dWx", err_Wx),
        ("GRUCell.dWh", err_Wh),
        ("GRUCell.db", err_b),
    ]


def check_time_gru():
    print("=== 正在检验时序门控 TimeGRU 梯度 ===")
    np.random.seed(42)
    N, T, D, H = 2, 3, 3, 4
    xs = np.random.randn(N, T, D).astype(np.float64)
    Wx = np.random.randn(D, 3 * H).astype(np.float64)
    Wh = np.random.randn(H, 3 * H).astype(np.float64)
    b = np.random.randn(3 * H).astype(np.float64)
    dhs = np.random.randn(N, T, H).astype(np.float64)

    time_gru = TimeGRU(Wx.copy(), Wh.copy(), b.copy(), stateful=False)
    hs = time_gru.forward(xs)
    dxs_ana, dh_prev_chunk = time_gru.backward(dhs)
    dWx_ana = time_gru.grads[0].copy()
    dWh_ana = time_gru.grads[1].copy()
    db_ana = time_gru.grads[2].copy()

    def f_xs(xs_val):
        layer = TimeGRU(Wx, Wh, b)
        out = layer.forward(xs_val)
        return np.sum(out * dhs)

    def f_Wx(Wx_val):
        layer = TimeGRU(Wx_val, Wh, b)
        out = layer.forward(xs)
        return np.sum(out * dhs)

    def f_Wh(Wh_val):
        layer = TimeGRU(Wx, Wh_val, b)
        out = layer.forward(xs)
        return np.sum(out * dhs)

    def f_b(b_val):
        layer = TimeGRU(Wx, Wh, b_val)
        out = layer.forward(xs)
        return np.sum(out * dhs)

    err_xs = compute_rel_error(dxs_ana, numerical_gradient(f_xs, xs))
    err_Wx = compute_rel_error(dWx_ana, numerical_gradient(f_Wx, Wx))
    err_Wh = compute_rel_error(dWh_ana, numerical_gradient(f_Wh, Wh))
    err_b = compute_rel_error(db_ana, numerical_gradient(f_b, b))

    return [
        ("TimeGRU.dxs", err_xs),
        ("TimeGRU.dWx", err_Wx),
        ("TimeGRU.dWh", err_Wh),
        ("TimeGRU.db", err_b),
    ]


def main():
    all_results = []
    all_results.extend(check_rnn_cell())
    all_results.extend(check_time_rnn())
    all_results.extend(check_gru_cell())
    all_results.extend(check_time_gru())
    all_results.extend(check_time_softmax_with_loss())

    if HAS_RICH:
        table = Table(title="[纯白盒计算图梯度校验 (Gradient Check Results)]", show_header=True, header_style="bold green")
        table.add_column("检验目标", style="bold cyan", width=26)
        table.add_column("相对误差 (Relative Error)", justify="right", width=26)
        table.add_column("数学精确度评级", justify="center", width=22)

        for name, err in all_results:
            if err < 1e-7:
                grade = "[bold green][PASS] 完美匹配 (100%)[/bold green]"
                err_str = f"[green]{err:.2e}[/green]"
            elif err < 1e-5:
                grade = "[green][PASS] 极高精度[/green]"
                err_str = f"[green]{err:.2e}[/green]"
            else:
                grade = "[bold red][FAIL] 异常偏离[/bold red]"
                err_str = f"[red]{err:.2e}[/red]"
            table.add_row(name, err_str, grade)
        console.print(table)
    else:
        for name, err in all_results:
            print(f"{name:26s} | Error: {err:.2e} | {'PASS' if err < 1e-5 else 'FAIL'}")


if __name__ == "__main__":
    main()
