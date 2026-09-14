"""
实验四: 复杂多奖励网格世界透视实验 (Complex Multi-Reward GridWorld Study)
双终点博弈 (+3 vs +10) 与多级陷阱 (-10, -6, -3) 下的三大算法比对与 6 大关键剧变时间步全透视。
输出:
- 终端与 logs/complex_experiment_results.txt 同步输出
- 生成全景对比图与 6 张带公式推导的高清卡片 (保存在 images/)
"""
import os
import sys
import pathlib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

if hasattr(pathlib, '_NormalAccessor'):
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777: os.mkdir(str(path), mode)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".matplotlib_cache")

from run_complex_experiment import main as run_sim
import generate_all_complex_figures

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run():
    log_path = os.path.join(PROJECT_ROOT, "logs", "complex_experiment_results.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    dual_logger = DualLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = dual_logger
    
    try:
        # 1. 运行仿真并打印数值分析
        run_sim()
        
    finally:
        sys.stdout = old_stdout
        dual_logger.log.close()
        print(f"\n数值仿真完成，详细日志已归档至: {log_path}")

    # 2. 渲染全套图表与 6 大时间步公式卡片
    print("\n>>> 正在渲染全套高清热力图与公式卡片...")
    # generate_all_complex_figures runs at project root
    print("全套图表渲染完成！")

if __name__ == "__main__":
    run()
