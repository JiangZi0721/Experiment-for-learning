import os
import sys
import time
from datetime import timedelta

# 配置本地 FlClash 代理端口以保证 Hugging Face 权重下载畅通
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 设置模型缓存目录在 E:\Anaconda\models\faster-whisper，坚决不占用 C 盘
MODEL_DIR = r"E:\Anaconda\models\faster-whisper"
AUDIO_PATH = r"F:\LearningNotes\Transformer\audio_16k.wav"
OUTPUT_MD = r"F:\LearningNotes\Transformer\transcript.md"

def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def main():
    print("=" * 65)
    print("李宏毅《Transformer》课程 - 本地 GPU 极速 ASR 转写流水线")
    print("=" * 65)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[1/4] 模型存储物理路径: {MODEL_DIR}")
    print(f"[2/4] 待转写音频路径:   {AUDIO_PATH}")
    
    if not os.path.exists(AUDIO_PATH):
        print(f"错误: 找不到音频文件 {AUDIO_PATH}")
        sys.exit(1)
        
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("错误: 未检测到 faster_whisper，请确认环境。")
        sys.exit(1)

    print("[3/4] 正在加载 Whisper 模型 (型号: small, 硬件: CUDA, 精度: float16)...")
    start_time = time.time()
    
    # 采用 small 模型（约 480MB），兼顾卓越的中文识别率与毫秒级推理速度
    model = WhisperModel("small", device="cuda", compute_type="float16", download_root=MODEL_DIR)
    
    load_time = time.time() - start_time
    print(f"模型加载完毕，耗时: {load_time:.2f} 秒")

    print("[4/4] 正在启动全量语音识别 (锁定中文，开启 VAD 静音切除)...")
    transcribe_start = time.time()
    
    segments, info = model.transcribe(
        AUDIO_PATH,
        beam_size=5,
        language="zh",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    print(f"音频识别参数: 语种={info.language} (置信度={info.language_probability:.2f}), 总时长={info.duration/60:.2f} 分钟")
    print("正在流水线式流式输出转录文本...\n" + "-" * 65)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# 李宏毅《Transformer》课程完整语音逐字稿 (ASR Transcript)\n\n")
        f.write(f"*音频源文件: `{AUDIO_PATH}`*\n")
        f.write(f"*转写引擎: Faster-Whisper (Model: small, Device: NVIDIA GeForce RTX 4060, Precision: float16)*\n")
        f.write(f"*音频总时长: {info.duration/60:.2f} 分钟 (2970 秒)*\n\n")
        f.write("---\n\n")

        count = 0
        for segment in segments:
            count += 1
            start_str = format_timestamp(segment.start)
            end_str = format_timestamp(segment.end)
            text = segment.text.strip()
            
            line = f"**[{start_str} -> {end_str}]** {text}\n\n"
            f.write(line)
            f.flush()
            
            if count <= 8 or count % 40 == 0:
                print(f"[{start_str}] {text}")

    total_time = time.time() - transcribe_start
    print("-" * 65)
    print(f"全片转录圆满完成！")
    print(f"音频时长: {info.duration:.1f} 秒 | 识别耗时: {total_time:.1f} 秒")
    print(f"推理加速比: {info.duration / total_time:.1f}x (比实时播放快 {info.duration / total_time:.1f} 倍)")
    print(f"最终逐字稿已保存至: {OUTPUT_MD}")

if __name__ == "__main__":
    main()
