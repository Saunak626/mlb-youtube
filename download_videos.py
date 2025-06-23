import os
import json
import subprocess

# 视频保存目录
save_dir = 'videos/'
os.makedirs(save_dir, exist_ok=True)

# 加载JSON数据文件
with open('data/mlb-youtube-segmented.json', 'r') as f:
    data = json.load(f)

total_videos = len(data)
print(f"总共需要处理 {total_videos} 个视频。")

# 遍历视频条目
# 修复bug：原始JSON是字典，需要遍历其 .values()
for i, entry in enumerate(data.values()):
    yturl = entry['url']
    ytid = yturl.split('=')[-1]

    print(f"\n[{i+1}/{total_videos}] 正在处理视频: {ytid}")

    # 使用yt-dlp进行下载
    # --continue: 断点续传，如果找到部分下载的文件会继续下载
    # --no-overwrites: 如果文件已完整存在，则跳过下载
    # --cookies: 使用导出的cookie文件进行身份验证，避免浏览器锁定问题
    #   - 推荐使用 "Get cookies.txt LOCALLY" 浏览器插件导出youtube.com的cookies
    #   - 将导出的文件命名为 'youtube-cookies.txt' 并放在项目根目录
    cmd = [
        'yt-dlp',
        '--cookies', 'youtube-cookies.txt',
        '--continue',
        '--no-overwrites',
        '--format', 'best[ext=mkv]/best[ext=mp4]/best',
        '--output', os.path.join(save_dir, '%(id)s.%(ext)s'),
        yturl
    ]

    try:
        # 执行下载命令，不再捕获输出，而是让其直接显示在控制台
        # 这样可以看到实时的下载进度，从而判断超时原因
        subprocess.run(cmd, check=True, text=True, timeout=300)
        print(f"  ✓ 处理成功。")

    except subprocess.CalledProcessError as e:
        # 当 yt-dlp 返回错误时，其错误信息已经直接打印到控制台
        # 这里仅记录下载失败的状态
        print(f"  ✗ 下载失败: yt-dlp 报告了一个错误。")
    except subprocess.TimeoutExpired:
        print(f"  ⏰ 下载超时 (5分钟)。可能是网络速度过慢或文件太大。")
    except FileNotFoundError:
        print("  ✗ 命令 'yt-dlp' 未找到。请确保 yt-dlp 已经安装并且在系统的 PATH 中。")
        print("    安装方法: pip install yt-dlp")
    except Exception as e:
        print(f"  ✗ 发生未知错误: {e}")

print("\n=== 所有视频处理完成 ===")