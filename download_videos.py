import os
import json
import subprocess
import glob

# 视频保存目录
save_dir = 'videos/'
os.makedirs(save_dir, exist_ok=True)

# 在开始前，先扫描输出目录，创建已下载视频ID的集合，以便快速跳过
print(f"正在扫描 '{save_dir}' 目录以查找已下载的文件...")
downloaded_ids = set()
existing_files = glob.glob(os.path.join(save_dir, '*.*'))
for f_path in existing_files:
    filename = os.path.basename(f_path)
    # 修复bug：只有当文件不是.part文件时，才将其视为已完成下载并跳过
    if not filename.endswith('.part'):
        # 从完整路径中获取文件名 (e.g., RHlEdXq2DuI.mp4)
        # 假设视频ID是文件名的第一部分 (e.g., RHlEdXq2DuI)
        ytid = filename.split('.')[0]
        downloaded_ids.add(ytid)

if downloaded_ids:
    print(f"  ✓ 发现 {len(downloaded_ids)} 个已完整下载的视频。")
else:
    print(f"  ℹ️ 未发现预先下载的视频。")

# 加载JSON数据文件
with open('data/mlb-youtube-segmented.json', 'r') as f:
    data = json.load(f)

# 从所有条目中提取URL，并获取唯一的URL列表
# 原始JSON文件为每个"片段"都提供了一个条目，但多个片段可能来自同一个原始视频
# 因此，我们只下载唯一的视频，避免重复并得到正确的总数
all_urls = [entry['url'] for entry in data.values()]
unique_urls = sorted(list(set(all_urls)))

total_videos = len(unique_urls)
print(f"总共需要下载 {total_videos} 个唯一的原始视频。")

# 遍历唯一的视频URL进行下载
for i, yturl in enumerate(unique_urls):
    ytid = yturl.split('=')[-1]

    print(f"\n[{i+1}/{total_videos}] 正在处理视频: {ytid}")

    # 在发起网络请求前，先通过我们自己的扫描结果检查文件是否存在
    if ytid in downloaded_ids:
        print(f"  ✓ 文件已在本地存在，跳过网络请求。")
        continue

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
        # 将超时时间延长至15分钟 (900秒)
        subprocess.run(cmd, check=True, text=True, timeout=900)
        print(f"  ✓ 处理成功。")

    except subprocess.CalledProcessError as e:
        # 当 yt-dlp 返回错误时，其错误信息已经直接打印到控制台
        # 这里仅记录下载失败的状态
        print(f"  ✗ 下载失败: yt-dlp 报告了一个错误。")
    except subprocess.TimeoutExpired:
        print(f"  ⏰ 下载超时 (15分钟)。可能是网络速度过慢或文件太大。")
    except FileNotFoundError:
        print("  ✗ 命令 'yt-dlp' 未找到。请确保 yt-dlp 已经安装并且在系统的 PATH 中。")
        print("    安装方法: pip install yt-dlp")
    except Exception as e:
        print(f"  ✗ 发生未知错误: {e}")

print("\n=== 所有视频处理完成 ===")