import os
import json
import subprocess
import glob
import re


save_dir = 'videos/'
os.makedirs(save_dir, exist_ok=True)

# 网络配置
PROXY_URL = None  # 如果有代理，设置为 "http://127.0.0.1:1080" 等
DNS_SERVERS = ['8.8.8.8', '1.1.1.1', '223.5.5.5']  # 备用 DNS 服务器

def get_youtube_ip(dns_servers):
    """使用指定的DNS服务器解析YouTube的IP地址"""
    print("正在尝试直接解析 YouTube IP 地址...")
    for dns in dns_servers:
        try:
            result = subprocess.run(
                ['nslookup', 'www.youtube.com', dns],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # 将输出分割，只在"Non-authoritative answer:"之后的部分查找IP
                answer_section = result.stdout.split("Non-authoritative answer:")
                if len(answer_section) > 1:
                    # 优先使用IPv4地址
                    matches = re.findall(r'Address:\s*([\d\.]+)', answer_section[1])
                    if matches:
                        ip = matches[0]
                        # 确保我们没有意外地捕获到DNS服务器的IP
                        if ip != dns:
                            print(f"  ✓ 使用 DNS {dns} 成功解析到 IP: {ip}")
                            return ip
                else:
                    # 如果没有"Non-authoritative answer:", 可能是直接返回了，尝试原始匹配，但要更小心
                    matches = re.findall(r'Address:\s*([\d\.]+)', result.stdout)
                    # 过滤掉DNS服务器本身的IP
                    valid_ips = [m for m in matches if m != dns]
                    if valid_ips:
                        ip = valid_ips[0]
                        print(f"  ✓ 使用 DNS {dns} 成功解析到 IP: {ip}")
                        return ip
        except subprocess.TimeoutExpired:
            print(f"  ✗ DNS {dns} 查询超时")
        except FileNotFoundError:
            print("  ✗ 未找到 'nslookup' 命令，无法执行直接IP解析。")
            return None
        except Exception as e:
            print(f"  ✗ DNS {dns} 查询出错: {e}")
    
    print("  ✗ 无法使用任何备用 DNS 解析 YouTube IP。")
    return None

def test_network_connectivity():
    """测试网络连通性"""
    print("正在测试网络连通性...")
    
    # 测试基本网络
    try:
        result = subprocess.run(['ping', '-c', '1', '-W', '3', '8.8.8.8'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  ✓ 基本网络连通正常")
        else:
            print("  ✗ 基本网络连通失败")
            return False
    except subprocess.TimeoutExpired:
        print("  ✗ 网络连接超时")
        return False
    
    # 测试 YouTube 域名解析
    for dns in DNS_SERVERS:
        try:
            result = subprocess.run(['nslookup', 'www.youtube.com', dns], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'Address:' in result.stdout:
                print(f"  ✓ 使用 DNS {dns} 可以解析 YouTube")
                return True
        except:
            continue
    
    print("  ✗ 无法解析 YouTube 域名")
    return False

def check_existing_file(ytid, save_dir):
    """检查是否已存在完整或部分下载的文件"""
    base_path = os.path.join(save_dir, ytid)
    
    # 检查完整文件
    complete_files = glob.glob(f"{base_path}.*")
    video_extensions = ['.mkv', '.mp4', '.webm', '.avi', '.mov']
    
    for file_path in complete_files:
        if any(file_path.endswith(ext) for ext in video_extensions):
            return 'complete', file_path
    
    # 检查部分下载文件（yt-dlp的临时文件格式）
    partial_files = glob.glob(f"{base_path}.*.part") + glob.glob(f"{base_path}.*.part.*")
    if partial_files:
        return 'partial', partial_files[0]
    
    return 'none', None

def build_download_command(yturl, save_dir, yt_ip=None):
    """构建下载命令，可选择使用IP地址直接连接"""
    final_url = yturl
    
    cmd = [
        'yt-dlp',
        '--continue',
        '--no-overwrites',
        '--retries', '3',
        '--fragment-retries', '3',
        '--socket-timeout', '30',
        '--format', 'best[ext=mkv]/best[ext=mp4]/best',
        '--output', os.path.join(save_dir, '%(id)s.%(ext)s'),
        # 添加固定的User-Agent，有时可以避免被屏蔽
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    ]
    
    if yt_ip:
        final_url = yturl.replace('www.youtube.com', yt_ip)
        # 必须同时提供 '--resolve' 参数来让yt-dlp知道IP和域名的映射
        cmd.extend(['--resolve', f'www.youtube.com:443:{yt_ip}'])
        print(f"  ℹ️  将使用IP ({yt_ip}) 尝试直接连接。")

    if PROXY_URL:
        cmd.extend(['--proxy', PROXY_URL])
    
    cmd.append(final_url)
    return cmd

# 主程序开始
print("=== MLB YouTube 视频下载器 ===")

# 尝试获取YouTube IP
YOUTUBE_IP = get_youtube_ip(DNS_SERVERS)

# 测试网络连通性
if not test_network_connectivity():
    print("\n⚠️  网络连通性测试失败！")
    print("建议解决方案：")
    print("1. 检查网络连接")
    print("2. 如果有代理，请在脚本中设置 PROXY_URL")
    print("3. 考虑使用 VPN 或其他网络工具")
    print("\n是否仍要继续尝试下载？(按 Ctrl+C 取消，按 Enter 继续)")
    try:
        input()
    except KeyboardInterrupt:
        print("\n已取消下载")
        exit(0)

with open('data/mlb-youtube-segmented.json', 'r') as f:
    data = json.load(f)
    total_videos = len(data)
    processed = 0
    successful = 0
    
    print(f"\n开始处理 {total_videos} 个视频...")
    
    for entry in data.values():
        yturl = entry['url']
        ytid = yturl.split('=')[-1]
        processed += 1
        
        print(f"\n[{processed}/{total_videos}] 处理视频: {ytid}")
        
        # 检查文件状态
        status, file_path = check_existing_file(ytid, save_dir)
        
        if status == 'complete':
            print(f"  ✓ 已存在完整文件: {os.path.basename(file_path)}")
            successful += 1
            continue
        elif status == 'partial':
            print(f"  ↻ 发现部分下载文件，将断点续传: {os.path.basename(file_path)}")
        else:
            print(f"  ↓ 开始新下载")

        # 构建并执行下载命令
        cmd = build_download_command(yturl, save_dir, YOUTUBE_IP)
        
        try:
            # print(f"  执行命令: {' '.join(cmd[:3])} ...") # 减少日志干扰
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)  # 5分钟超时
            print(f"  ✓ 下载完成")
            successful += 1
        except subprocess.TimeoutExpired:
            print(f"  ⏰ 下载超时 (5分钟)")
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.strip() if e.stderr else ""
            print(f"  ✗ 下载失败: {e}")
            
            if "Connection refused" in error_output:
                print("    错误提示: 连接被拒绝。如果您通过 PROXY_URL 或环境变量设置了代理，")
                print("    请确保代理服务正在运行。如果不想使用代理，请不要设置它们。")
            elif e.stderr and len(error_output) > 0:
                # 只显示错误的关键部分
                error_lines = error_output.split('\n')[-3:]
                for line in error_lines:
                    if line.strip():
                        print(f"    {line.strip()}")
        except KeyboardInterrupt:
            print("\n\n⚠️  下载被用户中断")
            break

print(f"\n=== 下载完成 ===")
print(f"处理视频: {processed}/{total_videos}")
print(f"成功下载: {successful}")
print(f"失败/跳过: {processed - successful}")
