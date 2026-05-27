import os
import subprocess
import datetime

# ==================== 配置区域 ====================
# 1. 你的项目绝对路径
PROJECT_PATH = r"D:\Desktop\Link"

# 2. 注释开关：True 表示使用自定义注释，False 表示使用自动生成的通用注释
USE_CUSTOM_COMMENT = True

# 3. 自定义注释内容（当 USE_CUSTOM_COMMENT 为 True 时生效）
CUSTOM_COMMENT = "Before Codex"


# ==================================================

def run_git_cmd(cmd):
    """运行 Git 命令并打印输出"""
    print(f"正在执行: {cmd}")
    # shell=True 允许我们在 Windows 环境下像在终端一样执行命令
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_PATH, capture_output=True, text=True, encoding='utf-8')

    if result.returncode == 0:
        if result.stdout:
            print(result.stdout.strip())
        return True
    else:
        print(f"❌ 错误: {result.stderr.strip()}")
        return False


def auto_git_sync():
    # 确保路径存在
    if not os.path.exists(PROJECT_PATH):
        print(f"错误：路径 {PROJECT_PATH} 不存在！")
        return

    print("🚀 开始自动化 Git 同步流程...")

    # 1. 自动决定注释内容
    if USE_CUSTOM_COMMENT:
        commit_message = CUSTOM_COMMENT
    else:
        # 如果不使用自定义注释，自动生成带时间戳的通用注释，如 "Auto update: 2026-05-27 15:30:00"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Auto update: {current_time}"

    # 2. 执行标准同步三部曲
    # 添加所有更改
    if not run_git_cmd("git add ."):
        return

    # 本地提交
    if not run_git_cmd(f'git commit -m "{commit_message}"'):
        # 有时候如果没有代码改动，git commit 会报错，这里做一个温和的提示
        print("💡 提示：可能没有检测到文件修改，跳过提交。")

    # 推送到云端 (默认推送到 origin 的 main 分支)
    if run_git_cmd("git push origin main"):
        print("\n🎉 同步成功！远程仓库已更新。")
    else:
        print("\n❌ 同步失败，请检查网络或远程仓库状态。")


if __name__ == "__main__":
    auto_git_sync()