import sys
import os
import yt_dlp

# @TODO: Update to claude skill style


def download_video_subtitles(url_or_id, save_path):
    """
    接收视频网址或ID，下载字幕并保存为srt文件
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["zh.*", "en.*"],
        "skip_download": True,
        "outtmpl": f"{save_path}/%(title)s.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegSubtitlesConvertor",
                "format": "srt",
            }
        ],
        "quiet": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
            ydl.download([url_or_id])
        return True
    except Exception as e:
        print(f"\n[错误] 处理失败: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("提示: 请输入视频网址或ID")
        print(f"用法: python3 {os.path.basename(__file__)} <视频地址>")
        sys.exit(1)

    user_input = sys.argv[1]

    if download_video_subtitles(user_input, "./downloads"):
        print("\n[完成] 字幕已存入下载文件夹")
    else:
        sys.exit(1)
