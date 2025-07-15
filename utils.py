import os
import shutil

def move_pdfs(src_dir: str, dst_dir: str) -> None:
    """
    将 src_dir 目录下的所有 .pdf 文件移动到 dst_dir，
    如果 dst_dir 中已存在同名文件则跳过。
    """
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 列出源目录中的所有文件
    for filename in os.listdir(src_dir):
        # 只处理 .pdf 文件（不区分大小写）
        if not filename.lower().endswith('.pdf'):
            continue

        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # 如果目标目录已存在同名文件，跳过
        if os.path.exists(dst_path):
            print(f"跳过已存在文件: {filename}")
            continue

        # 移动文件
        try:
            shutil.move(src_path, dst_path)
            print(f"已移动: {filename}")
        except Exception as e:
            print(f"移动失败 {filename}: {e}")

if __name__ == "__main__":
    source_directory = "/path/to/source_folder"
    destination_directory = "/path/to/destination_folder"
    move_pdfs(source_directory, destination_directory)
