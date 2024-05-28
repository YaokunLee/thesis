def create_file(file_size_in_mb):
    file_size = file_size_in_mb * 1024 * 1024  # Convert to bytes
    chunk_size = 1024 * 1024  # 1MB chunk size

    with open('temp', 'wb') as file:
        while file_size > 0:
            chunk = b'\0' * min(file_size, chunk_size)
            file.write(chunk)
            file_size -= chunk_size

    print(f"成功创建大小为 {file_size_in_mb}MB 的文件。")


if __name__ == "__main__":
    size = int(input("请输入文件大小（以MB为单位）："))
    create_file(size)
