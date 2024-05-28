import urllib.request

# 指定要请求的URL
url = 'http://localhost:8080/index.html'  # 请将URL替换为实际的服务器地址和文件路径

# 发送GET请求
try:
    with urllib.request.urlopen(url) as response:
        # 读取响应内容
        data = response.read()
        
        # 指定本地文件名，可以根据实际情况修改
        local_filename = 'downloaded_file.html'

        # 将响应内容保存到本地文件
        with open(local_filename, 'wb') as local_file:
            local_file.write(data)
            print(f"File '{local_filename}' downloaded successfully.")

except urllib.error.URLError as e:
    print(f"Failed to download file: {e}")
