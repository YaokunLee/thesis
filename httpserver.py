import http.server
import socketserver

# 指定要监听的端口号
PORT = 8080

# 创建一个自定义的请求处理类，继承自 http.server.SimpleHTTPRequestHandler
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 获取请求的路径
        path = self.path

        # 根据请求的路径构建本地文件路径
        # 例如，如果请求路径是 "/index.html"，则本地文件路径为 "./index.html"
        local_path = "." + path

        try:
            # 打开文件并发送给客户端
            with open(local_path, 'rb') as file:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(file.read())
        except FileNotFoundError:
            # 如果文件不存在，返回 404 错误
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'File not found')

# 创建一个 HTTP 服务器，并绑定到指定的地址和端口
with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Serving at port {PORT}")
    # 开始监听并处理请求
    httpd.serve_forever()
