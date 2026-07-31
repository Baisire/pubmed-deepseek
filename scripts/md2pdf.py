"""
md2pdf.py - 将 Markdown 使用说明书转为 HTML 和 PDF
使用 Edge 无头浏览器渲染

使用方式：
    cd scripts
    python md2pdf.py

输出：
    ../docs/使用说明书/使用说明书.html
    ../docs/使用说明书/使用说明书.pdf
"""

import os
import subprocess
import sys

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MD_PATH = os.path.join(PROJECT_DIR, "docs", "使用说明书", "使用说明书.md")
HTML_PATH = os.path.join(PROJECT_DIR, "docs", "使用说明书", "使用说明书.html")
PDF_PATH = os.path.join(PROJECT_DIR, "docs", "使用说明书", "使用说明书.pdf")

# Markdown 转 HTML（简易版，用 Python 内置方式）
def md_to_html(md_content: str) -> str:
    """将 Markdown 转为简单 HTML"""
    import re
    import html as html_lib

    lines = md_content.split('\n')
    html_lines = []
    in_table = False
    in_code = False
    code_lines = []

    for line in lines:
        # 代码块
        if line.strip().startswith('```'):
            if in_code:
                html_lines.append('</code></pre>')
                in_code = False
                code_lines = []
            else:
                html_lines.append('<pre><code>')
                in_code = True
            continue
        if in_code:
            html_lines.append(html_lib.escape(line))
            continue

        # 表格处理
        if '|' in line and line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().split('|')[1:-1]]
            if all(set(c) <= set('-: ') for c in cells):
                continue  # 分隔行
            if not in_table:
                html_lines.append('<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">')
                in_table = True
            cells_html = ''.join(f'<td>{html_lib.escape(c)}</td>' for c in cells)
            html_lines.append(f'<tr>{cells_html}</tr>')
            continue
        elif in_table:
            html_lines.append('</table>')
            in_table = False

        # 标题
        if line.startswith('# '):
            html_lines.append(f'<h1>{html_lib.escape(line[2:])}</h1>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{html_lib.escape(line[3:])}</h2>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{html_lib.escape(line[4:])}</h3>')
        elif line.startswith('#### '):
            html_lines.append(f'<h4>{html_lib.escape(line[5:])}</h4>')
        elif line.startswith('---'):
            html_lines.append('<hr>')
        elif line.startswith('- '):
            html_lines.append(f'<li>{html_lib.escape(line[2:])}</li>')
        elif line.startswith('> '):
            html_lines.append(f'<blockquote>{html_lib.escape(line[2:])}</blockquote>')
        elif line.strip() == '':
            html_lines.append('<br>')
        else:
            # 处理加粗和行内代码
            text = html_lib.escape(line)
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
            html_lines.append(f'<p>{text}</p>')

    if in_table:
        html_lines.append('</table>')

    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>使用说明书 - 医学PubMed检索（Streamlit版）</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.8;
            color: #333;
        }}
        h1 {{ color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        h3 {{ color: #34495e; }}
        table {{ margin: 10px 0; width: 100%; }}
        th, td {{ padding: 8px 12px; }}
        th {{ background: #ecf0f1; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #2980b9; margin-left: 0; padding-left: 15px; color: #666; }}
        li {{ margin: 5px 0; }}
    </style>
</head>
<body>
{chr(10).join(html_lines)}
</body>
</html>"""
    return html_template


def html_to_pdf(html_path: str, pdf_path: str) -> bool:
    """使用 Edge 无头模式将 HTML 转 PDF"""
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_exe = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_exe = p
            break

    if not edge_exe:
        print("[警告] 未找到 Edge 浏览器，跳过 PDF 生成")
        return False

    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={pdf_path}",
        f"file:///{html_path.replace(os.sep, '/')}"
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)
    return os.path.exists(pdf_path)


def main():
    if not os.path.exists(MD_PATH):
        print(f"[错误] 找不到 Markdown 文件：{MD_PATH}")
        sys.exit(1)

    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # MD -> HTML
    html_content = md_to_html(md_content)
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[完成] HTML 已生成：{HTML_PATH}")

    # HTML -> PDF
    if html_to_pdf(HTML_PATH, PDF_PATH):
        print(f"[完成] PDF 已生成：{PDF_PATH}")
    else:
        print("[跳过] PDF 生成失败，请手动使用 HTML 文件")


if __name__ == "__main__":
    main()
