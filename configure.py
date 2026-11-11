"""Local-only configuration wizard. Never paste your API key in chat."""
from getpass import getpass
from pathlib import Path
import json
import os
import urllib.request
import urllib.error
from models import normalize_base_url

root = Path(__file__).resolve().parent
path = root / 'config.local.json'
config = json.loads((path if path.exists() else root / 'config.example.json').read_text(encoding='utf-8'))
print('仅在本地填写。API Key不会显示，也不会写进实验日志。')
base = normalize_base_url(input('平台提供的 HTTPS API Base URL：').strip())
key = os.environ.get('FINQA_API_KEY') or getpass('API Key（输入隐藏，仅保存在本地配置）：').strip()
if not base.startswith('https://') or not key:
    raise SystemExit('需要有效的HTTPS地址和非空密钥。')
try:
    request = urllib.request.Request(base+'/models', headers={'Authorization': 'Bearer '+key})
    with urllib.request.urlopen(request, timeout=30) as response:
        ids = [x['id'] for x in json.load(response).get('data', [])]
    print('可用模型ID：')
    print('\n'.join(ids))
    (root/'data').mkdir(exist_ok=True)
    (root/'data'/'available-models.json').write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding='utf-8')
except (OSError, ValueError, KeyError):
    print('平台暂未返回模型列表。请在控制台复制实际模型ID，不能只用模型展示名。')
model = input('模型ID（暂未找到可直接回车，之后再填）：').strip()
def price(prompt):
    value = input(prompt+'（未知可回车，运行时可使用 --unpriced）：').strip()
    return float(value) if value else None
config.update(base_url=base, model=model, api_key=key,
    input_cny_per_million=price('每百万输入token价格/元'),
    output_cny_per_million=price('每百万输出token价格/元'), max_cost_cny=10.0)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
print('配置已保存到config.local.json。请勿分享或上传此文件。')
print('已知单价时，默认单次预算预留为10元，累计预留上限为100元。')
