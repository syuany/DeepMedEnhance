import os
from enhancer import ImageEnhancer
from bm3d_covid import BM3DEnhancer
import bm3d
import inspect

input_path = 'data/samples'
output_path = 'data/enhanced'
os.makedirs(output_path, exist_ok=True)

# enhancer = ImageEnhancer('best.pth')
# enhancer.process_directory(input_path, output_path)

# valid_params = inspect.getfullargspec(bm3d.bm3d).args
# print("当前版本有效参数:", valid_params)

bm3dEnhancer = BM3DEnhancer()
bm3dEnhancer.process_directory(input_path, output_path)

