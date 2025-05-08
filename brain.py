from compressor import compress_image
from decompressor import decompress_image

for i in range(20, 100, 20):
    compress_image("data/sea.png", "data/sea.raw", quality=i)
    decompress_image("data/sea.raw", f"sea/sea {i}.png")
