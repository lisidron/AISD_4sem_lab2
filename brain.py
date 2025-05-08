from compressor import compress_image
from decompressor import decompress_image

for i in range(20, 100, 20):
    compress_image("data/sea.png", "data/sea.raw", quality=i)
    # display_compressed_image("data/Tree.raw")
    decompress_image("data/sea.raw", f"sea/sea {i}.png")

# compress_image("data/Lenna.png", "data/Lenna.raw", quality=1)