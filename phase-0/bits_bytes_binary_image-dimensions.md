# Bits, Bytes, Binary Numbers, and Image Dimensions

Everything a computer stores is just a number. A photo, a video frame, a model weight file, and a text character are all the same thing to a machine: a long string of 0s and 1s. In computer vision, knowing how those numbers are laid out is what turns "an image" into rows, columns, and pixel values you can compute on.

---

## Bit (Binary Digit)

### What is a Bit?
A bit is the smallest unit of information. It holds exactly one of two values: `0` or `1`.

### Why "binary"?
Because two states are easy to represent physically with reliable electronic components. A switch is off/on, a magnet is one direction or the other, a charge is present or absent. Two distinct, dependable states beat ten fuzzy ones.

### Easy Example
A light switch is the everyday version of a bit: off (0) or on (1).

```
1 light switch  ->  2 possible values
2 light switches -> 2 x 2 = 4 possible values
n light switches -> 2^n possible values
```

### Note
One bit carries very little information. Meaning comes from **combinations** of bits, not from any single bit.

---

## Byte

### What is a Byte?
A byte is a group of 8 bits. It is the standard addressable unit of storage.

```
1 bit    = 2 values           (0-1)
1 byte   = 8 bits             = 2^8 = 256 values   (0-255)
1 KiB    = 1024 bytes
1 MiB    = 1024 KiB = 1,048,576 bytes
1 GiB    = 1024 MiB
```

### Why 8 bits and not some other number?
1. 256 values is enough to cover the printable ASCII character set.
2. It matches early computer hardware word boundaries, so a byte could be fetched with a single instruction.
3. Once chosen, the convention stuck and became universal.

### Easy Example
The ASCII code for `A` is 65 in decimal, which is `0100 0001` in binary, which is one byte.

```
'A'  =  65  =  0b01000001
```

### Note
Bit is about *information*, byte is about *storage and addressing*. Confusing the two is a common early mistake.

---

## Binary Numbers

### Positional Notation
Decimal works because of place value: the rightmost digit is worth 1, the next 10, then 100.

```
Decimal: 1 2 3 4  ->  1*100 + 2*10 + 3*1 + 4  =  123
```

Binary works the same way, with base 2: each place is worth a power of 2.

```
Binary: 1 0 1 1  ->  1*8 + 0*4 + 1*2 + 1*1  =  11
```

### Place Value Table
```
Place:        128  64  32  16  8   4   2   1
Weight:       2^7  2^6  2^5  2^4 2^3 2^2 2^1 2^0
```

### Easy Example: Converting 13 to binary
Take the largest power of 2 that fits, then subtract and repeat.

```
13
-8  (2^3)  -> 1, remainder 5
-4  (2^2)  -> 1, remainder 1
-2  (2^1)  -> 0, remainder 1
-1  (2^0)  -> 1, remainder 0

Result: 1101
```

### Decimal to Binary Table (Keep This Handy)
```
0  -> 0000        8  -> 1000
1  -> 0001        9  -> 1001
2  -> 0010       10  -> 1010
3  -> 0011       11  -> 1011
4  -> 0100       12  -> 1100
5  -> 0101       13  -> 1101
6  -> 0110       14  -> 1110
7  -> 0111       15  -> 1111
```

### Binary in Python
```python
format(13, "b")    # '1101'
bin(13)            # '0b1101'
0b1101            # 13
13.bit_length()   # 4
```

### Binary Addition
```
  0110
+ 0011
-----
  1001
```
Right to left: 0+1=1, 1+1=10 (write 0, carry 1), 1+0+1=10 (write 0, carry 1), 0+0+1=1. Result `1001` = 9, which is 6+3.

### Note
Bit patterns are only meaningful with an agreed interpretation. `1101` is 13 in unsigned, -3 in two's complement, and 13.0 in a float. The same bytes mean different things depending on the data type.

---

## Hexadecimal (Base 16)

### Why Learn Hex?
Hex is base 16: digits `0-9` then `A-F`. It is a *shorthand for bytes*, not a new idea.

One hex digit = 4 bits. Two hex digits = 8 bits = 1 byte. That makes hex the natural way humans read byte dumps, binary file headers, and memory addresses.

```
Binary:  1111 1111 1111 1111
Hex:       F    F    F    F
Decimal:   255  255  255  255
```

### Easy Example: Red, Green, Blue
Web colors are written as `#RRGGBB`, which is just hex bytes.

```
Red    = (255,   0,   0)  ->  #FF0000
Green  = (  0, 255,   0)  ->  #00FF00
Blue   = (  0,   0, 255)  ->  #0000FF
White  = (255, 255, 255)  ->  #FFFFFF
Black  = (  0,   0,   0)  ->  #000000
```

### Reading a PNG Header in Hex
```
89 50 4E 47 0D 0A 1A 0A   PNG signature
00 00 00 0D               IHDR chunk length = 13
49 48 44 52               "IHDR"
00 00 01 E0               width  = 0x01E0 = 480
00 00 01 00               height = 0x0100 = 256
```

### Hex in Python
```python
hex(255)         # '0xff'
0xFF             # 255
f"{255:02X}"     # 'FF'
bytes([255, 0, 0])          # b'\xff\x00\x00'
b"\x89PNG".hex()            # '89504e47'
```

### Note
Hex is only shorthand. `0xFF` and `1111 1111` and `255` are the same number in three notations. Use binary when you are thinking about individual bits, hex when reading bytes.

---

## Data Types and Bit Depth

### Bit Depth = Bits Per Value
Bit depth tells you how many bits each individual value (a pixel, a channel sample) gets.

```
bits per value -> distinct levels
1              -> 2    (binary: black/white)
4              -> 16   (16 grayscale levels)
8              -> 256  (standard 8-bit grayscale)
16             -> 65536 (16-bit grayscale, medical imaging, radiometry)
```

### Easy Example
An 8-bit grayscale pixel holds one of 256 intensity values, 0 (black) to 255 (white).

### Two's Complement (Signed Integers)
The negative-number encoding you will meet in image subtraction and mask arithmetic.

```
+3  = 0000 0011
-3  = 1111 1101
```

### Float32 in Vision
Most deep learning frameworks store images and tensors as 32-bit floats. A float32 is 32 bits (4 bytes) and can hold fractional values, which is why pixel values get normalized to something like 0.0 to 1.0.

```
uint8  array  -> values 0..255,      1 byte per value
float32 array -> values ~0..1,       4 bytes per value  (4x the memory)
```

### Note
Memory usage scales directly with bit depth. A float32 conversion of a large image array costs 4x the memory of uint8, and this is a real cause of out-of-memory errors during training.

---

## Unit Conventions (MB vs MiB)

Two different definitions are in common use, and mixing them makes numbers disagree.

```
1 KB = 1000 bytes      (decimal, manufacturers, storage ads)
1 KiB = 1024 bytes     (binary, operating systems, memory)
```

Windows reports in KiB/MiB but labels them "KB"/"MB". This is why a "1 TB" drive shows as 931 GB in the file manager. The difference is small, but it is not zero.

Throughout this note, `MB` means 10^6 bytes and `MiB` means 2^20 bytes, unless a number is quoted directly from a tool.

### Easy Example
```
6,220,800 bytes  =  6.22 MB (decimal)  =  5.93 MiB (binary)
```

### In Python
```python
os.path.getsize("photo.jpg") / 1e6            # MB
os.path.getsize("photo.jpg") / (1024 ** 2)    # MiB
arr.nbytes / 1e6                              # MB
```

### Note
When you report dataset or checkpoint sizes, say which unit you used. "The model is 350 MB" is ambiguous by a factor of up to 5%.

---

## Bitwise Operations

Bits combine with logic gates, not arithmetic. These operate on each bit independently, and they are the fastest operations in any language because they are single CPU instructions.

### The Four Operations
```
Bitwise NOT   ~x     flip every bit
Bitwise AND   x & y  1 only if both are 1
Bitwise OR    x | y  1 if either is 1
Bitwise XOR   x ^ y  1 if exactly one is 1
Shift left    x << n multiply by 2^n
Shift right   x >> n divide by 2^n (floor)
```

### Easy Example
```
  1100
& 1010
-----
  1000     12 AND 10 = 8

  1100
| 1010
-----
  1110     12 OR 10 = 14

  1100
^ 1010
-----
  0110     12 XOR 10 = 6

~ 1100
-----
  0011     NOT 12 (in 4 bits) = 3

1 << 10     = 1024            (this is how KiB is computed in code)
```

### Boolean Mask Operations
In vision, an 8-bit image where every pixel is 0 or 255 is a binary mask. The four operations map directly to set logic on the pixels:

```
AND   -> intersection   pixels where BOTH masks are set
OR    -> union          pixels where EITHER mask is set
XOR   -> difference     pixels in one mask but not both
NOT   -> complement     pixels that are NOT set
```

### Code (OpenCV)
```python
import cv2
import numpy as np

a = np.zeros((10, 10), np.uint8); a[2:6, 2:6] = 255   # square
b = np.zeros((10, 10), np.uint8); b[4:8, 4:8] = 255   # offset square

inter = cv2.bitwise_and(a, b)     # overlap region only
union = cv2.bitwise_or(a, b)      # both squares entirely
diff  = cv2.bitwise_xor(a, b)     # L-shaped non-overlap parts
invert = cv2.bitwise_not(a)       # everything except the square
```

### Easy Example: Alpha Blending
A cutout is stored as a color image plus an 8-bit mask. Per-pixel opacity is a bitwise AND:

```python
color = np.array([[200, 100, 50]], np.uint8)   # orange
alpha = np.array([[128]], np.uint8)            # 50% opacity
blend = cv2.bitwise_and(color, alpha)
```

### In Python
```python
0b1100 & 0b1010        # 0b1000
0b1100 | 0b1010        # 0b1110
0b1100 ^ 0b1010        # 0b0110
~0b1100                # -0b1101  (Python ints are signed, unlimited width)
255 & 0x0F              # 0x0F   keep the low nibble
(1 << 10) - 1          # 0x3FF  a 10-bit mask
```

### Note
`~` in Python returns a negative number because Python integers are signed and unbounded, unlike a fixed-width 8-bit integer. For image work use NumPy's `np.bitwise_not`, which stays inside the dtype. Mixing Python's `~` with NumPy arrays is a common source of confusing results.

---

## File Size vs Image Dimensions

This is the distinction that surprises most beginners. **Dimensions describe the image. File size describes the stored file.** They are different quantities that only sometimes correlate.

### Dimensions
The shape of the pixel grid: height and width, in pixels.

```
A 1920 x 1080 image has 1920 columns and 1080 rows.
```

### Uncompressed Memory Size
The raw pixel data, before any file compression. This is a calculation:

```
bytes = height x width x channels x (bits per channel / 8)
```

### Easy Example
A 1920 x 1080 RGB image with 8 bits per channel:

```
1920 x 1080 x 3 x 1 byte = 6,220,800 bytes = 6.22 MB = 5.93 MiB
```

Same image as float32:

```
1920 x 1080 x 3 x 4 bytes = 24,883,200 bytes = 24.88 MB = 23.73 MiB
```

### Compressed File Size
What actually takes up space on your disk, after the file format's compression:

```
PNG  -> lossless compression, file size is close to raw data, often smaller
JPEG -> lossy compression, file size is chosen by a quality setting
```

A 1920 x 1080 JPEG at quality 95 might be 2 MB. At quality 50, 300 KB. At quality 10, 60 KB. Same dimensions, same pixels-worth-of-information, wildly different file sizes.

### The Comparison Table
```
                Dimensions        Uncompressed size     JPEG file size
1920x1080 RGB  1920x1080x3       6.22 MB               0.06 - 2.5 MB
640x480   RGB  640x480x3         0.92 MB               0.02 - 0.2 MB
512x512   GRAY 512x512x1         0.26 MB               0.01 - 0.1 MB
```

### The Ratio That Matters
```
compression ratio = uncompressed size / compressed size

PNG of a 512x512 grayscale image:  uncompressed 0.26 MB, file 0.25 MB  -> ratio 1.0
JPEG of a photo, 1920x1080 RGB:    uncompressed 6.22 MB, file 0.3 MB  -> ratio 20
JPEG of a 1920x1080 gradient:      uncompressed 6.22 MB, file 0.9 MB  -> ratio 7
```

The same format gives different ratios for different content, because compression exploits redundancy. A flat synthetic image has almost none to remove, so it compresses poorly. A noisy photograph has a lot, so it compresses well.

### Key Relationships
```
More pixels           -> more raw data, usually a bigger file
More channels         -> more raw data (RGB is 3x grayscale)
Higher bit depth      -> more raw data (16-bit is 2x 8-bit)
Lossless compression  -> file size tracks raw size reasonably closely
Lossy compression     -> file size is set by quality, not by pixel count
```

### Case Study: Why File Size Lies
Two images with the same dimensions, one stored lossy and one lossless:

```
image_a.jpg  1920x1080   150 KB   quality 20, heavy blocking and ringing
image_b.png  1920x1080   5.40 MB  lossless, no artifacts
```

Same dimensions. The PNG is 36x larger on disk but reproduces its source pixels exactly, while the JPEG has permanently discarded information. Never infer image quality from file size.

### Case Study: Where Bigger File Does Not Mean Better
File size measures how well the format could exploit redundancy, not how much the image contains.

```
3840x2160 PNG screenshot, flat UI colors   12 MB   large file, little detail
1920x1080 JPEG photo, quality 90           300 KB  small file, lots of detail
```

The PNG is 40x larger yet represents a lower-fidelity capture of a scene with far less visible structure. Size follows compressibility, not importance.

### Bits Per Pixel (bpp)
A single number that summarizes how much data each pixel costs. It is the usual way compression is described in the literature.

```
bpp = file_size_in_bits / (height x width)

1920x1080 JPEG at 300 KB  ->  300*1024*8 / (1920*1080)  =  1.19 bpp
1920x1080 PNG at 5.4 MB   ->  5.4*1024*1024*8 / 2073600 =  21.8 bpp
Raw RGB                    ->  24 bpp exactly
```

An 8-bit grayscale JPEG can go below 1 bpp on a smooth image. Anything much above 24 bpp for RGB means metadata, padding, or a lossless format.

### Resolution, Not Just Pixels
"DPI" (dots per inch) controls print size, not image content. The same 1920x1080 pixels print 6.4 inches wide at 300 dpi and 19.2 inches wide at 100 dpi. This is a third quantity, independent of both dimensions and file size.

```
print size in inches = pixels / dpi
1920 / 300 = 6.4 in
1920 / 100 = 19.2 in
```

### Note
For training pipelines, **uncompressed size is what matters for RAM**, and **compressed size is what matters for storage and download bandwidth**. A DataLoader that decodes JPEG to a 224x224x3 float32 tensor uses 602,112 bytes per image in RAM regardless of the 30 KB the file occupies on disk.

---

## How a Digital Image Becomes a Number Grid

### Step 1: Light becomes an electrical signal
A sensor measures light at each point and converts it to an electrical signal.

### Step 2: Sampling
The signal is measured at discrete points, giving a grid of samples. This is spatial resolution: the width and height.

### Step 3: Quantization
Each sample's continuous signal value is rounded to one of a fixed number of levels. This is bit depth. Quantization is lossy: two slightly different light intensities can collapse to the same stored value.

### Step 4: Storage as an array
The result is a 2D (or 3D) array of integers or floats. Row index, column index, channel index.

### Easy Example: A 2x2 RGB Image
```
        col0        col1
row0    (255,0,0)  (0,255,0)
row1    (0,0,255)   (255,255,0)
```
In NumPy, this is a `2 x 2 x 3` uint8 array. Flattened in row-major order it becomes 12 bytes:
`255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0`

### Reading Real Values with Python
```python
from PIL import Image
import numpy as np

img = Image.open("photo.jpg")
arr = np.array(img)

print(img.size)         # (width, height) -- note the order!
print(arr.shape)        # (height, width, channels) -- different order!
print(arr.dtype)        # uint8
print(arr.nbytes)       # uncompressed size in bytes
print(arr.min(), arr.max())
```

### The Order Trap
`PIL.Image.size` returns `(width, height)`. NumPy arrays are indexed `[row, column]`, i.e. `(height, width)`. Mixing these up transposes every coordinate you compute. This is one of the most common bugs in early CV code.

### Note
Bounding boxes, matrices, and image arrays use different axis conventions. Box formats like `(x, y, w, h)` start from the top-left with x first; arrays index `y` before `x`. Write down the convention of every library you use.

---

## Lossless vs Lossy Compression

### Lossless
Every original pixel value is recoverable. Smaller than raw, but the image contains nothing that was not there.
- **PNG**: the standard choice for masks, diagrams, screenshots, and anything where exact values matter.
- **BMP**: uncompressed, huge, nearly obsolete.
- **WebP lossless**: modern, well supported.

### Lossy
Pixel values are permanently changed to shrink the file. Human perception often cannot tell.
- **JPEG**: the web default. Works in 8x8 blocks, which is why it struggles with sharp edges and text.
- **JPEG 2000**: better quality at low bitrates, less common support.
- **WebP lossy / AVIF**: newer, better compression, growing adoption.

### Easy Example: The JPEG 8x8 Block
JPEG divides an image into 8x8 blocks and quantizes each one. Flat regions compress to almost nothing; blocks straddling a sharp edge become visibly blocky. Crop a JPEG at a non-multiple-of-8 boundary and you can sometimes see the block grid in the shift.

### Quality Settings
```
JPEG quality 100 -> ~2.5 MB   nearly lossless
JPEG quality  95 -> ~1.2 MB   visually lossless
JPEG quality  75 -> ~500 KB   default, mild artifacts
JPEG quality  50 -> ~250 KB   visible artifacts on edges
JPEG quality  10 -> ~60 KB    obviously blocky
```

### When It Matters in Vision
- **Data augmentation**: JPEG compression introduces ringing artifacts near edges. Models trained on high-quality data can fail on aggressively compressed data. This is a real distribution shift.
- **Annotation masks**: JPEG cannot represent a mask with multiple discrete label values safely. Use PNG. Luma-channel bleeding turns label 1 into a mix of 1 and 2 near edges, corrupting ground truth.
- **Datasets**: dataset size affects download time, storage cost, and epoch speed. Compression settings are a real research variable for robustness studies.
- **Bit depth preservation**: 16-bit medical or satellite data must use lossless formats, or the quantization noise becomes clinically or scientifically meaningful.

### Note
Saving a JPEG as JPEG again loses information every time. A pipeline that opens, edits, and re-saves in JPEG degrades the data on every pass. Open once, keep working in memory or lossless, save once.

---

## Bits, Bytes, and Tensor Shapes Together

Every model input is a number whose shape you can compute in bytes.

### ResNet on ImageNet
```
Input:    batch 32 x 3 x 224 x 224 float32
Per image: 3 * 224 * 224 * 4 bytes        =     602,112 bytes = 0.60 MB
Per batch: 32 * 602,112                   = 19,267,584 bytes = 19.27 MB
```

### Resizing Changes Memory
```
Original 4000 x 3000 x 3 float32 = 144,000,000 bytes = 144 MB
Resized   224 x  224 x 3 float32 =     602,112 bytes = 0.60 MB
```
A 239x reduction in memory. This is why resizing early in a pipeline is the single biggest memory optimization available.

### Code
```python
import numpy as np

arr = np.zeros((224, 224, 3), dtype=np.float32)
print(arr.nbytes)                       # 602112
print(arr.nbytes / 1e6)                 # 0.602112 MB
print(arr.nbytes / (1024 ** 2))         # 0.574 MiB
```

### The Formula
```
bytes = product(shape) x (bits per value / 8)
```

### Note
Batch size, image resolution, and data type are your three memory levers. On a limited GPU or CPU budget, cutting any one of them buys training time, and it is almost always better to train on smaller images for longer than on large images that do not fit.

---

## Quick Reference Card

```
UNITS
1 bit        = 2 values                    (0, 1)
1 byte       = 8 bits = 256 values          (0-255)
1 hex digit  = 4 bits
2 hex digits = 1 byte                       0x00 - 0xFF
1 KB/MB/GB   = 1000^1, 1000^2, 1000^3       decimal
1 KiB/MiB/GiB= 1024^1, 1024^2, 1024^3       binary
(1 << 10)    = 1024, so KiB = 1 << 10

BIT DEPTH
1 bit depth    -> 2 levels
4 bit depth    -> 16 levels
8 bit depth    -> 256 levels               (standard 8-bit image)
16 bit depth   -> 65536 levels             (medical, satellite, radiometric)

SIZE FORMULAS
raw bytes   = height x width x channels x (bits per channel / 8)
bpp         = file_size_bits / (height x width)     bits per pixel
compression = raw_size / file_size
RAM decoded = product(shape) x (bits per value / 8)

THREE SIZES, DO NOT CONFUSE
dimensions  = shape of the pixel grid      (unchanged by compression)
file size   = bytes on disk                (set by compression, not pixel count)
memory      = decoded array size in RAM    (what actually consumes RAM)
dpi         = print density, unrelated to pixel dimensions

FORMATS
JPEG = lossy, 8x8 blocks, quality-controlled size
PNG  = lossless, exact values, required for masks and 16-bit data
WebP/AVIF = newer lossy and lossless, smaller than JPEG/PNG

BITWISE  & intersection   | union   ^ difference   ~ complement
MASKS    0 or 255 per pixel, combined with cv2.bitwise_*
```

---

## Self-Check

Answer these without looking back. If any is unclear, re-read that section.

1. How many values fit in 1 byte? In a 16-bit channel? -> 256; 65,536
2. Convert `1111 0000` to decimal. -> 240
3. What is `#00FF00` in decimal RGB? -> (0, 255, 0)
4. An image is 640x480 RGB, 8-bit. Uncompressed size in bytes? -> 921,600
5. Same image as float32. Size? -> 3,686,400 bytes
6. That 640x480 JPEG on disk is 60 KB. Compression ratio, roughly? -> ~15
7. `img.size` returns 480x640. What is `np.array(img).shape`? -> (640, 480, 3)
8. `0b1100 ^ 0b1010` equals? -> `0b0110` (6)
9. Which logical op gives "in mask A but not in mask B"? -> XOR, or `A AND NOT B`
10. Why must segmentation masks be PNG and not JPEG? -> JPEG's lossy chroma subsampling and DCT blur discrete label boundaries, corrupting ground truth
11. You load 1000 JPEGs at 300x300x3 float32. RAM per batch of 1? -> 1,080,000 bytes (~1.08 MB), regardless of the ~30 KB on disk
12. 6.22 MB of pixels displayed on a 300 dpi print is how many inches wide? -> 6,220,800 bytes at 1 byte/pixel = 6.22M pixels; 1920 px / 300 dpi = 6.4 inches wide. Dimensions and dpi are independent of file size.

---

## Common Misconceptions

| Misconception | Reality |
|---|---|
| A bit is a byte | A byte is 8 bits |
| Bigger file means better image | File size is set by compression, not quality |
| Higher resolution means more information | More pixels can mean more redundancy, not more detail |
| PNG and JPEG are interchangeable | Masks and 16-bit data need lossless formats |
| `uint8` and `float32` are the same image | Same values, 4x the memory in float32 |
| Dimensions stay fixed | JPEG has no dimensions; pixels depend on decode resolution |
| Pillow and NumPy agree on order | `size` is `(w, h)`, arrays are `(h, w, c)` |
| MB and MiB are the same | They differ by 2.4% at GB scale, and 5.1% at TB scale |
| More pixels always means a bigger file | A flat synthetic PNG can be larger than a detailed JPEG at the same size |
| `~x` gives a bit-flipped positive number | Python integers are signed, so `~` returns a negative value |
| Dots per inch affect image quality | DPI only sets print size; it does not change stored pixels |
| Bit depth is a property of the format | Bit depth is a property of the data; the same `.png` extension can hold 8-bit, 16-bit, or RGBA data |

---

## Practice Exercises

1. Write a script that creates a 100x100 RGB NumPy array of zeros and prints its shape, dtype, and `nbytes`. Verify the value against the formula.
2. Convert the same image to float32 and to uint8 and compare the memory reported by `nbytes`.
3. Create a 2x2 RGB array by hand, save it as PNG, reopen it, and confirm the values round-trip exactly.
4. Save the same array as JPEG at quality 100, 75, and 10. Print each file size and inspect the pixel values to show what changed.
5. Open a real photograph and report `Image.size`, `arr.shape`, `arr.dtype`, and the ratio of decoded memory to file size.
6. Resize an image to 224x224 and compute the memory reduction factor. Explain where you would put that resize in a training pipeline and why.
7. Convert 0, 1, 127, 128, 254, 255 to binary and back. Explain what breaks if a threshold is written as 0.5 instead of 127/255.
8. Take a JPEG, crop a 50-pixel strip, and save it. Reopen and compare to the original crop. Count how many pixels changed and by how much.
9. Convert `#336699` to a decimal RGB tuple and back. Print the binary of each channel.
10. Make two 10x10 binary masks (0/255) representing offset squares, and use `cv2.bitwise_and`, `or`, `xor`, `not` to produce intersection, union, difference, and complement. Save all four as one image grid and check the shapes.
11. Read the first 16 bytes of a PNG with `open(path, "rb").read(16)` and print them in hex. Confirm you recognize the PNG signature. Then find the width and height in the IHDR chunk.
12. Compute bits per pixel for a JPEG: `filesize * 8 / (width * height)`. Compare the result for a photo and for a synthetic flat image at the same dimensions and explain the difference.
13. Print the same byte count in KB and KiB, and MB and MiB, and note how far apart the two systems drift at 1 GB.

---

## Why This Matters for Computer Vision

- **Debugging**: a shape or dtype error usually traces back to a bits/bytes/shape misunderstanding
- **Memory budgeting**: you cannot plan a training run without estimating tensor memory
- **Data integrity**: wrong format or lossy re-saving silently corrupts ground truth
- **Preprocessing correctness**: normalization, thresholding, and channel scaling all depend on knowing the exact value range
- **Cost awareness**: resolution and compression choices determine storage, bandwidth, and training feasibility
- **Research validity**: uncontrolled compression differences between train and test sets are a form of leakage and distribution shift

---

## Related Topics

- NumPy dtypes, `nbytes`, `astype`, and memory layout
- NumPy array shape conventions and broadcasting
- Pillow and OpenCV image I/O, including RGB vs BGR channel order
- Quantization error and dithering
- Fixed-point and floating-point representations
- Binary file structure: headers, magic numbers, and reading raw bytes
- Color spaces and why HSV values are not the same as RGB values
- Image memory mapping for large datasets
- Network transfer size versus decoded size

---

## Referral Material

- Python docs: built-in functions, `int.bit_length`, `int.to_bytes`, `struct` module for packing/unpacking binary data
- NumPy docs: `dtype`, `nbytes`, `bitwise_and` / `bitwise_or` / `bitwise_xor` / `bitwise_not`, `astype`
- Pillow docs: `Image.open`, `Image.size`, `Image.mode`, saving with a `quality` argument
- OpenCV docs: `imread`/`imwrite`, `bitwise_and` / `bitwise_or` / `bitwise_xor` / `bitwise_not`
- Wikipedia: [Bit](https://en.wikipedia.org/wiki/Bit), [Byte](https://en.wikipedia.org/wiki/Byte), [Data compression](https://en.wikipedia.org/wiki/Data_compression), [Lossy compression](https://en.wikipedia.org/wiki/Lossy_compression)
- Szeliski, *Computer Vision: Algorithms and Applications*, section on image representation (free online)

---

*This note establishes the numerical foundation for reading, storing, and reasoning about image data.*
