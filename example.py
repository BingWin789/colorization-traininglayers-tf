import tensorflow as tf
from lab_rgb_cvt import rgb_to_lab
from datalayer import quantilization, rebalanceweight, decode_to_ab

img_path = ''
# read image and convert it to Lab
# H * W * C
rgbimg = tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(img_path), channels=3), tf.float32)
labimg = rgb_to_lab(rgbimg)

abimg = labimg[:, :, 1:3]

# quantilize ab channels
quant_ab = tf.py_func(quantilization, [abimg], tf.float32)
# get weight for loss
lossweight = tf.py_func(rebalanceweight, [quant_ab], tf.float32)

# restore ab channels from quantilization result
new_ab = decode_to_ab(quant_ab)

