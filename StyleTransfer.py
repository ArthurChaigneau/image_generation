import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import PIL.Image
import time
import functools
if not tf.config.list_physical_devices('GPU'):
    print("Aucune carte graphique configuré l'entrainement sera extrémement lent, si vous possedez une carte Nvidia, contactez moi pour que je vous aide à installer les librairies nécessaires")

class StyleTransferNetwork:
    def __init__(self, image_principale, image_style, name_file_output):
        self.extractor = None
        self.outputs = None
        self.content_layers = ['block5_conv2']
        self.name_file = name_file_output
        self.style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']
        self.base_image = self.load_img(image_principale)
        self.style_image = self.load_img(image_style)
        self.network = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        self.create_extractor()
        self.create_outputs = self.extractor(self.style_image*255)
        self.stlecontentmodel = StyleContentModel(self.style_layers, self.content_layers)
        self.results = self.stlecontentmodel(tf.constant(self.base_image))
        self.style_targets = self.stlecontentmodel(self.style_image)['style']
        self.content_targets = self.stlecontentmodel(self.base_image)['content']

        self.image = tf.Variable(self.base_image)
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        self.num_content = len(self.content_layers)
        self.num_style = len(self.style_layers)

        self.style_weight=1e-2
        self.content_weight=1e4

    def create_extractor(self):
        self.network.trainable = False
        outputs = [self.network.get_layer(name).output for name in self.style_layers]
        self.extractor = tf.keras.Model([self.network.input], outputs)

    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def load_img(self, image):
        max_dim = 512
        img = tf.io.read_file(image)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content
        loss = style_loss + content_loss
        return loss

    @tf.function
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.stlecontentmodel(image)
            loss = self.style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(self.clip_0_1(image))

    def tensor_to_image(self, tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def fit(self, epochs=10, steps_per_epoch=100):
        start = time.time()
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step(self.image)
                print(".", end='')
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end - start))
        self.tensor_to_image(self.image).save(self.name_file)


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = self.vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [self.gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content':content_dict, 'style':style_dict}

  def gram_matrix(self, input_tensor):
      result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
      input_shape = tf.shape(input_tensor)
      num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
      return result / (num_locations)

  def vgg_layers(self,layer_names):
      vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
      vgg.trainable = False
      outputs = [vgg.get_layer(name).output for name in layer_names]
      model = tf.keras.Model([vgg.input], outputs)
      return model