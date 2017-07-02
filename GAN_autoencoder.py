from keras.models import Sequential
from keras.layers import *
from keras.applications import VGG16
from keras.optimizers import *
import numpy as np
from os import listdir
from os.path import join,isdir,splitext
from random import sample
import cv2
import tensorflow as tf
from keras.callbacks import *
import datetime,time
____graph = tf.get_default_graph()

'''
    Data preparation function
'''
def get_sample_image_batch(encoder,size = 16,filepath = 'sample_frames'):
    # check parameters
    assert size > 0
    assert isdir(filepath), 'file path given doesnt exist'
    # check encoder input shape
    image_shape = encoder.input_shape[1:]
    assert len(image_shape) == 3, 'Invalid encoder input shape, expect 3 dimensions, found {}'.format(len(image_shape))

    # check sample size
    files = [join(filepath,f) for f in sample(listdir(filepath),size) if splitext(f)[1] == '.npy']
    assert len(files) == size, "Unable to get {} sample images, got {} instead".format(size,len(files))

    # prepare and check input
    output_size = tuple(i if i else size for i in encoder.output_shape)
    expected_input_size = (size,) + image_shape
    encoder_input = np.stack([cv2.resize(np.load(f),image_shape[:2]) for f in files])
    assert expected_input_size == encoder_input.shape, 'Input size mismatch. Expected encoder input with shape {}, found {}'.format(expected_input_size,encoder_input.shape)
    # predict and check output
    encoder_output = encoder.predict(encoder_input)
    assert len(encoder_output.shape) == 2, 'Encoder output dimension mismatch. Expected encoder output with dimension {}, found {}'.format(2,len(encoder_output.shape))

    return encoder.predict(encoder_input) + np.random.normal(size = output_size)

def get_generated_image_batch(encoder,generator,size,get_image = False):
    assert size > 0
    encoded_image = get_sample_image_batch(encoder,size)
    generator_output = generator.predict(encoded_image)
    return encoder.predict(generator_output) if not get_image else generator_output

def get_discriminator_training_sample_generator(encoder,generator,ratio = 0.5):
    global ____graph
    assert 0 <= ratio <= 1
    size = 1 # for Keras fit_generator
    # generator.trainable = False
    # generator.compile(loss =  'binary_crossentropy',optimizer = 'adam')
    with ____graph.as_default():
        while True:
                isGenerated = np.random.rand() > ratio
                output = (get_sample_image_batch(encoder,size) if not isGenerated else get_generated_image_batch(encoder,generator,size), np.array(1.).reshape(1,1) if isGenerated else np.array(0.).reshape(1,1))
                yield output

def get_generator_sample_generator(encoder,generator,discriminator):
    global ____graph
    with ____graph.as_default():
        while True:
            output = (get_sample_image_batch(encoder,1), np.array(0.).reshape(1,1))
            yield output

def get_original_image_batch(filepath = 'sample_frames',size = 16):
    files = [join(filepath,f) for f in sample(listdir(filepath),size) if splitext(f)[1] == '.npy']
    assert len(files) == size, "Unable to get {} sample images, got {} instead".format(size,len(files))

    # prepare and check input
    output_size = tuple(i if i else size for i in encoder.output_shape)
    expected_input_size = (size,) + image_shape
    encoder_input = np.stack([cv2.resize(np.load(f),image_shape[:2]) for f in files])
    assert expected_input_size == encoder_input.shape, 'Input size mismatch. Expected encoder input with shape {}, found {}'.format(expected_input_size,encoder_input.shape)
    return encoder_input

# For autoencoder purpose
def get_original_image_sample_generator(filepath = 'sample_frames'):
    assert isdir(filepath), '{} is not a directory'.format(filepath)
    while True:
        batch = get_original_image_batch(filepath,1)
        yield (batch,batch / 255.)
'''
    Models
'''
def get_generator(input_dim,output_dim):
    assert type(input_dim) == int
    assert type(output_dim) == tuple
    from math import sqrt
    model = Sequential()
    model.add(Dense(input_dim * 2, input_shape = (input_dim,),activation = 'relu'))
    dim = int(sqrt(input_dim * 2))
    assert dim ** 2 == input_dim * 2 , 'input dimension must be a square number'
    model.add(Reshape((dim,dim,1)))
    # model.add(UpSampling2D())
    model.add(Conv2D(128,5,activation = 'relu'))
    model.add(BatchNormalization(momentum = 0.9))
    model.add(Conv2D(64,5,activation = 'relu'))
    model.add(BatchNormalization(momentum = 0.9))
    model.add(UpSampling2D())
    model.add(Conv2D(64,5,activation = 'relu'))
    model.add(Conv2D(64,5,activation = 'relu'))
    model.add(Conv2D(64,5,activation = 'relu'))
    model.add(BatchNormalization(momentum = 0.9))

    model.add(Conv2D(32,5,activation = 'relu'))
    model.add(Conv2D(32,5,activation = 'relu'))
    model.add(Conv2D(32,5,activation = 'relu'))
    model.add(BatchNormalization(momentum = 0.9))

    model.add(Conv2D(32,3,activation = 'relu'))
    model.add(Conv2D(16,3,activation = 'relu'))
    model.add(Conv2D(8,3))
    model.add(Conv2D(3,3))
    assert model.output_shape[1:] == output_dim, "Expected generator outputs tensors of shape {}. Gets {}".format(output_dim,model.output_shape[1:])
    return model

def get_encoder(input_dim):
    model = Sequential()
    model.add(Lambda(lambda a: a/255.,input_shape = input_dim,output_shape = input_dim))
    model.add(VGG16(include_top = False,weights = 'imagenet'))
    model.add(Flatten())
    model.trainable = False
    return model

def get_discriminator(encoded_dim):
    model = Sequential()
    model.add(Dense(encoded_dim,input_shape = (encoded_dim,),activation = 'relu'))
    model.add(Dense(encoded_dim * 2,activation = 'relu'))
    model.add(Dense(encoded_dim,activation = 'relu'))
    model.add(Dense(1,activation = 'sigmoid'))
    return model

'''
    Training function
'''
def train_autoencoder(encoder,generator,sample_generator,steps = 5000,callbacks = None):
    ae = Sequential()
    ae.add(encoder)
    ae.add(generator)
    ae.compile(loss = 'mse',optimizer = RMSprop(lr = 0.00005,decay = 1e-6))
    for epoch in range(10):
        try:
            ae.fit_generator(sample_generator,steps_per_epoch = 32, epochs = 10000,callbacks = callbacks or [])
            print "Getting sample from autoencoder"
            ae_sample_output = get_output_from_model(ae,get_original_image_batch,1)[0] * 255
            assert len(ae_sample_output.shape) == 3, 'Dimension of autoencoder sample output is not as expected. Expected dimension {},found {}'.format(3,len(ae_sample_output))
            if not save_sample_image(ae_sample_output,"autoencoder_sample_output"):
                print 'Unable to save autoencoder sample output'
        except KeyboardInterrupt:
            print "early stopped"
            generator.save_weights('generator.h5f')
            #TODO: save checkpoint here
            break

def train_discriminator(discriminator,sample_generator,generator,steps = 5000,test_generator = None,callbacks = None):
    original_generator_weights = generator.get_weights()[0]
    discriminator.fit_generator(sample_generator,steps_per_epoch = 16,epochs = steps,validation_data = test_generator,validation_steps = 16 if test_generator else None,callbacks = callbacks)
    assert np.array_equal(original_generator_weights,generator.get_weights()[0])

def train_generator(encoder,generator,discriminator,sample_generator,steps = 5000,test_generator = None,callbacks = None):
    original_discriminator_weights = discriminator.get_weights()[0]
    original_generator_weights = generator.get_weights()[0]
    adversarial_network = Sequential()
    adversarial_network.add(generator)
    adversarial_network.add(encoder)
    adversarial_network.add(discriminator)
    adversarial_network.layers[-1].trainable = False
    adversarial_network.compile(loss = 'binary_crossentropy',optimizer = RMSprop(lr = 0.005,decay = 1e-8))

    adversarial_network.fit_generator(sample_generator,steps_per_epoch = 16,epochs = steps,validation_data = test_generator,validation_steps = 16 if test_generator else None,callbacks = callbacks)
    # assert not np.array_equal(original_generator_weights,generator.get_weights()[0]), 'Generator is not learning anything'
    assert np.array_equal(original_discriminator_weights,discriminator.get_weights()[0]),'Discriminator learns, but it should not'

'''
    Auxillary function
'''
def get_generator_training_early_stopping_criteria():
    return [
        TerminateOnNaN(),
        EarlyStopping(monitor = 'loss',min_delta = 1e-6,patience = 10),
        ReduceLROnPlateau(monitor = 'loss', patience = 5)
    ]
def get_discriminator_training_early_stopping_criteria():

    return [
        TerminateOnNaN(),
        EarlyStopping(monitor = 'loss',min_delta = 1e-7,patience = 200),
        ReduceLROnPlateau(monitor = 'loss', patience = 10)
    ]
def get_autoencoder_training_callbacks():
    return [
        TerminateOnNaN(),
        EarlyStopping(monitor = 'loss',min_delta = 1e-5,patience = 10),
        ReduceLROnPlateau(monitor = 'loss',patience = 5)
    ]
def get_output_from_model(model,sample_source,num_output):
    data = sample_source(size = num_output)
    return model.predict(data)

def save_sample_image(img,img_title = None):
    return cv2.imwrite('{}-{}.jpg'.format(img_title or "sample_image",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')),img)
'''
    Main
'''
if __name__ == '__main__':
    image_shape = (80,80,3)
    total_round = 100

    encoder = get_encoder(image_shape)
    encoder.compile(loss = 'mse',optimizer = 'adam')
    generator = get_generator(int(encoder.output_shape[1]),image_shape)
    generator.compile(loss = 'binary_crossentropy',optimizer = RMSprop(lr = 0.0001,decay = 1e-6))
    discriminator = get_discriminator(encoder.output_shape[1])
    discriminator.compile(loss = 'binary_crossentropy',optimizer = RMSprop(lr = 0.00001,decay = 1e-6))
    # callbacks
    generator_training_callbacks = get_generator_training_early_stopping_criteria()
    discriminator_training_callbacks = get_discriminator_training_early_stopping_criteria()
    autoencoder_training_callbacks = get_autoencoder_training_callbacks()
    # data generators
    generator_sample_generator = get_generator_sample_generator(encoder,generator,discriminator)
    original_image_sample_generator = get_original_image_sample_generator()
    # check that the discriminator weight is being adjusted
    discriminator_weight_anchor = discriminator.get_weights()[0]
    # training
    train_autoencoder(encoder,generator,original_image_sample_generator,callbacks = autoencoder_training_callbacks)
    # for round in range(1000):
    #     discriminator_sample_generator = get_discriminator_training_sample_generator(encoder,generator,ratio = 0.5)
    #     print 'round {}:'.format(round)
    #
    #     print 'discriminator\'s turn:'
    #     train_discriminator(discriminator,discriminator_sample_generator,generator,1,callbacks = discriminator_training_callbacks or [])
    #
    #     print 'generator\'s turn:'
    #     train_generator(encoder,generator,discriminator,generator_sample_generator,steps = 1,callbacks = generator_training_callbacks or [])
    #
    #     print 'saving some generated images'
    #     generated_image = get_generated_image_batch(encoder,generator,1,True)[0,:,:,:]
    #     cv2.imwrite('generated_image-{}.jpg'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')),generated_image)
    #     assert not np.array_equal(discriminator_weight_anchor,discriminator.get_weights()[0]), 'discriminator is not learning'
