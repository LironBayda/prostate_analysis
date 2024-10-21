import os
import numpy as np

import nibabel as nib
import tensorflow as tf


class DenoiseWithMrGdd():
    def __init__(self,dyn_pet_prefix):
        self.__num_of_unet_blocks, self.__conv_filter = 4, 3

        self.__anat_folder, self.__anat_prefix = 'dynPet', 'mean_pet'
        self.__dyn_pet_folder, self.__dyn_pet_prefix = 'dynPet',dyn_pet_prefix

        self.__win_x1, self.__win_y1, self.__win_z1 = 60, 60, 40

        self.__win_x2, self.__win_y2,self.__win_z2 = 124, 124, 72

        self.__slices = 1

        self.__MR_GDD_x, self.__MR_GDD_y, self.__MR_GDD_z =  64, 64, 32
        self.__MR_GDD_slices =1
        self.__output_suffix = 'nn'
        self.__num_of_epoch =2000

        self.__file_list = None
        self.__affine = None

        self.__noise = None
        self.__y_train = None
        self.__model = None
        self.__X_train = None

    def _conv_block(self,x, conv_filter, alpha):
        c = tf.keras.layers.Conv3D(conv_filter, (3, 3,3), kernel_initializer='he_normal', padding='same',
                                   use_bias=False)(x)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.LeakyReLU(alpha=alpha)(c)
        return c

    def _downsampling_block(self,x, conv_filter, alpha):
        c = tf.keras.layers.Conv3D(conv_filter, (3, 3, 3), strides=(2, 2, 2), kernel_initializer='he_normal',
                                   padding='same',
                                   use_bias=False)(x)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.LeakyReLU(alpha=alpha)(c)
        return c

    
    def encoder_block(self,x, conv_filter, alpha):
        c = self._conv_block(x, conv_filter, alpha)
        c = self._conv_block(c, conv_filter, alpha)
        c = self._downsampling_block(c, conv_filter, alpha)
        return c

    def decoder(self,conv_filter, alpha):
        y = []

        for i in range(self.__num_of_unet_blocks):
            result = tf.keras.Sequential()

            result.add(tf.keras.layers.UpSampling3D((2, 2, 2)))
            for j in range(2):
                result.add(
                    tf.keras.layers.Conv3D(conv_filter , (3, 3, 3),
                                           kernel_initializer='he_normal',
                                           padding='same',
                                           use_bias=False))
                result.add(tf.keras.layers.BatchNormalization())
                result.add(tf.keras.layers.LeakyReLU(alpha=alpha))

            y.append(result)

        return y

    def encoder(self,conv_filter, alpha, anat_shape):
        s = tf.keras.layers.Input(shape=anat_shape)
        x = s
        y = [s]

        for i in range(self.__num_of_unet_blocks):
            x = self._conv_block(x, conv_filter, alpha)
            x = self._conv_block(x,  conv_filter, alpha)
            x = self._downsampling_block(x,conv_filter, alpha)
            y.append(x)

        x = self._conv_block(x, conv_filter, alpha)
        x = self._conv_block(x,  conv_filter, alpha)
        y.append(x)

        model = tf.keras.Model(inputs=s, outputs=y)
      #   plot_model(model, to_file='encoder.png', show_shapes=True, dpi=64)

        return model

    def first_block_deep_decoder(self,s, decoder_layer, conv_filter, alpha):
        conv_layer = self._conv_block(s, conv_filter, alpha)
        fru_model = self._fru(decoder_layer, conv_layer, conv_filter, alpha)
        z = fru_model([decoder_layer, conv_layer])

        return z

    def MR_GDD(self,anat_shape):
        down_stack = self.encoder(self.__conv_filter, 0.3, anat_shape)
        inputs = tf.keras.layers.Input(shape=anat_shape, name="anatomy")
        skips = down_stack(inputs)
        s = tf.keras.layers.Input(skips[-1].shape[1:4] + [1], name="noise")
        z = self.first_block_deep_decoder(
            s, skips[-1], self.__conv_filter , 0.3)

        x = skips[-1]
        skips = reversed(skips[:-2], )
        up_stack = self.decoder(
            self.__conv_filter, 0.3)
        #    DenoiseWithMrGdd.__conv_filter * DenoiseWithMrGdd.__num_of_unet_blocks , 0.3)
       # conv_filter = DenoiseWithMrGdd.__conv_filter * (DenoiseWithMrGdd.__num_of_unet_blocks + 1)
        conv_filter=self.__conv_filter
        for up, skip in zip(up_stack, skips):
            # unet network
            x = up(x)

            # deep decoder
            _uru_model = self._uru(skip, z, conv_filter, 0.3)
            z = _uru_model([skip, z])
            z = self._conv_block(z, conv_filter, 0.3)
            _fru_model = self._fru(x, z, conv_filter, 0.3)
            z = _fru_model([x, z])

            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        z = tf.keras.layers.Conv3D(self.__slices, (1, 1, 1), kernel_initializer='he_normal', padding='same',
                                   use_bias=False)(z)
        model = tf.keras.Model(inputs=[inputs, s], outputs=z)
        return model


    def Unet(self,anat_shape):
        down_stack = self.encoder(self.__conv_filter, 0.3, anat_shape)
        inputs = tf.keras.layers.Input(shape=anat_shape, name="anatomy")
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-2], )
        up_stack = self.decoder(
            self.__conv_filter, 0.3)
        #    DenoiseWithMrGdd.__conv_filter * DenoiseWithMrGdd.__num_of_unet_blocks , 0.3)
       # conv_filter = DenoiseW
        for up, skip in zip(up_stack, skips):
            # unet network
            x = up(x)

            # deep decoder
           
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

     
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    

    def get_img(self,path, save_affine):
        data = nib.load(path)
        if save_affine:
            self.__affine = data.affine
        data = data.get_fdata()
        data=data/np.max(data)
        return data[self.__win_x1:self.__win_x2
        , self.__win_y1:self.__win_y2
        , self.__win_z1:self.__win_z2]

    def set_x_train(self,anatomy_path):
        anatomy = [f for f in os.listdir(os.path.join(anatomy_path, self.__anat_folder)) if
                   f.startswith(self.__anat_prefix)]
        anatomy_window = self.get_img(
            os.path.join(anatomy_path,self.__anat_folder, anatomy[0]), 0)
        X_train = np.zeros([1, self.__MR_GDD_x, self.__MR_GDD_y,
                            self.__MR_GDD_z,  self.__slices])
##        for i in range(self.__slices):
        X_train[0,:,:,:,0] = anatomy_window
        self.__X_train=tf.convert_to_tensor(X_train)

    def set_y_train(self,path):
        self.__file_list = [f for f in os.listdir(os.path.join(path, self.__dyn_pet_folder))
                                        if f.startswith(self.__dyn_pet_prefix)]

        y_train = np.zeros([1, self.__MR_GDD_x, self.__MR_GDD_y,
                            self.__MR_GDD_z, self.__slices])
##
##        for i in range(self.__slices):
        window = self.get_img(os.path.join(
                path, self.__dyn_pet_folder, self.__file_list[0]), 0)
        y_train[0,:,:,:,0] = window

        self.__y_train = tf.convert_to_tensor(y_train)

    def set_noise(self):
        self.__noise = tf.random.uniform(shape=[1] + self.__model.inputs[1].shape[1:5], dtype=tf.dtypes.double)
        print(self.__noise.shape)

    def generator(self):
        for i in range(self.__num_of_epoch):
            print()
            yield self.__X_train, self.__y_train
                 
                  # "noise": DenoiseWithMrGdd.__noise + 1 * DenoiseWithMrGdd.get_guassian_noise(i)}, \

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generator,output_types=(tf.double,tf.double))
                                                # output_types=({"anatomy": tf.double, "noise": tf.double}, tf.double))
        return dataset

    def l2(self,y_true, y_pred):
        return tf.norm(y_true - y_pred, ord=2, axis=None, keepdims=None, name=None
                       )

    @staticmethod
    def get_guassian_noise(self,seed_num):

        guassian_noise = tf.random.normal(
            shape=[1] + self.__model.inputs[1].shape[1:5], mean=0, stddev=1,
            dtype=tf.dtypes.double, seed=seed_num)
        return guassian_noise

    def save_images(self,path):

        img = self.__model.predict([self.__X_train])
        for i in range(self.__slices):
            img_y = nib.load(os.path.join(path, self.__dyn_pet_folder, self.__file_list[0])).get_fdata()*0
            single_img = img[0, :, :, :, i]

            img_y[self.__win_x1:self.__win_x2
            , self.__win_y1:self.__win_y2
            , self.__win_z1:self.__win_z2] = single_img

            array_img = nib.Nifti1Image(img_y.astype(np.float64), self.__affine)
            nib.save(array_img,
                     os.path.join(path, self.__dyn_pet_folder,
                                  self.__output_suffix + self.__file_list[i]))

    def denoise_images(self,path):
        self.__model = self.Unet([self.__MR_GDD_x, self.__MR_GDD_y,
                                         self.__MR_GDD_z, self.__MR_GDD_slices])
        self.set_x_train(path)
        self.set_y_train(path)
##        DenoiseWithMrGdd.set_noise()

        dataset = self.get_dataset()
        self.__model.compile(loss=self.l2)
        self.__model.summary()
        model_history = self.__model.fit(dataset, epochs=self.__num_of_epoch,
                                  steps_per_epoch=1)
        self.save_images(path)

