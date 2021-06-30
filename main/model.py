import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Activation,Flatten
from tensorflow.keras.layers import BatchNormalization,Dropout,Concatenate,Add,Lambda
from tensorflow.keras.layers import Conv1D,MaxPooling1D,GlobalMaxPooling1D,UpSampling1D
from tensorflow.keras.layers import Conv2D,GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import Conv2D,GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import GRU,LSTM,SimpleRNN,RNN,Bidirectional
from tensorflow.keras.layers import RepeatVector,TimeDistributed,Cropping1D, ZeroPadding1D
from tensorflow.keras.layers import Conv3D,GlobalAveragePooling3D,MaxPooling3D
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.applications.resnet50 import ResNet50

    
class LSTM_AE(tf.keras.Model):
    
    def __init__(self, N_SEQUENCE, N_FEATURE):
        super(LSTM_AE, self).__init__(name='LSTM_AE')
        self.N_SEQUENCE = N_SEQUENCE  # 1セットのデータ数
        self.N_FEATURE = N_FEATURE    # 入力層・出力層のニューロン数
        
    def build_model(self):
        
        inputs = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
                
        """ Encoder """
        # x = LSTM(100, activation='relu')(inputs)
        x = LSTM(100)(inputs)
        encoded = RepeatVector(self.N_SEQUENCE)(x)
        
        """ Decoder """
        # x = LSTM(100, activation='relu', return_sequences=True)(encoded)
        x = LSTM(100, return_sequences=True)(encoded)
        decoded= TimeDistributed(Dense(self.N_FEATURE))(x)
        
        model = Model(inputs=inputs,outputs=decoded) 
        
        return model
    
    def build_conditional_model(self):
        
        inputs = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
                
        """ Encoder """
        # x = LSTM(100, activation='relu',name="Encoder")(inputs)
        x = LSTM(100, name="Encoder")(inputs)
        encoded = RepeatVector(self.N_SEQUENCE)(x)
        
        reverse_input = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
        x = Concatenate()([encoded, reverse_input])
        
        """ Decoder """
        # x = LSTM(100, activation='relu', return_sequences=True, name="Decoder")(x)
        x = LSTM(100, return_sequences=True, name="Decoder")(x)
        decoded = TimeDistributed(Dense(self.N_FEATURE))(x)
                
        model = Model(inputs=[inputs,reverse_input],outputs=decoded) 
        
        return model


class GRU_AE(tf.keras.Model):

    def __init__(self, N_SEQUENCE, N_FEATURE):
        super(GRU_AE, self).__init__(name='GRU_AE')
        self.N_SEQUENCE = N_SEQUENCE  # 1セットのデータ数
        self.N_FEATURE = N_FEATURE    # 入力層・出力層のニューロン数
    
    def build_model(self):
        
        inputs = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
                
        """ Encoder """
        # x = GRU(100, activation='relu',name="Encoder")(inputs)
        x = GRU(100, name="Encoder")(inputs)
        encoded = RepeatVector(self.N_SEQUENCE)(x)
        
        """ Decoder """
        # x = GRU(100, activation='relu', return_sequences=True, name="Decoder")(encoded)
        x = GRU(100, return_sequences=True, name="Decoder")(encoded)
        decoded = TimeDistributed(Dense(self.N_FEATURE))(x)
                
        model = Model(inputs=inputs,outputs=decoded) 
        
        return model
    
    def build_conditional_model(self):
        
        inputs = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
                
        """ Encoder """
        # x = GRU(100, activation='relu',name="Encoder")(inputs)
        x = GRU(100, name="Encoder")(inputs)
        encoded = RepeatVector(self.N_SEQUENCE)(x)
        
        reverse_input = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
        x = Concatenate()([encoded, reverse_input])
        
        """ Decoder """
        # x = GRU(100, activation='relu', return_sequences=True, name="Decoder")(x)
        x = GRU(100, return_sequences=True, name="Decoder")(x)
        decoded = TimeDistributed(Dense(self.N_FEATURE))(x)
                
        model = Model(inputs=[inputs,reverse_input],outputs=decoded) 
        
        return model
    

class Time_AE(tf.keras.Model):
    
    def __init__(self, N_SEQUENCE, N_FEATURE):
        super(Time_AE, self).__init__(name='Time_AE')
        self.N_SEQUENCE = N_SEQUENCE  # 1セットのデータ数
        self.N_FEATURE = N_FEATURE    # 入力層・出力層のニューロン数
        
    def build_model(self):
        
        inputs = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
        
        """ Encoder """
        x = Conv1D(filters=64, kernel_size=7, padding="same", strides=2, activation="relu")(inputs)
        # x = Dropout(rate=0.2)(x)
        x = Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu")(x)
        # x = Dropout(rate=0.2)(x)
        # x = Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu")(x)

        """ Decoder """
        x = UpSampling1D(size=2)(x)
        x = Conv1D(filters=32, kernel_size=7, padding="same", strides=1, activation="relu")(x)
        # x = Dropout(rate=0.2)(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(filters=64, kernel_size=7, padding="same", strides=1, activation="relu")(x)
        
        x = Conv1D(filters=self.N_FEATURE, kernel_size=3, padding="same", strides=1)(x)
        x = Lambda(lambda x: x[:, :self.N_SEQUENCE], output_shape=(self.N_SEQUENCE,self.N_FEATURE))(x)
        outputs = x
        
        return Model(inputs,outputs) 
    

# TS: time step (t~t+k wave in,  t+k+1 wave out)
class RNN_TS(tf.keras.Model):
    
    def __init__(self, N_SEQUENCE, N_FEATURE):
        super(RNN_TS, self).__init__(name='RNN_TS')
        self.N_SEQUENCE = N_SEQUENCE  # 1セットのデータ数
        self.N_FEATURE = N_FEATURE    # 入力層・出力層のニューロン数
        
    def build_model(self):

        inputs = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
        x = SimpleRNN(128, return_sequences=True, recurrent_dropout=0.0)(inputs)
        x = SimpleRNN(64,  return_sequences=False, recurrent_dropout=0.0)(x)
        outputs = Dense(self.N_FEATURE)(x)
                
        model = Model(inputs,outputs) 
        return model


class LSTM_TS(tf.keras.Model):
    
    def __init__(self, N_SEQUENCE, N_FEATURE):
        super(LSTM_TS, self).__init__(name='LSTM_TS')
        self.N_SEQUENCE = N_SEQUENCE  # 1セットのデータ数
        self.N_FEATURE = N_FEATURE    # 入力層・出力層のニューロン数
        
    def build_model(self):

        inputs = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
        x = LSTM(128, return_sequences=True, recurrent_dropout=0.0)(inputs)
        x = LSTM(64,  return_sequences=False, recurrent_dropout=0.0)(x)
        # x = Dropout(0.4)(x)
        outputs = Dense(self.N_FEATURE)(x)
                
        model = Model(inputs,outputs) 
        return model
        

class GRU_TS(tf.keras.Model):
    
    def __init__(self, N_SEQUENCE, N_FEATURE):
        super(GRU_TS, self).__init__(name='GRU_TS')
        self.N_SEQUENCE = N_SEQUENCE  # 1セットのデータ数
        self.N_FEATURE = N_FEATURE    # 入力層・出力層のニューロン数
        
    def build_model(self):

        inputs = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
        x = GRU(128, return_sequences=True, recurrent_dropout=0.0)(inputs)
        x = GRU(64,  return_sequences=False, recurrent_dropout=0.0)(x)
        outputs = Dense(self.N_FEATURE)(x)
                
        model = Model(inputs,outputs) 
        return model


# Wave + Image Model
class ResNet18(tf.keras.Model):

    def __init__(self, img_shape=(256,256,1)):

        super(ResNet18,self).__init__()
        self.img_shape = img_shape
    
    def build(self):        
        
        num_filters = 64
        num_blocks = 4
        num_sub_blocks = 2

        # Main VGG Model
        inputs = Input(shape=self.img_shape, name="Image_input")
        
        x = Conv2D(filters=num_filters, kernel_size=(7,7), padding='same', strides=2, kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='block2_pool')(x)

        for i in range(num_blocks):
            for j in range(num_sub_blocks):
                
                strides=1
                
                is_first_layer_but_not_first_block=False
                if j==0 and i>0:
                    is_first_layer_but_not_first_block=True
                    strides=2

                y = Conv2D(num_filters, kernel_size=3, padding='same', strides=strides, kernel_initializer='he_normal')(x)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(y)
                y = BatchNormalization()(y)
                
                # Skip structure
                if is_first_layer_but_not_first_block:
                    x = Conv2D(num_filters, kernel_size=1, padding='same', strides=2, kernel_initializer='he_normal')(x)
                x = Add()([x, y])
                x = Activation('relu')(x)
                
            num_filters *= 2

        outputs = x
        # outputs = GlobalAveragePooling2D()(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model    


class WaveModel(tf.keras.Model):
    
    def __init__(self, 
                N_SEQUENCE, 
                N_FEATURE, 
                IMG_SHAPE=(256,256,1),
                image_model="ResNet18",
                pretrained=False):

        super(WaveModel,self).__init__(name='WaveModel')
        self.N_SEQUENCE  = N_SEQUENCE  # 1セットのデータ数
        self.N_FEATURE   = N_FEATURE    # 入力層・出力層のニューロン数
        self.IMG_SHAPE   = IMG_SHAPE
        self.image_model = image_model
        self.pretrained  = pretrained

        if pretrained:
            assert self.IMG_SHAPE[-1]==3,"Pretrained model's input must be 3ch image."
    
    def build_model(self):

        wave_model = self.build_wave_feature_extractor()
        image_model = self.build_image_feature_extractor()

        wave_feature = wave_model.output
        image_feature = image_model.output
    
        wave_feature = tf.squeeze(wave_feature, axis=-1)
        
        x = Concatenate(axis=-1)([wave_feature,image_feature])
        # x = Dense(512,activation="relu")(x)
        x = Dense(64,activation="relu")(x)
        x = Dense(1)(x)
        out = x

        model = Model(inputs=[wave_model.input,image_model.input], outputs=out)
        
        return model

    def build_wave_feature_extractor(self):
        
        # Wave Feature Extractor
        input_wave = Input(shape=(self.N_SEQUENCE, self.N_FEATURE),name="wave_input")

        x = Conv1D(filters=64, kernel_size=11, padding="same", strides=1, activation="relu")(input_wave)
        # x = Conv1D(filters=64, kernel_size=9, padding="same", strides=1, activation="relu")(x)
        x = Conv1D(filters=1 , kernel_size=7, padding="same", strides=1, activation="relu")(x)
        out = x
    
        return Model(inputs=input_wave,outputs=out, name='Wave_Feature_Extracor')  

    def build_image_feature_extractor(self):
        
        if self.image_model=="ResNet18":
            base_model = ResNet18(self.IMG_SHAPE).build()
            if self.pretrained:
                print("Cannot use pretrained ResNet18.")

        elif self.image_model=="ResNet50":
            weights=None
            if self.pretrained:
                weights='imagenet'
            base_model = ResNet50(include_top=False, weights=weights, input_shape=(self.IMG_SHAPE))
        else:
            ValueError(f"image model {self.image_model} is invalud !!")
        
        x = base_model.output # [None, 8, 8, 2048]
        x = GlobalAveragePooling2D()(x) # [None, 2048] 
        out = x

        return Model(inputs=base_model.input, outputs=out, name='Image_Feature_Extracor')


# U-time
class Utime(tf.keras.Model):
    """ U-Time
        
        paper: U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging
        arxiv: https://arxiv.org/pdf/1910.11162.pdf
        git: https://github.com/perslev/U-Time/blob/master/utime/models/utime.py

    """
    def __init__(self, 
                 N_SEQUENCE = 1400, 
                 N_FEATURE = 1, 
                 filters = 16, 
                 depth = 4,
                 dilation = 9,
                 down_size = (2, 2, 2, 2), 
                 movie_branch = False,
                 movie_input_shape = (256,256,1400,1)):
        
        # wave
        self.N_SEQUENCE = N_SEQUENCE
        self.N_FEATURE = N_FEATURE
        self.filters = filters
        self.depth = depth
        self.dilation=dilation
        self.down_size = down_size #(10, 8, 6, 4)
        self.up_size = self.down_size[::-1]
        self.n_crop=0

        # image
        self.movie_branch = movie_branch
        self.movie_input_shape = movie_input_shape
    
    def build_model(self):
        
        def encoding_layer(filters, x, dilation, pool):
            
            x = Conv1D(filters=filters, kernel_size=5, dilation_rate=dilation, activation="relu", padding="same",kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Conv1D(filters=filters, kernel_size=5, dilation_rate=dilation, activation="relu", padding="same",kernel_initializer="he_normal")(x)
            bn = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=pool)(bn)
            
            return x, bn 
        
        def bottom_layer(filters, x):
            
            x = Conv1D(filters=filters, kernel_size=5, dilation_rate=1, activation="relu", padding="same",kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Conv1D(filters=filters, kernel_size=5, dilation_rate=1, activation="relu", padding="same",kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            
            return x

        def decoding_layer(filters, x, skip, fs, dilation):
            
            x = UpSampling1D(size=fs)(x)
            x = Conv1D(filters=filters, kernel_size=5, dilation_rate=dilation, activation="relu", padding="same",kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)

            # Crop and concatenate
            cropped_skip = crop_nodes_to_match(skip, x)
            x = Concatenate(axis=-1)([cropped_skip, x])
        
            x = Conv1D(filters=filters, kernel_size=5, dilation_rate=dilation, activation="relu", padding="same",kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Conv1D(filters=filters, kernel_size=5, dilation_rate=dilation, activation="relu", padding="same",kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            
            return x
        
        def crop_nodes_to_match(node1, node2):
            """
                If necessary, applies Cropping2D layer to node1 to match shape of node2
                端数が出る場合は、cropした回数に合わせて、前に上に追加した場合は、次は下に追加するように、
                上下交互に追加していく方式をとっている。
            
            """
            s1 = np.array(node1.get_shape().as_list())[1]
            s2 = np.array(node2.get_shape().as_list())[1]
            # print(f"s1 {node1.shape} != s2 {node2.shape}") 
            
            if s1 != s2:
                self.n_crop += 1
                c  = (s1-s2)
                cr = [c//2,c//2]
                cr[self.n_crop % 2] += c%2
                cropped_node1 = Cropping1D(cr)(node1)
            else:
                cropped_node1 = node1

            return cropped_node1

        
        """ Inputs """
        inputs = Input(shape=(self.N_SEQUENCE, self.N_FEATURE))
        filters = [self.filters*(2**i) for i in range(self.depth+1)] 
        
        """ Encoder """
        skips=[]
        x = inputs
        for i in range(self.depth):
            x, skip = encoding_layer(filters[i], x, dilation=self.dilation, pool=self.down_size[i])
            skips.append(skip)

        """ Bottom """
        if self.movie_branch:
            movie_feature_extractor = ResNet18_3D(movie_input_shape=self.movie_input_shape).build()
            movie_inputs = movie_feature_extractor.inputs
            movie_outputs = movie_feature_extractor.output
            x = FusionLayer(x,movie_outputs)

        x = bottom_layer(filters[self.depth], x)
        
        """ Decoder """
        filters = filters[::-1][1:]
        skips   = skips[::-1]
        for i in range(self.depth):
            x = decoding_layer(filters=filters[i], x=x, skip=skips[i], fs=self.up_size[i], dilation=self.dilation)
            
        """ Last layer """
        s = self.N_SEQUENCE - x.get_shape().as_list()[1]        
        x = ZeroPadding1D(padding=[s//2, (s//2)+(s%2)])(x)
        x = Conv1D(filters=filters[-1],kernel_size=1,activation="tanh")(x)
        x = Conv1D(filters=1,kernel_size=1,activation="linear")(x)
        outputs = x        
        
        if self.movie_branch:
            return Model(inputs=[inputs,movie_inputs], outputs=outputs)
        else:
            return Model(inputs=inputs, outputs=outputs)

def FusionLayer(node1, node2):
    """
        node1: U-Time feature of bottom layer
        node2: 3D CNN feature
    """
    shape_1 = node1.shape[1]
    node2 = RepeatVector(shape_1)(node2)
    fused_feature = Concatenate()([node1, node2]) # → (?,28,28,1257)

    return fused_feature

class ResNet18_3D(tf.keras.Model):

    def __init__(self, movie_input_shape=(256,256,1400,1)):

        super(ResNet18_3D,self).__init__()
        self.movie_input_shape = movie_input_shape
    
    def build(self):        
        
        num_filters = 64
        num_blocks = 4
        num_sub_blocks = 2

        # Main VGG Model
        inputs = Input(shape=self.movie_input_shape, name="Image_input")
        
        x = Conv3D(filters=num_filters, kernel_size=7, padding='same', strides=2, kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=3, strides=2, padding='same', name='block2_pool')(x)

        for i in range(num_blocks):
            for j in range(num_sub_blocks):
                
                strides=1
                
                is_first_layer_but_not_first_block=False
                if j==0 and i>0:
                    is_first_layer_but_not_first_block=True
                    strides=2

                y = Conv3D(num_filters, kernel_size=3, padding='same', strides=strides, kernel_initializer='he_normal')(x)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = Conv3D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(y)
                y = BatchNormalization()(y)
                
                # Skip structure
                if is_first_layer_but_not_first_block:
                    x = Conv3D(num_filters, kernel_size=1, padding='same', strides=2, kernel_initializer='he_normal')(x)
                x = Add()([x, y])
                x = Activation('relu')(x)
                
            num_filters *= 2

        # outputs = x
        outputs = GlobalAveragePooling3D()(x) # → 512
        # outputs = Dense(128)(outputs) # → 128
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model    