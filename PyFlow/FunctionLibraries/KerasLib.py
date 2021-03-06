from tensorflow.python.keras import layers, callbacks
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import applications
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.layers.base import Layer

from PyFlow.Core.AGraphCommon import DataTypes, NodeTypes
from PyFlow.Core.FunctionLibrary import FunctionLibraryBase, IMPLEMENT_NODE


class KerasLib(FunctionLibraryBase):
    '''doc string for KerasLib'''
    def __init__(self):
        super(KerasLib, self).__init__()





    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|function', 'Keywords': ['build']})
    def Build(Input=(DataTypes.Layer, None), Layers=(DataTypes.Layer, None)):
        '''Sum of two ints.'''
        m = Model(Input, Layers)
        m.summary()
        return m


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),
                    nodeType=NodeTypes.Callable,
                    meta={'Category': 'Keras|Layers', 'Keywords': ['input']})
    def Input(input_size=(DataTypes.Int, 224),
              input_channel=(DataTypes.Int, 3),
              LayerName=(DataTypes.String, "")):
        '''Input definition of the model'''
        return layers.Input(shape=(input_size, input_size, input_channel),name=LayerName)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None),
                    nodeType=NodeTypes.Callable,
                    meta={'Category': 'Keras|function', 'Keywords': ['save']})
    def Compile(model=(DataTypes.Any, None),
                optimizer=(DataTypes.Any, None),
                metrics=(DataTypes.Array, []),
                loss=(DataTypes.Any, None)):
        '''Compile Model'''
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("compile")
        return model

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None),
                    nodeType=NodeTypes.Pure,
                    meta={'Category': 'Keras|function', 'Keywords': ['load_model']})
    def LoadModel(model_file=(DataTypes.Files, "model.h5"),
                  yaml_file=(DataTypes.Files, "yaml.yaml")):
        '''load model from the path.'''

        yaml_file = open(yaml_file, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()

        model = applications.models.model_from_yaml(loaded_model_yaml)
        model.load_weights(model_file)
        model.summary()

        return model

    @staticmethod
    @IMPLEMENT_NODE(returns=None,
                    nodeType=NodeTypes.Callable,
                    meta={'Category': 'Keras|function', 'Keywords': ['save_model']})
    def SaveModel(model=(DataTypes.Any, None),
             out_filename=(DataTypes.String, "model_body"),
             overwrite=(DataTypes.Bool, True)):
        '''save model to the output file. the name must not include extension since it will be used for yaml and weights'''
        yaml_model = model.to_yaml()
        with open(out_filename + ".yaml", "w") as json_file:
            json_file.write(yaml_model)

        model.save(out_filename + ".h5", overwrite)

    # region Layers
    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['+', 'merge', 'concate']})
    def Merge(Layer_1=(DataTypes.Layer, Layer(0)), Layer_2=(DataTypes.Layer, Layer(0)),LayerName=(DataTypes.String, "")):
        return layers.Concatenate(name=LayerName)([Layer_1, Layer_2])


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['Normalization',"batch"]})
    def BatchNormalization(Input=(DataTypes.Layer, Layer(0)),LayerName=(DataTypes.String, "")):
        return layers.BatchNormalization(name=LayerName)(Input)


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['Activation']})
    def Activation(Input=(DataTypes.Layer, Layer(0)), Type=(DataTypes.String, "relu"),LayerName=(DataTypes.String, "")):
        return layers.Activation(activation=Type,name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['AveragePooling2D',"pool"]})
    def MaxPooling2D(Input=(DataTypes.Layer, Layer(0)),pool_size=(DataTypes.Int, 32),strides=(DataTypes.Int, 1),padding=(DataTypes.String, "valid"),LayerName=(DataTypes.String, "")):
        return layers.MaxPooling2D(pool_size=pool_size,strides=strides,padding=padding, name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['AveragePooling2D',"pool"]})
    def AveragePooling2D(Input=(DataTypes.Layer, Layer(0)),pool_size=(DataTypes.Int, 32),strides=(DataTypes.Int, 1),padding=(DataTypes.String, "valid"),LayerName=(DataTypes.String, "")):
        return layers.AveragePooling2D(pool_size=pool_size,strides=strides,padding=padding, name=LayerName)(Input)


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['SeparableConv2D',"Conv"]})
    def SeparableConv2D(Input=(DataTypes.Layer, Layer(0)),filters=(DataTypes.Int, 128),kernel_size=(DataTypes.Int, 32),strides=(DataTypes.Int, 1),padding=(DataTypes.String, "valid"),LayerName=(DataTypes.String, "")):
        return layers.SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['SeparableConv2D',"Conv"]})
    def Conv2D(Input=(DataTypes.Layer, Layer(0)),filters=(DataTypes.Int, 128),kernel_size=(DataTypes.Int, 32),strides=(DataTypes.Int, 1),padding=(DataTypes.String, "same"),LayerName=(DataTypes.String, "")):
        return layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['Dense']})
    def Dense(Input=(DataTypes.Layer, Layer(0)), units=(DataTypes.Int, 1024),Type=(DataTypes.String, "relu"),LayerName=(DataTypes.String, "")):

        return layers.Dense(units=units, activation=Type,name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['Flatten']})
    def Flatten(Input=(DataTypes.Layer, Layer(0)), LayerName=(DataTypes.String, "")):
        return layers.Flatten(name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|Layers', 'Keywords': ['input']})
    def Input(input_size=(DataTypes.Int, 224),input_channel=(DataTypes.Int, 3), LayerName=(DataTypes.String, "")):
        return layers.Input(shape=(input_size, input_size, input_channel),name=LayerName)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None), nodeType=NodeTypes.Callable,
                    meta={'Category': 'Keras|Layers', 'Keywords': ['Dropout']})
    def Dropout(Input=(DataTypes.Layer, Layer(0)),rate=(DataTypes.Float, 0), LayerName=(DataTypes.String, "")):
        return layers.Dropout(rate=rate, name=LayerName)(Input)


    # endregion

    # region Callback
    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None),nodeType=NodeTypes.Pure, meta={'Category': 'Keras|callbacks', 'Keywords': ['checkpoint']})
    def Checkpoint(period=(DataTypes.Int, 1),save_weights_only=(DataTypes.Bool, True),verbose=(DataTypes.Int, 1), filepath=(DataTypes.String, "logs"), monitor=(DataTypes.String, "val_loss")):
        return callbacks.ModelCheckpoint(filepath=filepath,monitor=monitor, save_best_only=True, mode='auto', save_weights_only=save_weights_only, period=period, verbose=verbose)


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None),nodeType=NodeTypes.Pure, meta={'Category': 'Keras|callbacks', 'Keywords': ['stop', 'early']})
    def EarlyStopping(patience=(DataTypes.Int, 1),min_delta=(DataTypes.Float, 0),verbose=(DataTypes.Int, 0), monitor=(DataTypes.String, "val_loss")):

        return callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode='auto', baseline=None)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None),nodeType=NodeTypes.Pure, meta={'Category': 'Keras|callbacks', 'Keywords': ['stop', 'early']})
    def EarlyStopping(patience=(DataTypes.Int, 1),min_delta=(DataTypes.Int, 0),verbose=(DataTypes.Int, 0), monitor=(DataTypes.String, "val_loss")):
        '''Sum of two ints.'''
        # if(LayerName==""):
        #     LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))
        return callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode='auto', baseline=None)


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None), nodeType=NodeTypes.Pure,
                    meta={'Category': 'Keras|callbacks', 'Keywords': ['LR', 'learning', 'reduce', 'plateau']})
    def ReduceLROnPlateau(min_lr=(DataTypes.Float, 1e-7), factor=(DataTypes.Float, 0.10), patience=(DataTypes.Int, 10),monitor=(DataTypes.String, "val_loss")):
        return callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=min_lr)


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None), nodeType=NodeTypes.Pure,
                    meta={'Category': 'Keras|callbacks', 'Keywords': ['TerminateOnNaN', 'learning', 'callback']})
    def TerminateOnNaN():
        '''Sum of two ints.'''
        # if(LayerName==""):
        #     LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))
        return callbacks.TerminateOnNaN()


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None), nodeType=NodeTypes.Pure,
                    meta={'Category': 'Keras|callbacks', 'Keywords': ['tensorboard', 'board', 'callback']})
    def TensorBoard(batch_size=(DataTypes.Int, 32),histogram_freq=(DataTypes.Int, 0),log_dir=(DataTypes.String, "logs"),write_images=(DataTypes.Bool, False),
                    write_graph=(DataTypes.Bool, True),write_grads=(DataTypes.Bool, False)):
        return callbacks.TensorBoard(log_dir=log_dir, batch_size=batch_size, write_images=write_images,
                histogram_freq=histogram_freq, write_graph=write_graph, write_grads=write_grads)

    # endregion


    # region optimiser
    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, None), nodeType=NodeTypes.Pure,
                    meta={'Category': 'Keras|optimiser', 'Keywords': ['Adam', 'optimiser']})
    def Adam(learning_rate=(DataTypes.Float, 0.001),beta_1=(DataTypes.Float, 0.9),beta_2=(DataTypes.Float, 0.999),
        epsilon = (DataTypes.Any, None),decay=(DataTypes.Float, 0.), amsgrad=(DataTypes.Bool, False)):
        return optimizers.Adam(lr=learning_rate, beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,decay=decay,amsgrad=amsgrad)

