from multiprocessing.dummy import Pool as ThreadPool

from PySide2.QtWidgets import QApplication
from tensorflow.python.keras.callbacks import LambdaCallback

from ..Core import Node
from ..Core.AbstractGraph import *


class FitGenerator(Node):
    def __init__(self, name, graph):
        super(FitGenerator, self).__init__(name, graph)

        self.in0 = self.addInputPin('in0', DataTypes.Exec, self.compute)
        self.compiled_model_pin = self.addInputPin('compiled_model', DataTypes.Any, defaultValue=None)
        self.training_pin = self.addInputPin('Training generator', DataTypes.Any, defaultValue=None)
        self.validation_pin = self.addInputPin('Validation generator', DataTypes.Any, defaultValue=None)

        self.steps_per_epoch_pin = self.addInputPin('Step per epoch', DataTypes.Int, 0)
        self.validation_steps_pin = self.addInputPin('validation step', DataTypes.Int, 0)

        self.callbacks_pin = self.addInputPin('callbacks', DataTypes.Array,defaultValue=[])
        self.batch_size_pin = self.addInputPin('batch size', DataTypes.Int,defaultValue=32)
        self.initial_epoch_pin = self.addInputPin('initial_epoch', DataTypes.Int,defaultValue=0)
        self.num_epoch_pin = self.addInputPin('num_epoch', DataTypes.Int,defaultValue=32)


        self.history_pin = self.addOutputPin('history', DataTypes.Any)
        self.batch_pin = self.addOutputPin('on_end_Batch', DataTypes.Exec)
        self.epoch_pin = self.addOutputPin('on_end_Epoch', DataTypes.Exec)
        self.completed_pin = self.addOutputPin('Completed', DataTypes.Exec)

        self.threadpool = ThreadPool(32)


        #pinAffects(self.in0, self.completed_pin)

    @staticmethod
    def pinTypeHints():
        '''
            used by nodebox to suggest supported pins
            when drop wire from pin into empty space
        '''
        return {'inputs': [], 'outputs': []}



    @staticmethod
    def category():
        '''
            used by nodebox to place in tree
            to make nested one - use '|' like this ( 'CatName|SubCatName' )
        '''
        return 'Keras|function'

    @staticmethod
    def keywords():
        '''
            used by nodebox filter while typing
        '''
        return ['generator','label']


    @staticmethod
    def description():
        '''
            used by property view and node box widgets
        '''
        return 'default description'

    def serialize(self):
        template = Node.serialize(self)
        # if hasattr(template["value"], '__class__'):
        #     template['value'] = None
        for i in list(template["outputs"])+template["inputs"]:
            i["value"] = None

        return template

    def UpdateBatch(self,batch, e_logs):
        QApplication.instance().processEvents()
        self.batch_pin.call()
        pass

    def UpdateEpoch(self,Epoch, e_logs):
        self.epoch_pin.call()
        pass

    def compute(self):

        print("fit gen")
        try:
            compiled_model = self.compiled_model_pin.getData()
            trainingGenerator = self.training_pin.getData()
            validationGenerator = self.validation_pin.getData()
            steps_per_epoch = self.steps_per_epoch_pin.getData()
            validation_steps_pin = self.steps_per_epoch_pin.getData()
            callbacks = self.callbacks_pin.getData()
            batch_size = self.batch_size_pin.getData()
            initial_epoch = self.initial_epoch_pin.getData()
            num_epoch = self.num_epoch_pin.getData()

            MyUpdateCB = LambdaCallback( on_batch_end=self.UpdateBatch,on_epoch_end=self.UpdateEpoch)
            callbacks.append(MyUpdateCB)
            if compiled_model != None:
                try:

                    print("keras memory")
                    from tensorflow.python.keras.backend import set_session
                    import tensorflow as tf
                    tf.logging.set_verbosity('DEBUG')
                    tfconfig = tf.ConfigProto()
                    tfconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
                    self.sess = tf.Session(config=tfconfig)
                    set_session(self.sess)  # set this TensorFlow session as the default session for Keras

                    print("fit_generator")
                    history = compiled_model.fit_generator( generator=trainingGenerator,
                                                            validation_data=validationGenerator,
                                                            steps_per_epoch=steps_per_epoch, #len(trainingGenerator) // batch_size,
                                                            epochs=initial_epoch + num_epoch,
                                                            validation_steps=validation_steps_pin, #len(validationGenerator) // batch_size,
                                                            initial_epoch=initial_epoch,
                                                            callbacks=callbacks,use_multiprocessing=False)

                    self.history_pin.setData(history)

                    self.completed_pin.call()

                except Exception as e:
                    import traceback
                    import sys
                    traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)

                finally:
                    self.sess.close()


        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)

        finally:
            self.sess.close()





