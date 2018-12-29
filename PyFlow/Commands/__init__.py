import os

from PyFlow.Commands import ConnectPin

for n in os.listdir(os.path.dirname(__file__)):
    if n.endswith(".py") and "__init__" not in n:
        commandName = n.split(".")[0]
        exec("from PyFlow.Commands import {0}".format(commandName))



# import os
# for n in os.listdir(os.path.dirname(__file__)):
#     if n.endswith(".py") and "__init__" not in n:
#         commandName = n.split(".")[0]
#         exec("from {0} import *".format(commandName))
