# Author: Adam Loeckle
# Date: 6/17/2022
# Description: Tests

from block import Block
from utils import Utilities
from transaction import Transaction

from ellipticcurve.ecdsa import Ecdsa
from ellipticcurve.privateKey import PrivateKey

block = Block('janwdoiuboiubajfwf')
print(block.calcHash())



transaction = Transaction()