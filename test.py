# Author: Adam Loeckle
# Date: 6/17/2022
# Description: Tests

from block import Block
from utils import Utilities
from transaction import Transaction

from ellipticcurve.ecdsa import Ecdsa
from ellipticcurve.privateKey import PrivateKey

# Block
block = Block('janwdoiuboiubajfwf')
print(block.calcHash())

# Key creation
privateKey = PrivateKey()
publicKey = privateKey.publicKey()
message = "message12512fAEfaq21f3"
signature = Utilities.applyECDSA(privateKey, message)
verify = Utilities.verifyECDSA(publicKey, message, signature)
print(verify)

# Transaction
transaction = Transaction()