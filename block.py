# Author: Adam Loeckle
# Date: 6/17/2022
# Description: Block object and methods

from hashlib import sha256
from datetime import datetime

class Block:
    hash = ""
    prevHash = ""
    merkleRoot = ""
    transactions = []
    timeStamp = 0.0
    nonce = 0

    def __init__(self, prevHash):
        self.prevHash = prevHash
        self.timeStamp = datetime.now()
        self.hash = Block.calcHash(self)
    
    def calcHash(self):
        calculatedHash = sha256(bytes(Block.prevHash + str(Block.timeStamp) + str(Block.nonce) + Block.merkleRoot, "utf-8"))
        return calculatedHash
    
    def mineBlock(diff):
        pass
