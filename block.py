# Author: Adam Loeckle
# Date: 6/17/2022
# Description: Block object and methods

from hashlib import sha256
from datetime import datetime

class Block:
    hash = None
    prevHash = None
    merkleRoot = None
    transactions = []
    timeStamp = 0.0
    nonce = 0

    def init(self, prevHash):
        self.prevHash = prevHash
        self.timeStamp = datetime.now()
        self.hash = Block.calcHash()
    
    def calcHash(self):
        calculatedHash = sha256(Block.prevHash + str(Block.timeStamp) + str(Block.nonce) + Block.merkleRoot)
        return calculatedHash
    
    def mineBlock(self, diff):
        pass
