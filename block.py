

from hashlib import sha256
from datetime import datetime

class Block:
    hash = ""
    prevHash = ""
    merkleRoot = ""
    transactions = [""]
    timeStamp = 0.0
    nonce = 0

    def init(self, prevHash):
        self.prevHash = prevHash
        self.timeStamp = datetime.now()
        self.hash = Block.calcHash()
    
    def calcHash():
        calculatedHash = sha256(Block.prevHash + str(Block.timeStamp) + str(Block.nonce) + Block.merkleRoot)
        return calculatedHash
    
    def mineBlock(diff):
        pass
