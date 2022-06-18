# Author: Adam Loeckle
# Date: 6/17/2022
# Description: Block object and methods

from hashlib import sha256
from datetime import datetime
from utils import Utilities

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
        calculatedHash = sha256(bytes(self.prevHash + str(self.timeStamp) + str(self.nonce) + self.merkleRoot, "utf-8"))
        return calculatedHash
    
    def addTransaction(self, transaction):
        if (transaction == ""):
            return False
        
        if (self.prevHash != "0"):
            if (transaction.processTransaction() != True):
                print("ERROR: Transaction failed to process")
                return False
        
        self.transactions.append(transaction)
        print("CONSOLE: Transaction successfully added to block")
        return True

