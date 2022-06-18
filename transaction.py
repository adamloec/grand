# Author: Adam Loeckle
# Date: 6/17/2022
# Description: Transaction object and methods
from utils import Utilities

class Transaction:
    id = None
    sender = None
    recipient = None
    value = 0.0
    signature = None
    inputs = []
    outputs = []
    sequence = 0

    def __init__(self, sender, recipient, value, inputs):
        self.sender = sender
        self.recipient = recipient
        self.value = value
        self.inputs = inputs
    
    def calcHash(self):
        self.sequence += 1
        return Utilities.applySha256()
    
    def generateSig(self, privateKey):
        data = Utilities.getStringFromKey(self.sender) + Utilities.getStringFromKey(self.recipient) + str(self.value)
        self.signature = Utilities.applyECDSA(privateKey, data)
    
    def verifySig(self):
        data = Utilities.getStringFromKey(self.sender) + Utilities.getStringFromKey(self.recipient) + str(self.value)
        return Utilities.verifyECDSA(self.sender, data, self.signature)
        