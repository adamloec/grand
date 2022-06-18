# Author: Adam Loeckle
# Date: 6/17/2022
# Description: Transaction object and methods
from utils import Utilities
from chain import Chain

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
    
    def processTransaction(self):
        if (self.verifySig() == False):
            print("ERROR: Transaction signature could not be verified")
        
        for input in self.inputs:
            input.utxo = Chain.utxos[input.transactionOutputId]

        if (self.getInputsValue() < Chain.minTransaction):
            print("ERROR: Transaction input is too small")
            return False
        
        leftOver = self.getInputsValue() - self.value
        self.transactionId = self.calcHash()

        self.outputs.append(TransactionOutput(self.recipient, self.value, self.transactionId))
        self.outputs.append(TransactionOutput(self.sender, leftOver, self.transactionId))

        for output in self.outputs:
            Chain.utxos.update({output.id: output})
        
        for input in self.inputs:
            if (input.utxo == None):
                continue
            else:
                Chain.utxos.pop(input.utxo.id)

        return True

    def getInputsValue(self):
        total = 0.0
        for input in self.inputs:
            if (input.utxo == None):
                continue
            else:
                total += input.utxo.value
        return total

    def getOutputsValue(self):
        total = 0.0
        for output in self.outputs:
            total += output.value
        return total

class TransactionOutput:
    id = ""
    recipient = None
    value = 0.0
    parentTransactionId = ""

    def __init__(self, recipient, value, parentTransactionId):
        self.recipient = recipient
        self.value = value
        self.parentTransactionId = parentTransactionId
        self.id = Utilities.applySha256(Utilities.getStringFromKey(self.recipient) + str(self.value) + self.parentTransactionId)

    def owner(self, publicKey):
        return (Utilities.getStringFromKey(publicKey) == Utilities.getStringFromKey(self.recipient))

class TransactionInput:
    transactionOutputId = ""
    utxo = TransactionOutput(None, None, None)

    def __init__(self, transactionOutputId):
        self.transactionOutputId = transactionOutputId