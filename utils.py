# Author: Adam Loeckle
# Date: 6/17/2022
# Description: Utility methods

from hashlib import sha256
from ellipticcurve.ecdsa import Ecdsa
from ellipticcurve.privateKey import PrivateKey

class Utilities:

    def applySha256(input):
        input = bytes(input, "utf-8")
        return sha256(input)
    
    def applyECDSA(privateKey, input):
        return Ecdsa.sign(input, privateKey)
    
    def verifyECDSA(publicKey, input, signature):
        return Ecdsa.verify(input, signature, publicKey)
    
    def getMerkle(transactions):
        count = len(transactions)
        previousTreeLayer = []

        for transaction in transactions:
            previousTreeLayer.append(transaction.id)
        
        
        