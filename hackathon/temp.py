import os
import glob

import numpy as np

class User():
  def __init__(self,name, balance):
    self.name = name
    self.balance = balance

  def changement_balance(self,somme):
    self.balance = self.balance + somme
    print(self.balance)
    return str(self.balance)

  def donne(self,receiver_name, amount):
    self.changement_balance( -amount)
    receiver_name.changement_balance(amount)

Jeff = User('Jeff', 20)
Bob = User('Bob', 100)
Jeff.donne(Bob, 40)

Bob.add_cash()
