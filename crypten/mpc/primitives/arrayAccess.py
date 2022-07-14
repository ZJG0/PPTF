'''
Author: ZJG
Date: 2022-07-05 16:00:32
LastEditors: ZJG
LastEditTime: 2022-07-07 19:58:02
'''
import crypten
import crypten.communicator as comm
import torch
import random
from crypten.common.util import count_wraps
from crypten.config import cfg
from .arithmetic import ArithmeticSharedTensor


def reverse(arr,start,end):
    while start<end:
        temp = arr[start]
        arr[start] = arr[end]
        arr[end] = temp
        start += 1
        end -= 1
 
def rightShift(arr,k):
    if arr == None:
        print("Paramter Invalid!")
        return
    lens = len(arr)
    k %= lens
    reverse(arr,0,lens-k-1)
    reverse(arr,lens-k,lens-1)
    reverse(arr,0,lens-1)


def SecureArrayAccess(array, index):
    if comm.get().get_world_size() != 3:
        raise NotImplementedError(
            "SecureArrayAccess is only implemented for world_size == 3."
        )

    rank = comm.get().get_rank()

    c = ArithmeticSharedTensor.PRSS(array.size(), device=array.device)

    if rank == 0:
        r1 = random.randint(0, 10000);
        r3 = random.randint(0, 10000);
        comm.get().send(r1, 1)
        comm.get().send(r3, 2)
        rightShift(array, r1)
        array = array + c
        rightShift(array, r3)
        index = (index+r1+r3)%array.size()
        comm.get().send(array.share, 1)
        comm.get().send(index, 1)

    elif rank == 1:
        r1 = comm.get().recv(r1, 0)
        h1 = comm.get().recv(index, 0)
        a = comm.get().recv(array.share, 0)

        h = index + h1
        array = rightShift(array, r1)
        array = array + c

        comm.get().send(array.share, 2)
        comm.get().send(h, 2)

    elif rank == 2:
        a = comm.get().recv(array.share, 1)
        h = comm.get().recv(h, 1)
        r3 = comm.get().recv(r3, 0)

        a = a + c
        rightShift(a, r3)

    
    