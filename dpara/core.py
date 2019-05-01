#MIT License
#
#Copyright (c) 2019 kurosawa
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

from mpi4py import MPI
import os, time
import threading
import numpy as np
import asyncio
import concurrent.futures

class Dpara():
    def __init__(self):
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        self.comm = MPI.COMM_WORLD
        self.loop = asyncio.get_event_loop()
        self.dev_flag = [1 for i in range(self.size)]
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def loop_para(self, func, args, i):
        self.executor.submit(self.__bcast())
        if i % self.size == self.rank:
            while True:
                if self.dev_flag[self.rank]:
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(self.rank)
                    self.dev_flag[self.rank] = 0
                    func(*args)
                    self.dev_flag[self.rank] = 1
                    break
        self.executor.submit(self.__bcast())

    def __bcast(self):
        for i in range(self.size):
            dev_flag_local = self.dev_flag[i]
            dev_flag_local = self.comm.bcast(dev_flag_local, root=i)
            if dev_flag_local > 1:
                dev_flag_local = 1
            self.dev_flag[i] = dev_flag_local
            assert self.dev_flag[i] <= 1, f"falg:{self.dev_flag[i]}"
