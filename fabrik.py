import time
import math
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps

dist=lambda a,b:((((a[0]-b[0])**2)+((a[1]-b[1])**2))**0.5)
unit_vector=lambda v:v/np.linalg.norm(v)

def timer(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start=time.time()
        result=func(*args,**kwargs)
        end=time.time()
        print(f"{func.__name__} Execution time:{end-start}")
        return result
    return wrapper

class Structure:
    def __init__(self,structure:np.ndarray,length:np.ndarray)->None:
        self.structure=structure.copy()
        self.base=structure[0]
        self.length=length.copy()

    def plot(self)->None:
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        for i in range(self.structure.shape[0]-1):
            ax.plot([self.structure[i,0],self.structure[i+1,0]],[self.structure[i,1],self.structure[i+1,1]],[self.structure[i,2],self.structure[i+1,2]],c='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

class Fabrik:
    def __init__(self,model:Structure)->None:
        self.base=model.structure[0].copy()
        self.structure=model.structure.copy()
        self.length=model.length.copy()
        self.target=np.zeros(3)

    def isreachable(self,target:np.ndarray)->bool:
        if dist(self.base,target)>np.sum(self.length):
            return False
        return True

    def forward(self):
        self.structure[-1]=self.target
        for i in range(1,self.structure.shape[0]):
            self.structure[-i-1]=self.structure[-i]+unit_vector(self.structure[-i-1]-self.structure[-i])*self.length[-i]

    def backward(self):
        self.structure[0]=self.base
        for i  in range(0,self.structure.shape[0]-1):
            self.structure[i+1]=self.structure[i]+unit_vector(self.structure[i+1]-self.structure[i])*self.length[i]

    def solve(self,target:np.ndarray,tolerance:float)->np.ndarray:
        if self.isreachable(target):
            self.target=target.copy()
            cnt=0
            while (diff:=dist(self.structure[-1],self.target))>tolerance:
                self.forward()
                self.backward()
                cnt+=1
        else:
            return self.structure

    def forward1(self):
        self.structure[-1]=(self.a*unit_vector(self.structure[1]-self.structure[0]))*np.dot((self.structure[1]-self.structure[0]),self.target-self.structure[-1])/np.sum(self.length)+(self.b*unit_vector(self.target-self.structure[0]))*(np.dot(self.target-self.structure[0],self.target-self.structure[-1])/np.sum(self.length))+self.structure[-1]
        for i in range(1,self.structure.shape[0]):
            self.structure[-i-1]=self.structure[-i]+unit_vector(self.structure[-i-1]-self.structure[-i])*self.length[-i]

    def solve1(self,target:np.ndarray,tolerance:float,a:float,b:float)->np.ndarray:
        self.a=a
        self.b=b
        if self.isreachable(target):
            self.target=target.copy()
            cnt=0
            while (diff:=dist(self.structure[-1],self.target))>tolerance:
                self.forward1()
                self.backward()
                cnt+=1
        else:
            return self.structure

    def plot(self)->None:
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        for i in range(self.structure.shape[0]-1):
            ax.plot([self.structure[i,0],self.structure[i+1,0]],[self.structure[i,1],self.structure[i+1,1]],[self.structure[i,2],self.structure[i+1,2]],c='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()