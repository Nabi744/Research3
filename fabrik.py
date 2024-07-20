import time
import math
import matplotlib.pyplot as plt
import numpy as np

dist=lambda a,b:((((a[0]-b[0])**2)+((a[1]-b[1])**2))**0.5)
unit_vector=lambda v:v/np.linalg.norm(v)

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
    def fit_model(self,model:Structure)->None:
        self.base=model.structure[0].copy()
        self.structure=model.structure.copy()
        self.length=model.length.copy()
        self.target=np.zeros(3)

    def isreachable(self,target:np.ndarray)->bool:
        if dist(self.base,target)>np.sum(self.length):
            return False
        return True

    def __forward(self):
        self.structure[-1]=self.target
        for i in range(1,self.structure.shape[0]):
            self.structure[-i-1]=self.structure[-i]+unit_vector(self.structure[-i-1]-self.structure[-i])*self.length[-i]

    def __backward(self):
        self.structure[0]=self.base
        for i  in range(0,self.structure.shape[0]-1):
            self.structure[i+1]=self.structure[i]+unit_vector(self.structure[i+1]-self.structure[i])*self.length[i]

    def __forward1(self,a:float,b:float):
        self.structure[-1]=self.__fake_target(a,b)
        # self.structure[-1]=(a*unit_vector(self.structure[1]-self.structure[0]))*np.dot((self.structure[1]-self.structure[0]),self.target-self.structure[-1])/np.sum(self.length)+(b*unit_vector(self.target-self.structure[0]))*(np.dot(self.target-self.structure[0],self.target-self.structure[-1])/np.sum(self.length))+self.target
        for i in range(1,self.structure.shape[0]):
            self.structure[-i-1]=self.structure[-i]+unit_vector(self.structure[-i-1]-self.structure[-i])*self.length[-i]

    def __fake_target(self,a:float,b:float):
        fake_target=self.target.copy()
        total_length=np.sum(self.length)
        fake_target+=a*unit_vector(self.structure[1]-self.structure[0])*np.dot(self.structure[1]-self.structure[0],fake_target-self.structure[-1])/total_length
        fake_target+=b*unit_vector(fake_target-self.structure[0])*np.dot(fake_target-self.structure[0],fake_target-self.structure[-1])/total_length

        return fake_target

    def solve(self,target:np.ndarray,tolerance:float=0.05,track:bool=False,name:str='wow')->(float,int,np.ndarray):
        if track:
            track=np.ndarray()
            np.concatenate((track,self.structure),axis=0)
        if self.isreachable(target):
            self.target=target.copy()
            cnt=0
            st=time.time()
            while (diff:=dist(self.structure[-1],self.target))>tolerance:
                self.__forward()
                self.__backward()
                cnt+=1
                if track:
                    np.concatenate((track,self.structure),axis=0)

                if cnt>200:
                    print("This Algorithm is not converging")
                    return -1,-1,self.structure

            if track:
                # download np.ndarray in structure_process directory by name
                np.save(f"structure_process/{name}.npy",track)
            progress_time=time.time()-st
            return progress_time*1000,cnt,self.structure
        else:
            print("Target is unreachable")
            return -1,-1,self.structure

    def solve1(self,target:np.ndarray,tolerance:float,a:float,b:float,track:bool=False,name:str='wow')->(float,int,np.ndarray):
        if track:
            track=np.ndarray()
            np.concatenate((track,self.structure),axis=0)
        if self.isreachable(target):
            self.target=target.copy()
            cnt=0
            st=time.time()
            while (diff:=dist(self.structure[-1],self.target))>tolerance:
                self.__forward1(a,b)
                self.__backward()
                cnt+=1
                if track:
                    np.concatenate((track,self.structure),axis=0)
                if cnt>200:
                    print("This Algorithm is not converging")
                    return -1,-1,self.structure
            if track:
                np.save(f"structure_process/{name}.npy",track)
            progress_time=time.time()-st
            return progress_time*1000,cnt,self.structure
        else:
            print("Target is unreachable")
            return -1,-1,self.structure

    # def plot(self)->None:
    #     fig=plt.figure()
    #     ax=fig.add_subplot(111,projection='3d')
    #     for i in range(self.structure.shape[0]-1):
    #         ax.plot([self.structure[i,0],self.structure[i+1,0]],[self.structure[i,1],self.structure[i+1,1]],[self.structure[i,2],self.structure[i+1,2]],c='r')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #
    #     plt.show()