from pandas import DataFrame

import fabrik
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Tester:
    def iteration_cost(self,model:fabrik.Structure,target:np.ndarray,tolerance:float=0.05,track:bool=False,name:str='wow')->float:
        f=fabrik.Fabrik()
        f.fit_model(model)
        progress_time,cnt,structure=f.solve(target,tolerance,track,name)
        return cnt if cnt!=-1 else np.nan

    def iteration_cost1(self,model:fabrik.Structure,target:np.ndarray,tolerance:float=0.05,a:float=0.5,b:float=0.5,track:bool=False,name1:str='wow')->float:
        f=fabrik.Fabrik()
        f.fit_model(model)
        progress_time1,cnt1,structure1=f.solve1(target,tolerance,a,b,track,name1)
        return cnt1 if cnt1!=-1 else np.nan

    def iteration_cost_average(self,model_list:list,target_list:list,tolerance:float=0.05)->float:
        total=0
        for i in range(len(model_list)):
            tmp=self.iteration_cost(model_list[i],target_list[i],tolerance)
            if tmp == np.nan:
                return np.nan
            total+=tmp

        return total/len(model_list)

    def iteration_cost_average1(self,model_list:list,target_list:list,tolerance:float=0.05,a:float=0.5,b:float=0.5)->float:
        total1=0
        for i in range(len(model_list)):
            tmp1=self.iteration_cost1(model_list[i],target_list[i],tolerance,a,b)
            if tmp1==np.nan:
                return np.nan
            total1+=tmp1

        return total1/len(model_list)

    def motion_cost(self,model:fabrik.Structure,target:np.ndarray,tolerance:float=0.05,mode:bool=True)->float:
        f=fabrik.Fabrik()
        f.fit_model(model)
        progress_time,cnt,structure=f.solve(target,tolerance)
        if cnt==-1:
            return np.nan
        motion_cost=0
        for i in range(1,structure.shape[0]):
            if mode:
                motion_cost+=np.linalg.norm(model.structure[i]-structure[i])
            else:
                motion_cost+=np.linalg.norm(model.structure[i]-structure[i])*(model.length[i:].sum()/model.length.sum())
        return motion_cost

    def motion_cost1(self,model:fabrik.Structure,target:np.ndarray,tolerance:float=0.05,a:float=0.5,b:float=0.5,mode:bool=True)->float:
        f=fabrik.Fabrik()
        f.fit_model(model)
        progress_time1,cnt1,structure1=f.solve1(target,tolerance,a,b)
        print(structure1)
        if cnt1==-1:
            return np.nan
        motion_cost1=0
        for i in range(1,structure1.shape[0]):
            if mode:
                motion_cost1+=np.linalg.norm(model.structure[i]-structure1[i])
            else:
                motion_cost1+=np.linalg.norm(model.structure[i]-structure1[i])*(model.length[i:].sum()/model.length.sum())
        return motion_cost1


    def motion_cost_average(self,model_list:list,target_list:list,tolerance:float=0.05,mode:bool=True)->float:
        total=0
        for i in range(len(model_list)):
            tmp=self.motion_cost(model_list[i],target_list[i],tolerance,mode)
            total+=tmp
            if tmp==np.nan:
                return np.nan

        return total/len(model_list)

    def motion_cost_average1(self,model_list:list,target_list:list,tolerance:float=0.05,a:float=0.5,b:float=0.5,mode:bool=True)->float:
        total1=0
        for i in range(len(model_list)):
            tmp1=self.motion_cost1(model_list[i],target_list[i],tolerance,a,b,mode)
            total1+=tmp1
            if tmp1==np.nan:
                return np.nan

        return total1/len(model_list)

    def test(self,model_list:list,target_list:list,tolerance:float=0.05,mode:bool=True)-> DataFrame:
        result=[]
        motion_cost=self.motion_cost_average(model_list,target_list,tolerance,mode)
        iteration_cost=self.iteration_cost_average(model_list,target_list,tolerance)
        result.append([iteration_cost,motion_cost])

        return pd.DataFrame(result,columns=['iteration_cost','motion_cost'])

    def test1(self,model_list:list,target_list:list,tolerance:float=0.05,final_a:float=2,final_b:float=2,step_a:float=0.05,step_b:float=0.05,mode:bool=True)-> DataFrame:
        result1=[]
        total_iteration =int(((final_a // step_a + 1) * (final_b // step_b + 1))  )
        present_iteration=0
        for a in np.arange(-2,final_a,step_a):
            for b in np.arange(-2,final_b,step_b):
                motion_cost1=self.motion_cost_average1(model_list,target_list,tolerance,a,b,mode)
                iteration_cost1=self.iteration_cost_average1(model_list,target_list,tolerance,a,b)
                result1.append([iteration_cost1,motion_cost1,a,b])
                present_iteration+=1
                print(f"iteration:{present_iteration}/{total_iteration}")

        return pd.DataFrame(result1,columns=['iteration_cost','motion_cost','a','b'])

    def optimize1(self,model_list:list,target_list:list,tolerance:float=0.05,final_a:float=2,final_b:float=2,step_a:float=0.05,step_b:float=0.05,mode:bool=True)-> DataFrame:
        loss1_df=self.test1(model_list,target_list,tolerance,final_a,final_b,step_a,step_b,mode)
        # find minimum loss1 in loss1_df by a,b
        min_loss1_df=loss1_df.sort_values(by='iteration_cost')

        return min_loss1_df

    def optimize1_with_result(self,result1_df:DataFrame)->DataFrame:
        return result1_df.sort_values(by='iteration_cost')

    def plot(self,model_list:list,target_list:list,tolerance:float=0.05,final_a:float=2,final_b:float=2,step_w:float=0.05,step_a:float=0.05,step_b:float=0.05,mode:bool=True)->None:
        min_loss1_df=self.optimize1(model_list,target_list,tolerance,final_a,final_b,step_w,step_a,step_b,mode)
        loss_df=self.test(model_list,target_list,tolerance,step_w,mode)

        # plot loss_df and min_loss1_df by w_time
        self.plot_with_result(loss_df,min_loss1_df)


    def plot_with_result(self,result_df:DataFrame,result1_df:DataFrame)->None:
        # scatter plot loss_df of loss by w_time and plot min_loss1_df of loss1 by w_time in the same plot
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=result_df,x='w_iteration',y='loss',label='loss')
        sns.scatterplot(data=result1_df,x='w_iteration',y='loss1',label='loss1')
        plt.show()




"""
ex) Test Case

np.ndarray, np.ndarray
각각 structure, length
structure 의 shape은 (n,3) 이고 length의 shape은 (n-1,) 이다.

"""
class TestCase:
    def example(self):
        self.structure = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.], [4., 0., 0.]])
        self.length = np.array([1., 1., 1., 1.])
        self.target = np.array([3., 1., 0.])
        return [fabrik.Structure(self.structure, self.length)], [self.target]

    def build_structure_straight(self, n: int = 5, t: int = 50) -> None:
        list_structure = []
        list_target = []
        for j in range(t):
            self.structure = [np.array([0, 0, 0])]
            for i in range(1, n):
                length = 1.0
                self.structure.append(self.structure[-1] + np.array([length, 0, 0]))
            self.structure = np.array(self.structure)
            self.length = np.array([np.linalg.norm(self.structure[i + 1] - self.structure[i]) for i in range(n - 1)])
            self.target = self.structure[0] + np.random.uniform(-1.5, 1.5, 3)
            list_structure.append(fabrik.Structure(self.structure, self.length))
            list_target.append(self.target)
        return list_structure, list_target

    def build_structure_straight_random(self, n: int = 5, t: int = 50) -> None:
        list_structure = []
        list_target = []
        for j in range(t):
            self.structure = [np.array([0, 0, 0])]
            for i in range(1, n):
                length = np.random.uniform(0.5, 1.5)
                self.structure.append(self.structure[-1] + np.array([length, 0, 0]))
            self.structure = np.array(self.structure)
            self.length = np.array([np.linalg.norm(self.structure[i+1] - self.structure[i]) for i in range(n-1)])
            self.target = self.structure[0] + np.random.uniform(-1.5, 1.5, 3)
            list_structure.append(fabrik.Structure(self.structure, self.length))
            list_target.append(self.target)
        return list_structure, list_target

    def build_structure_angles(self, n: int = 5, t: int = 50) -> None:
        list_structure = []
        list_target = []
        for j in range(t):
            self.structure = [np.array([0, 0, 0])]
            length = 1.0
            for i in range(1, n):
                angle = np.random.uniform(0, 2 * np.pi)
                self.structure.append(self.structure[-1] + length * np.array([np.cos(angle), np.sin(angle), 0]))
            self.structure = np.array(self.structure)
            self.length = np.array([length for _ in range(n-1)])
            self.target = self.structure[0] + np.random.uniform(-1, 1, 3)
            list_structure.append(fabrik.Structure(self.structure, self.length))
            list_target.append(self.target)
        return list_structure, list_target

    def build_structure_random(self, n: int = 5, t: int = 50) -> None:
        list_structure = []
        list_target = []
        for j in range(t):
            self.structure = [np.array([0, 0, 0])]
            for i in range(1, n):
                length = np.random.uniform(0.5, 1.5)
                angle = np.random.uniform(0, 2 * np.pi)
                self.structure.append(self.structure[-1] + length * np.array([np.cos(angle), np.sin(angle), 0]))
            self.structure = np.array(self.structure)
            self.length = np.array([np.linalg.norm(self.structure[i+1] - self.structure[i]) for i in range(n-1)])
            self.target = self.structure[0] + np.random.uniform(-1.5, 1.5, 3)
            list_structure.append(fabrik.Structure(self.structure, self.length))
            list_target.append(self.target)
        return list_structure, list_target