import fabrik
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import customtkinter as ctk
import CTkSpinbox

from tkinter import filedialog
from PIL import Image, ImageTk

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("FABRIK")
        self.geometry("1000x700")
        self.resizable(True,True)

        #DataFrame
        self.data = pd.DataFrame({
            "Test": [],
            "FABRIK(ms)": [],
            "FABRIK(t)": [],
            "H-FABRIK 1(ms)": [],
            "H-FABRIK 1(t)": [],
            "H-FABRIK 2(ms)": [],
            "H-FABRIK 2(t)": [],
            "H-FABRIK 3(ms)": [],
            "H-FABRIK 3(t)": []
        })

        self.view_index=0
        self.view_iteration=0

        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.grid(row=0, column=10, padx=20, pady=20)
        self.display_table()


        #Plot Image
        self.display_image()

        #Buttons/Labels
        self.before_before_button=ctk.CTkButton(self,text="<<",command=self.before_before_plot)
        self.before_before_button.grid(row=10, column=0, pady=20)
        self.before_button=ctk.CTkButton(self,text="←",command=self.before_plot)
        self.before_button.grid(row=10, column=1, pady=20)
        self.after_button=ctk.CTkButton(self,text="→",command=self.after_plot)
        self.after_button.grid(row=10, column=2, pady=20)
        self.after_after_button=ctk.CTkButton(self,text=">>",command=self.after_after_plot)
        self.after_after_button.grid(row=10, column=3, pady=20)

        self.iteration_label=ctk.CTkLabel(self,text=f"0/0")
        self.iteration_label.grid(row=10,column=4,pady=20)

        algorithms=["FABRIK","H-FABRIK 1","H-FABRIK 2","H-FABRIK 3"]
        self.algo_optionmenu=ctk.CTkOptionMenu(self,values=algorithms,command=self.change_algorithm)
        self.algo_optionmenu.grid(row=11,column=0,pady=20)


        self.test_case_label=ctk.CTkLabel(self,text="Test Case")
        self.test_case_label.grid(row=11,column=1,pady=20)

        self.test_case_spinbox=CTkSpinbox.CTkSpinbox(self,min_value=1,max_value=5,scroll_value=1,command=self.view_test_case)
        self.test_case_spinbox.grid(row=11,column=2,pady=20)



        self.upload_button = ctk.CTkButton(self, text="Upload", command=self.upload_file)
        self.upload_button.grid(row=12, column=0, pady=20)

        self.file_name_label = ctk.CTkLabel(self, text="No file uploaded")
        self.file_name_label.grid(row=12, column=1, pady=20)

        self.refresh_button = ctk.CTkButton(self, text="Run", command=self.run)
        self.refresh_button.grid(row=12, column=2, pady=20)

    def display_table(self):
        """ This function displays the DataFrame as a table using CTkLabels. """
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        for i, (index, row) in enumerate(self.data.iterrows()):
            for j, value in enumerate(row):
                label = ctk.CTkLabel(self.table_frame, text=str(value))
                label.grid(row=i, column=10+j, sticky="nsew", padx=5, pady=5)

        # Add column headers
        for idx, column in enumerate(self.data.columns):
            label = ctk.CTkLabel(self.table_frame, text=str(column), height=20)
            label.grid(row=0, column=10+idx, sticky="nsew", padx=5, pady=5)

    def display_image(self,image="cat"):
        """ This function displays an image using a CTkLabel. """
        self.image=Image.open(f"image//{image}.png")
        self.photo=ctk.CTkImage(self.image,size=(150,200))
        self.image_label=ctk.CTkLabel(self,image=self.photo)
        self.image_label.grid(row=0,column=0,rowspan=10,columnspan=10,sticky="nw",padx=20,pady=20)

    def upload_file(self)->bool:
        """ Upload test data from a file """
        file_path=filedialog.askopenfilename()
        if file_path:
            self.file_name_label.configure(text=file_path)
            self.file_name_label.grid(row=12, column=1, pady=20)
            self.df=pd.read_csv(file_path)
            return True
        return False

    def run(self):
        """ Modify data randomly for demonstration purposes """
        self.view_index=0

        for idx in range(self.df.shape[0]):
            self.data.at[idx,"Test"]=idx
            pass

        self.display_table()

    def view_test_case(self,value:int):
        pass

    def change_algorithm(self,algo:str):
        pass

    def before_before_plot(self):
        if self.view_iteration>5:
            self.view_iteration-=5

        self.display_iteration_label()

    def before_plot(self):
        if self.view_iteration>0:
            self.view_iteration-=1

        self.display_iteration_label()

    def after_plot(self):
        if self.view_iteration<self.data.shape[0]-1:
            self.view_iteration+=1

        self.display_iteration_label()

    def after_after_plot(self):
        if self.view_iteration<self.data.shape[0]-5:
            self.view_iteration+=5

        self.display_iteration_label()

    def display_iteration_label(self):
        self.iteration_label.configure(text=f"{self.view_iteration}/{50}")

app=App()
app.mainloop()