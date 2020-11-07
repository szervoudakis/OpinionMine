import os
import tkinter as tk
import glob
from os import listdir
from os.path import isfile, join
from tkinter import scrolledtext, INSERT, messagebox
from tkinter.ttk import Combobox
from tkinter import *
from createModel import createModel
from evaluationOfNewTweet import evaluationOfNewTweet

LARGE_FONT = ("Verdana", 16)

class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Home", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        frame = tk.Frame(self)
        frame.pack()

        bottomframe = tk.Frame(self)
        bottomframe.pack(side=tk.BOTTOM)
        trainButton = tk.Button(frame, text="Train Model", fg="black", command=lambda: controller.show_frame(PageOne))
        trainButton.pack(side=tk.LEFT, pady=30)
        testButton = tk.Button(frame, text="Trained Model", fg="black", command=lambda: controller.show_frame(PageTwo))
        testButton.pack(side=tk.LEFT, pady=30,padx=10)
        exitButton = tk.Button(frame, text="     Exit     ", fg="red", command=lambda: quit())
        exitButton.pack(side=tk.LEFT, pady=30, padx=10)


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Train Model", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        message = ["Chose the trainset"]
        onlyfiles = [f for f in listdir('datasets/') if isfile(join('datasets/', f))]
        listOfOptions = message + onlyfiles

        combo = Combobox(self)
        combo['values'] = listOfOptions
        combo.current(0)
        txt = scrolledtext.ScrolledText(self, width=120, height=30)
        frame = tk.Frame(self)
        frame.pack()
        bottomframe = tk.Frame(self)
        bottomframe.pack(side=tk.BOTTOM)
        homeButton = tk.Button(frame, text="Back to Home", fg="green", command=lambda: controller.show_frame(StartPage))
        homeButton.pack(side=tk.LEFT, pady=30)
        incrementalButton = tk.Button(frame, text="Incremental Learning", fg="black",
                                command=lambda: beginTrain(str(combo.get()), message, txt, label1))
        incrementalButton.pack(side=tk.LEFT, pady=30, padx=10)
        clearButton = tk.Button(frame, text="Clear", fg="black",
                                command=lambda: deleteModel(txt,label1))
        clearButton.pack(side=tk.LEFT, pady=30, padx=10)
        txt.config(state='normal')
        if os.path.exists('models/model.txt'):
         with open("models/model.txt", "r") as myfile:
            rules = myfile.readlines()
         rulesInsert = ""
         for i in range(1,len(rules)):
            rulesInsert = rulesInsert + rules[i] + "\n"
         txt.insert(INSERT, rulesInsert)
         txt.config(state=DISABLED)
         if len(rules)>0:
            mes="Τhe model was trained with "+rules[0]+ "samples"
         else:
            mes="No training on the model"
        else:
            mes = ""
            txt.insert(INSERT, "Τhere is no trained model")

        label1 = tk.Label(self, text=mes, font=("Arial Bold", 8))
        label1.pack(pady=10, padx=10)
        combo.pack()
        clearButton.pack()
        homeButton.pack()
        txt.pack()

class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Trained Model", font=LARGE_FONT)
        label1= tk.Label(self, text="Please insert new tweet",font=("Arial Bold", 10))
        label.pack(pady=10, padx=10)
        label1.pack(pady=16, padx=10)

        txt = scrolledtext.ScrolledText(self, width=60, height=8)
        txt.pack()
        label2 = tk.Label(self, text="Please insert your Account location (user's home location)", font=("Arial Bold", 10))
        label2.pack(pady=16, padx=10)
        txtLocation = scrolledtext.ScrolledText(self, width=60, height=0.15)
        txtLocation.pack()
        label3 = tk.Label(self, text="Opinion for tweet", font=("Arial Bold", 10))

        txtForOpinion = scrolledtext.ScrolledText(self, width=60, height=15)
        frame = tk.Frame(self)
        frame.pack()
        bottomframe = tk.Frame(self)
        bottomframe.pack(side=tk.BOTTOM)
        homebutton = tk.Button(frame, text="Back to Home", fg="green", command=lambda: controller.show_frame(StartPage))
        homebutton.pack(side=tk.LEFT,pady=30)
        predictButton = tk.Button(frame, text="Evaluate", fg="blue", command=lambda:showPrediction(txt, txtLocation, txtForOpinion))
        predictButton.pack(side=tk.LEFT,pady=30,padx=10)
        clearbutton = tk.Button(frame, text="Clear", fg="black", command=lambda: clearFields(txt, txtLocation, txtForOpinion))
        clearbutton.pack(side=tk.LEFT, pady=30)
        label3.pack(pady=16, padx=10)
        txtForOpinion.pack()

def deleteModel(txt,lbl):
  if os.path.exists('models/model.txt'):
    txt.configure(state='normal')
    txt.delete(1.0,tk.END)
    txt.insert(1.0,"Τhere is no trained model")
    txt.configure(state=DISABLED)
    path = 'models/*'
    r = glob.glob(path)
    for i in r:
       os.remove(i)
    lbl['text']="there is no trained model"
def clearFields(txt,txtLocation,txtForOpinion):
    txtForOpinion.configure(state='normal')
    txt.delete(1.0, tk.END)
    txt.insert(1.0, "")
    txtLocation.delete(1.0, tk.END)
    txtLocation.insert(1.0, "")
    txtForOpinion.delete(1.0, tk.END)
    txtForOpinion.insert(1.0, "")
    txtForOpinion.config(state=DISABLED)

def showPrediction(txt,txtLoc,txtOp):
    fetched_content = txt.get('1.0', 'end-1c')
    fetched_content1 = txtLoc.get('1.0', 'end-1c')


    if fetched_content == "" or fetched_content1=="":
        messagebox.showinfo('Message', 'Please fill in the form')

    else:
       result = evaluationOfNewTweet(fetched_content, fetched_content1)
       print(result)
       per = float(result*100.0)
       strPer=str(per)
       if result > 0.50:
          if result > 0.70:
              opinionForPercentage=" based on the tweet posted there is a high probability that the user visited Crete or will visit Crete in the near future "
              opinionForVisitLocation=" positive evaluation"
          else:
              opinionForPercentage = " based on the tweet posted there is a probability that the user visited Crete or will visit Crete in the near future "
              opinionForVisitLocation = " positive evaluation"
       else:
           if result <= 0.40:
            opinionForPercentage = " based on the tweet posted there is a low probability that the user visited Crete or will visit Crete in the near future"
            opinionForVisitLocation = " negative evaluation"
           else:
               opinionForPercentage = " the probability of visiting Crete is close to 50% but the opinion we extract is that the user will not visit Crete"
               opinionForVisitLocation = " negative evaluation"

       finalMes="Opinion :  "+opinionForPercentage+"\n"+"\n"+"Visit Crete with probability : "+str(strPer[0]+strPer[1]+strPer[2]+strPer[3])+"%"
       txtOp.config(state='normal')
       txtOp.delete(1.0,tk.END)
       txtOp.insert(1.0,finalMes)
       txtOp.config(state=DISABLED)

def beginTrain(selectedValue,message,txt,lbl):
    if (selectedValue == message[0]):
        messagebox.showinfo('Message title', 'Please insert the dataset that we use in the train proccess ')
    else:
        txt.config(state='normal')
        txt.delete(1.0, tk.END)
        createModel(selectedValue)
        with open("models/model.txt", "r") as myfile:
            rules = myfile.readlines()
        rulesInsert = ""
        for i in range(1,len(rules)):
            rulesInsert = rulesInsert + rules[i] + "\n"
        txt.insert(1.0,rulesInsert)
        lbl['text']="Τhe model was trained with "+rules[0]+ "samples"
        txt.config(state=DISABLED)
        messagebox.showinfo('Success', 'Τhe models learning was successful ')
app = SeaofBTCapp()
app.mainloop()