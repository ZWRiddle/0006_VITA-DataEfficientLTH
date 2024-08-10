
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # Use the TkAgg backend
import os
import datetime

def plotlwd(data,t,x,name):
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(16, 12))
    # line, = ax.plot([], [], lw=2)
    plt.title(name)
    # plt.subplots_adjust(bottom=0.2, left=0.2,right=0.8, top=0.8)
    
    # Define initialization function for animation
    def init():        
        X = np.arange(x)
        ax.set_xlim(0, x)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Epochs')     
        bars = ax.bar(X, np.zeros(x))  # Initialize empty bars
        return bars

    def update(frame):
        X = np.arange(x)
        Y = data[frame, :]["Density"]  # Extract data for the first entry of the third dimension
        # line.set_data(X, Y)
        bars = ax.bar(X, Y)  # Create block plot
        # Clear old annotations
        for text in ax.texts:
            text.remove()
        # Add new annotations
        for i in range(len(X)):
            ax.annotate(f'{data[frame, i]["Active"]}', (X[i], Y[i]), xytext=(5, 5), textcoords='offset points')
        # Show current state
        text_str = f'State: {frame + 1}'  # Assuming 'frame' is the current frame index
        ax.text(0.95, 0.95, text_str, transform=ax.transAxes, ha='right', va='top')

        setticks = 0
        if setticks == 0:
            # Set X-axis ticks and labels
            ax.set_xticks(X)
            ax.set_xticklabels(lwd[0, :]['Name'], rotation=90, ha='right')
            # Set X-axis ticks and labels for ax2
            ax2 = ax.twiny()
            ax2.set_xlabel('Total Neurons in the Layer')    
            ax2.set_xticks(X)
            ax2.set_xticklabels(lwd[0, :]['Total'], rotation=45, ha='right')  
            ax.figure.tight_layout
            setticks = 1
         
        return bars

    ani = FuncAnimation(fig, update, frames=range(t), init_func=init, blit=True,interval=1024)
    
    # Display the animation
    # plt.tight_layout()
    print(f"saving LWD ...")
    ani.save(name, writer='pillow')

def plotlc(data,t,x,name):
    # Create figure and axis objects
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16, 12),sharex=1,sharey=1)
    plt.title(name)
    axs = (ax1,ax2,ax3)
    

    # Define initialization function for animation
    def init():
        lines = []
        for ax in axs:
            X = np.arange(x)
            ax.set_xlim(0, x)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Epochs')
            line, = ax.plot([], [], lw=2)
            lines.append(line)
        return lines        

    def update(frame):
        lines = []
        j = 1
        for ax in axs:
            X = np.arange(x)
            # Assuming the second entry is Best_Train_Acc, third entry is Best_Val_Acc, and fourth entry is LR
            Y = data[frame, :, j]  # You might need to adjust the index based on your data structure
            line, = ax.plot(X, Y, lw=1,label=frame)
            lines.append(line)
            j += 1

            # Clear old annotations
            for text in ax.texts:
                text.remove()
            # Show current state
            text_str = f'State: {frame + 1}'  # Assuming 'frame' is the current frame index
            ax.text(0.95, 0.95, text_str, transform=ax.transAxes, ha='right', va='top')
            ax.legend()

        return lines

    ani = FuncAnimation(fig, update, frames=range(t), init_func=init, blit=True,interval=1024)
    
    # Display the animation
    print(f"saving LC ...")
    ani.save(name, writer='pillow')


print(datetime.datetime.now())    

log_addr = "plotting_data/01-27_12-26_0.02_0.02_autoaug-1.txt"
name = "0.02_0.02_autoaug-1"

nepoch = 160
nstate = 16
nlayer = 20

dt_lwd = np.dtype([('Name', 'U10'), ('Active', int), ('Total', int), ('Density', float)])
lwd = np.zeros((nstate,nlayer), dtype=dt_lwd) # Name | Active | Total | Density
learning_curve = np.zeros((nstate,nepoch,4)) # Epoch | Train_Acc | Val_Acc | lr

with open(log_addr, 'r') as file:

    curr_epoch = 0 # initiate epoch counter for switching states
    curr_state_lc = 0 # initiate current state for learning curve
    curr_layer = 0 # initiate layer counter for switching states
    curr_state_lwd = 0 # initiate current state for layer wise density

    for line in file:
        
        if " 	|	active =  " in line:

            if curr_layer == nlayer:
                curr_layer = 0
                curr_state_lwd += 1

            parts = line.split("|")
            for part in parts:
                if "active" in part:
                    lwd[curr_state_lwd,curr_layer]['Active'] = int(part.split("=")[1].strip())
                elif "total" in part:
                    lwd[curr_state_lwd,curr_layer]['Total'] = int(part.split("=")[1].strip())
                elif "layerwise sparsity" in part:
                    lwd[curr_state_lwd,curr_layer]['Density'] = float(part.split("=")[1].strip())
                else:
                    lwd[curr_state_lwd,curr_layer]['Name'] = str(part)
            curr_layer += 1
            if curr_layer > nlayer:
                print(f"ERROR: curr_layer > nlayer!")
                exit

        if ", best train: " in line:

            if curr_epoch == nepoch:
                curr_epoch = 0
                curr_state_lc += 1

            parts = line.split(",")
            for part in parts:
                if "epoch" in part:
                    learning_curve[curr_state_lc,curr_epoch,0] = int(part.split(":")[1].strip())
                elif "best train" in part:
                    learning_curve[curr_state_lc,curr_epoch,1] = float(part.split(":")[1].strip())
                elif "best val" in part:
                    learning_curve[curr_state_lc,curr_epoch,2] = float(part.split(":")[1].strip())
                elif "lr" in part:
                    learning_curve[curr_state_lc,curr_epoch,3] = float(part.split(":")[1].strip())
            curr_epoch += 1
            if curr_epoch > nepoch:
                print(f"ERROR: curr_epoch > nepoch!")
                exit
            
lwd_name = "plots/LWD_plots/" + name + "_LWD.gif"
plotlwd(lwd,nstate,nlayer,lwd_name)

lc_name = "plots/LC_plots/" + name + "_LC.gif"
plotlc(learning_curve,nstate,nepoch,lc_name)

        
