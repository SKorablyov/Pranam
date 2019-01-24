import os,sys,time
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


db_root = "./results"
runfiles = [f for f in os.listdir(db_root) if os.path.isfile(os.path.join(db_root, f))]

for runfile in runfiles:
    dframe = pd.read_csv(os.path.join(db_root,runfile),header=1)
    print dframe


print runfiles
time.sleep(100)


#out_path = "/home/viacheslav/Documents/9520_final/Plots"
#folders = ["cfg4_30","cfg4_31","cfg4_32","cfg4_33","cfg4_34","cfg4_35","cfg4_36","cfg4_37","cfg4_38","cfg4_39"]
#folders= ["cfg4_a11","cfg4_a2","cfg4_a12","cfg4_b1","cfg4_b2","cfg4_b12","cfg4_c1","cfg4_c2","cfg_em12"]
#folders= ["cfg4_c1","cfg4_c2","cfg_em12"]
#folders=["1","2","3","4"]

#from os import listdir
#from os.path import isfile, join


runs_means = []
runs_filts = []
for folder_name in folders:
    folder = os.path.join(db_root, folder_name)
    runfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    runs_data = []
    for runfile in runfiles:
        run_data = np.loadtxt(os.path.join(folder,runfile))
        runs_data.append(run_data)
    # work on data
    runs_data = np.asarray(runs_data)
    runs_mean = np.mean(runs_data,axis=0)

    runs_filt = savgol_filter(runs_mean,101,4)
    runs_means.append(runs_mean)
    runs_filts.append(runs_filt)

#print(runs_filts)

X = np.arange(0,100000) * 0.001

fig = plt.figure()
matplotlib.rcParams.update({'font.size': 18})
fig.set_size_inches(12.8, 12.8)
ax = fig.gca(yscale="log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")



# plot SGD

plt.plot(X,runs_filts[0]) # color='c', alpha=0.3

plt.plot(X,runs_filts[1])


plt.plot(X,runs_filts[2])


plt.plot(X,runs_filts[3]) # color='c', alpha=0.3

"""
plt.plot(X,runs_filts[4])

plt.plot(X,runs_filts[5])

plt.plot(X,runs_filts[5])

plt.plot(X,runs_filts[6]) # color='c', alpha=0.3
plt.plot(X,runs_filts[7])
plt.plot(X,runs_filts[8])

"""
#ax.legend(["Em-256 1,0","Em-256 0,1","Em-256 1,1","Em-64 1,0","Em-64 0,1","Em-64 1,1","Em-1 1,0","Em-1 0,1","Em-1 1,1"])
ax.legend(["Deep embedding test accuracy ","Deep embedding train accuracy","1x8192 embedding test accuracy ","1x8192 embedding train accuracy",])
#,prop={'size': 12})


plt.savefig(os.path.join(out_path,  "Test+Train3.png"))
plt.close()



# # plot SGD
# plt.plot(X,runs_filts[8],color="C0") # color='c', alpha=0.3
# plt.plot(X,runs_filts[9],color="C0",alpha=0.5)
# # plot ADAM
# plt.plot(X,runs_filts[2],color="C1")
# plt.plot(X,runs_filts[3],color="C1",alpha=0.5)
# # plot RMSprop
# plt.plot(X,runs_filts[4],color="C2")
# plt.plot(X,runs_filts[5],color="C2",alpha=0.5)
# # plot AdaDelta
# plt.plot(X,runs_filts[6],color="C3")
# plt.plot(X,runs_filts[7],color="C3",alpha=0.5)
#
# ax.legend(["SGD","Pranam-SGD","Adam","Pranam-Adam","RMSProp","Pranam-RMSProp","AdaDelta","Pranam-AdaDelta"])#,prop={'size': 12})
# plt.savefig(os.path.join(out_path,  "optimizers.png"))
# plt.close()




# plot some 2d stuff:
#            fig = plt.figure()
#            fig.set_size_inches(12.8, 12.8)
#            ax = fig.gca(projection='3d')
#            x = np.linspace(-1, 1, 100)
#            y = np.linspace(-1, 1, 100)
#            xv, yv = np.meshgrid(x, y)
#            zv = the_function(torch.from_numpy(np.stack([xv,yv],1))).numpy()
#            surf = ax.plot_surface(xv,yv, zv,rstride=1, cstride=1, cmap=cm.coolwarm, color='c', alpha=0.3, linewidth=0)
#            xys = network().detach().numpy()
#            zs = the_function(network()).detach().numpy()
#            #print zs
#            ax.scatter(xys[:,0],xys[:,1],zs,color="k", s=50)
#            if not os.path.exists(os.path.join(args.save_dir,str(lr))): os.makedirs(os.path.join(args.save_dir,str(lr)))
#            plt.savefig(os.path.join(args.save_dir,str(lr),"surf_" + str(i) + ".png"))
#            plt.close()

