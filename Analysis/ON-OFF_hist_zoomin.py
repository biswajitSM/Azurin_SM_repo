import matplotlib.pyplot as plt
from pylab import *
from Analysis_CuAz_April2017 import histograms
pointnumbers = 31; pot = 18; max_his_on = 0.25; max_his_off = 3; rnge_on = [[0,max_his_on], [0,max_his_on]] #range on histograms, form: [[0,0.25], [0,0.25]] --> [xrange], [yrange]
rnge_off = [[0,max_his_off], [0, max_his_off]] #range off histograms
bins_on = 40; bins_off = 40 #bins for the on histograms;
proteins = 'Cu'
x_shift = 10
current_dir = r'D:\Research\Experimental\Analysis\2017analysis\201702\Analysis_Sebby_March_2017\S101d14Feb17_60.5_635_A2_CuAzu655';
from mpl_toolkits.axes_grid.inset_locator import inset_axes
def inset_hist(axis, df,max_range = 2, bins=100, bins_inset=400,specific_potential=0):
    n_off,bins_off1,patches_off = axis.hist(df, range=(0,max_range),bins=200)
    axis.set_yticks([])
    axis.set_ylabel("#")
#     axis.set_yscale("log")
    axis.set_ylim(0, max(n_off))
    axis.text(0.25, 0.8,str(specific_potential)+' mV', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    insert_ax = inset_axes(axis,height="50%", width="50%")
    n_off,bins_off1,patches_off = insert_ax.hist(df, range=(0,max_range),bins=bins_inset)
    insert_ax.set_xlim(0, 0.1)

    insert_ax.set_yticks([])
    insert_ax.set_xticks([0, 0.1])
    insert_ax.set_ylabel("#")
#     insert_ax.set_xscale("log")
    return(axis, insert_ax)
#----------------------------------------------------#-----------------------------------
fig, axes = plt.subplots(4, 2, figsize=(10, 20), squeeze=False) #Set Figure/Subplot parameters
#0 mV: Define Potential Here
specific_potential = 0
df_on, df_on_shifted, df_on_shifted_x, df_off, df_off_shifted, df_off_shifted_x = histograms(pot, pointnumbers, specific_potential, rnge_on, rnge_off, bins_on, bins_off,
                                                       proteins, current_dir, max_his_on, max_his_off, x_shift,plots=False)
ax, in_ax = inset_hist(axes[0, 0], df_on, max_range = 0.25, bins=100, bins_inset=100)#ON hist-plot
ax.set_xlim(0, 0.25)
ax.text(0.25, 0.8,str(specific_potential)+' mV', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
ax.set_title('ON time histogram')
ax, in_ax = inset_hist(axes[0, 1], df_off, max_range = 2, bins=100, bins_inset=400)#OFF hist-plot
ax.set_xlim(0, 1.5)
ax.text(0.25, 0.8,str(specific_potential)+' mV', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
ax.set_title('OFF time histogram')
in_ax.set_yticks([]); in_ax.set_xticks([0, 0.1])

#----------------------------------------------------#-----------------------------------
#25 mV: Define Potential Here
specific_potential =25
df_on, df_on_shifted, df_on_shifted_x, df_off, df_off_shifted, df_off_shifted_x = histograms(pot, pointnumbers, specific_potential, rnge_on, rnge_off, bins_on, bins_off,
                                                       proteins, current_dir, max_his_on, max_his_off, x_shift,plots=False)
ax, in_ax = inset_hist(axes[1,0], df_on, max_range = 0.25, bins=100, bins_inset=100)#ON hist-plot
ax.set_xlim(0, 0.25)
ax.text(0.25, 0.8,str(specific_potential)+' mV', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
ax, in_ax = inset_hist(axes[1,1], df_off, max_range = 2, bins=100, bins_inset=400)#OFF hist-plot
ax.set_xlim(0, 1.5)
ax.text(0.25, 0.8,str(specific_potential)+' mV', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
ax.set_ylabel("#"); in_ax.set_ylabel("#")

#----------------------------------------------------#-----------------------------------
#50 mV: Define Potential Here
specific_potential =50
df_on, df_on_shifted, df_on_shifted_x, df_off, df_off_shifted, df_off_shifted_x = histograms(pot, pointnumbers, specific_potential, rnge_on, rnge_off, bins_on, bins_off,
                                                       proteins, current_dir, max_his_on, max_his_off, x_shift,plots=False)
ax, in_ax = inset_hist(axes[2,0], df_on, max_range = 0.25, bins=100, bins_inset=100)#ON hist-plot
ax.set_xlim(0, 0.25)
ax.text(0.25, 0.8,str(specific_potential)+' mV', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
ax, in_ax = inset_hist(axes[2,1], df_off, max_range = 2, bins=100, bins_inset=400)#OFF hist-plot
ax.set_xlim(0, 1.5)
ax.text(0.25, 0.8,str(specific_potential)+' mV', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
ax.set_ylabel("#"); in_ax.set_ylabel("#")

#----------------------------------------------------#-----------------------------------
#100 mV: Define Potential Here
specific_potential =100
df_on, df_on_shifted, df_on_shifted_x, df_off, df_off_shifted, df_off_shifted_x = histograms(pot, pointnumbers, specific_potential, rnge_on, rnge_off, bins_on, bins_off,
                                                       proteins, current_dir, max_his_on, max_his_off, x_shift,plots=False)
ax, in_ax = inset_hist(axes[3,0], df_on, max_range = 0.25, bins=100, bins_inset=100)#ON hist-plot
ax.set_xlim(0, 0.25)
ax.text(0.25, 0.8,str(specific_potential)+' mV', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
ax.set_xlabel(r"$\tau{on}[s]$")

ax, in_ax = inset_hist(axes[3,1], df_off, max_range = 2, bins=100, bins_inset=400)#OFF hist-plot
ax.set_xlim(0, 1.5)
ax.text(0.25, 0.8,str(specific_potential)+' mV', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
ax.set_xlabel(r"$\tau{off}[s]$")
in_ax.set_xlim(0, 0.2)
# tight_layout()
