from mpl_toolkits.axes_grid1 import make_axes_locatable

# State visualize function 
def visualize_matrix(M,strs='',
                     fontsize=15,
                     cmap='turbo',
                     title='Title',
                     title_fs=15,
                     REMOVE_TICK_LABELS=True):
    """
    Visualize a matrix colors and strings 
    """
    
    n_row,n_col = M.shape[0],M.shape[1]
    fig,ax = plt.subplots()
    divider = make_axes_locatable(ax)
    im = ax.imshow(M,cmap=plt.get_cmap(cmap),extent=(0,n_col,n_row,0),
              interpolation='nearest',aspect='equal')
    ax.set_xticks(np.arange(0,n_col,1))
    ax.set_yticks(np.arange(0,n_row,1))
    ax.grid(color='w', linewidth=2)
    ax.set_frame_on(False)
    x,y = np.meshgrid(np.arange(0,n_col,1.0),np.arange(0,n_row,1.0))
    if len(strs) == n_row*n_col:
        idx = 0
        for x_val,y_val in zip(x.flatten(), y.flatten()):
            c = strs[idx]
            idx = idx + 1
            ax.text(x_val+0.5,y_val+0.5,c,va='center', ha='center',size=fontsize)
    cax = divider.append_axes('right', size='5%', pad=0.05)            
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.suptitle(title,size=title_fs) 
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if REMOVE_TICK_LABELS:
        plt.setp(ax.get_xticklabels(),visible=False)
        plt.setp(ax.get_yticklabels(),visible=False)
    plt.show()

# Value visualize function
def plot_pi_v(Pi,V,
              title='Value Function',
              cmap='viridis',
              title_fs=15,
              REMOVE_TICK_LABELS=True):
    """
    Visualize pi and V
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    n_row,n_col = V.shape[0],V.shape[1]
    fig,ax = plt.subplots()
    divider = make_axes_locatable(ax)
    im = ax.imshow(V,cmap=plt.get_cmap(cmap),extent=(0,n_col,n_row,0))
    ax.set_xticks(np.arange(0,n_col,1))
    ax.set_yticks(np.arange(0,n_row,1))
    ax.grid(color='w', linewidth=2)
    arr_len = 0.2
    for i in range(4):
        for j in range(4):
            s = i*4+j
            if Pi[s][0]> 0: plt.arrow(j+0.5,i+0.5,-arr_len,0,
                          color="r",alpha=Pi[s][0],width=0.01,
                          head_width=0.5,head_length=0.2,overhang=1)
            if Pi[s][1]> 0: plt.arrow(j+0.5,i+0.5,0,arr_len,
                          color="r",alpha=Pi[s][1],width=0.01,
                          head_width=0.5,head_length=0.2,overhang=1)
            if Pi[s][2]> 0: plt.arrow(j+0.5,i+0.5,arr_len,0,
                          color="r",alpha=Pi[s][2],width=0.01,
                          head_width=0.5,head_length=0.2,overhang=1)
            if Pi[s][3]> 0: plt.arrow(j+0.5,i+0.5,0,-arr_len,
                          color="r",alpha=Pi[s][3],width=0.01,
                          head_width=0.5,head_length=0.2,overhang=1)
    cax = divider.append_axes('right', size='5%', pad=0.05)            
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.suptitle(title,size=title_fs) 
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if REMOVE_TICK_LABELS:
        plt.setp(ax.get_xticklabels(),visible=False)
        plt.setp(ax.get_yticklabels(),visible=False)
    plt.show()

# Policy visualize function 
def plot_policy(Pi, title='Policy', REMOVE_TICK_LABELS=True):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig,ax = plt.subplots()
    divider = make_axes_locatable(ax)
    im = ax.imshow(np.ones((4,4,3)),extent=(0,4,4,0),cmap='Reds')
    ax.set_xticks(np.arange(0,4,1))
    ax.set_yticks(np.arange(0,4,1))
    ax.grid(color='k', linewidth=1)
    ax.set_frame_on(False)
    arr_len = 0.2
    for i in range(4):
        for j in range(4):
            s = i*4+j
            if Pi[s][0]> 0:
                plt.arrow(j+0.5,i+0.5,-arr_len,0,color="r",alpha=Pi[s][0],
                          width=0.01,head_width=0.5,head_length=0.2,overhang=1)
            if Pi[s][1]> 0:
                plt.arrow(j+0.5,i+0.5,0,arr_len,color="r",alpha=Pi[s][1],
                          width=0.01,head_width=0.5,head_length=0.2,overhang=1)
            if Pi[s][2]> 0:
                plt.arrow(j+0.5,i+0.5,arr_len,0,color="r",alpha=Pi[s][2],
                          width=0.01,head_width=0.5,head_length=0.2,overhang=1)
            if Pi[s][3]> 0:
                plt.arrow(j+0.5,i+0.5,0,-arr_len,color="r",alpha=Pi[s][3],
                          width=0.01,head_width=0.5,head_length=0.2,overhang=1)
    cax = divider.append_axes('right', size='5%', pad=0.05)            
    im.set_clim(vmin=0,vmax=1) # Reds
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.suptitle(title,size=15) 
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if REMOVE_TICK_LABELS:
        plt.setp(ax.get_xticklabels(),visible=False)
        plt.setp(ax.get_yticklabels(),visible=False)
    plt.show()