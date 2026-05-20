import numpy as np
import matplotlib.pyplot as plt

def solve_convective_combined():
    print("\n2D Steady State Heat Conduction Solver")
    print("Right Boundary Convection (3 Sides Fixed)")
    print("3 Sides Convection (Left Side Fixed)\n")
    try:
        L = float(input("Enter plate length L (m): "))
        n_div = int(input("Enter divisions (n_div): "))
        k = float(input("Enter conductivity k (W/mK): "))
        h = float(input("Enter convection coeff h (W/m^2K): "))
        t_inf = float(input("Enter fluid temperature T_inf (K): "))
        t_left = float(input("Enter Left boundary temperature (K): "))
        t_bottom = float(input("Enter Bottom boundary temperature (K): "))
        t_top = float(input("Enter Top boundary temperature (K): "))
    except ValueError:
        print("Invalid input.")
        return

    n_nodes = n_div + 1
    total_nodes = n_nodes**2
    dx = L / n_div
    Bi_dx = (h * dx) / k # Biot-based factor
    def get_idx(i, j): return i * n_nodes + j

    # SOLVER 1: RIGHT BOUNDARY ONLY
    A1 = np.zeros((total_nodes, total_nodes))
    B1 = np.zeros(total_nodes)

    for i in range(n_nodes):
        for j in range(n_nodes):
            idx = get_idx(i, j)
            
            # Dirichlet Boundaries (Handling corners with average of temperatures)
            if i == 0 and j == 0: 
                A1[idx, idx] = 1; B1[idx] = (t_bottom + t_left) / 2.0
            elif i == n_nodes - 1 and j == 0: 
                A1[idx, idx] = 1; B1[idx] = (t_top + t_left) / 2.0
            elif i == 0: 
                A1[idx, idx] = 1; B1[idx] = t_bottom
            elif i == n_nodes - 1: 
                A1[idx, idx] = 1; B1[idx] = t_top
            elif j == 0: 
                A1[idx, idx] = 1; B1[idx] = t_left
            
            # Convective Boundary (Right Wall) Node
            elif j == n_nodes - 1:
                # Top-Right Corner (Fixed Top + Convective Right)
                if i == n_nodes - 1:
                    A1[idx, idx] = 1; B1[idx] = t_top # Fix at T_top
                # Bottom-Right Corner (Fixed Bottom + Convective Right)
                elif i == 0:
                    A1[idx, idx] = 1; B1[idx] = t_bottom # Fix at T_bottom
                # Normal Right Wall
                else:
                    A1[idx, idx] = -(4 + 2*Bi_dx)
                    A1[idx, get_idx(i, j-1)] = 2 # Ghost node substitution
                    A1[idx, get_idx(i-1, j)] = 1
                    A1[idx, get_idx(i+1, j)] = 1
                    B1[idx] = -2 * Bi_dx * t_inf
                B1[idx] = -2 * Bi_dx * t_inf
            
            # Interior Nodes
            else:
                A1[idx, idx] = -4
                A1[idx, get_idx(i+1, j)] = 1
                A1[idx, get_idx(i-1, j)] = 1
                A1[idx, get_idx(i, j+1)] = 1
                A1[idx, get_idx(i, j-1)] = 1
                B1[idx] = 0

    T_flat1 = np.linalg.solve(A1, B1)
    T1 = T_flat1.reshape((n_nodes, n_nodes))

    # SOLVER 2: 3 SIDES CONVECTIVE    #Using t_left as the singular fixed side
    A2 = np.zeros((total_nodes, total_nodes))
    B2 = np.zeros(total_nodes)

    for i in range(n_nodes):
        for j in range(n_nodes):
            idx = get_idx(i, j)
            
            # Left Boundary (Fixed Temperature)
            if j == 0: 
                A2[idx, idx] = 1; B2[idx] = t_left
            
            # Bottom-Right Corner (2 Convective boundaries)
            elif j == n_nodes - 1 and i == 0: 
                A2[idx, idx] = -(2 + 2*Bi_dx) # specific 2D corner derivation (k/2 factored)
                A2[idx, get_idx(i+1, j)] = 1 # Interior Top
                A2[idx, get_idx(i, j-1)] = 1 # Interior Left
                B2[idx] = -2 * Bi_dx * t_inf
            
            # Top-Right Corner (2 Convective boundaries)
            elif j == n_nodes - 1 and i == n_nodes - 1: 
                A2[idx, idx] = -(2 + 2*Bi_dx) # specific 2D corner derivation
                A2[idx, get_idx(i-1, j)] = 1 # Interior Bottom
                A2[idx, get_idx(i, j-1)] = 1 # Interior Left
                B2[idx] = -2 * Bi_dx * t_inf
            
            # Bottom Boundary (Convective)
            elif i == 0: 
                A2[idx, idx] = -(4 + 2*Bi_dx)
                A2[idx, get_idx(i+1, j)] = 2 # 2 * T_top ghost substitution
                A2[idx, get_idx(i, j-1)] = 1 # T_left
                A2[idx, get_idx(i, j+1)] = 1 # T_right
                B2[idx] = -2 * Bi_dx * t_inf
            
            # Top Boundary (Convective)
            elif i == n_nodes - 1: 
                A2[idx, idx] = -(4 + 2*Bi_dx)
                A2[idx, get_idx(i-1, j)] = 2 # 2 * T_bottom ghost substitution
                A2[idx, get_idx(i, j-1)] = 1 # T_left
                A2[idx, get_idx(i, j+1)] = 1 # T_right
                B2[idx] = -2 * Bi_dx * t_inf
            
            # Right Boundary (Convective)
            elif j == n_nodes - 1: 
                A2[idx, idx] = -(4 + 2*Bi_dx)
                A2[idx, get_idx(i, j-1)] = 2 # 2 * T_left ghost substitution
                A2[idx, get_idx(i-1, j)] = 1 # T_bottom
                A2[idx, get_idx(i+1, j)] = 1 # T_top
                B2[idx] = -2 * Bi_dx * t_inf
            
            # Interior Nodes
            else: 
                A2[idx, idx] = -4
                A2[idx, get_idx(i+1, j)] = 1
                A2[idx, get_idx(i-1, j)] = 1
                A2[idx, get_idx(i, j+1)] = 1
                A2[idx, get_idx(i, j-1)] = 1
                B2[idx] = 0

    T_flat2 = np.linalg.solve(A2, B2)
    T2 = T_flat2.reshape((n_nodes, n_nodes))

    # Results
    print("")
    print("RESULTS: Case 1 (1-Side Convective Right Wall)")
    print(f"Temperatures at Convective Right Wall (x = {L}):")
    for i in range(n_nodes):
        print(f"Node ({n_nodes-1},{i}): {T1[i, -1]:.2f} K")

    print("")
    print("RESULTS: Case 2 (3-Sides Convective Walls)")
    print(f"Temperatures at Convective Right Wall (x = {L}):")
    for i in range(n_nodes):
        print(f"Node ({n_nodes-1},{i}): {T2[i, -1]:.2f} K")
        
    print(f"Temperatures at Convective Top Wall (y = {L}):")
    for j in range(n_nodes):
        print(f"Node ({j},{n_nodes-1}): {T2[-1, j]:.2f} K")

    print(f"Temperatures at Convective Bottom Wall (y = 0):")
    for j in range(n_nodes):
        print(f"Node ({j},0): {T2[0, j]:.2f} K")

    # Plotting Both Variations
    x_vals = np.linspace(0, L, n_nodes)
    y_vals = np.linspace(0, L, n_nodes)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.canvas.manager.set_window_title("2D Convection Configurations")
    
    # 1. Right Side Setup
    c1 = axs[0].contourf(x_vals, y_vals, T1, cmap='inferno', levels=20)
    axs[0].set_title(f'1 Side Convection (h={h} W/m²K)', weight='bold')
    axs[0].set_xlabel('Position x (m)')
    axs[0].set_ylabel('Position y (m)')
    fig.colorbar(c1, ax=axs[0], fraction=0.046, pad=0.04, label='Temperature (K)')
    
    # 2. 3-Sides Setup
    c2 = axs[1].contourf(x_vals, y_vals, T2, cmap='inferno', levels=20)
    axs[1].set_title(f'3 Sides Convection (h={h} W/m²K)', weight='bold')
    axs[1].set_xlabel('Position x (m)')
    fig.colorbar(c2, ax=axs[1], fraction=0.046, pad=0.04, label='Temperature (K)')
    fig.tight_layout()
    def add_plot_annotations(ax, T_data):
        ax.grid(color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xticks(x_vals); ax.set_yticks(y_vals)
        for i in range(n_nodes):
            for j in range(n_nodes):
                ha = 'left' if j == 0 else 'right' if j == n_nodes - 1 else 'center'
                va = 'bottom' if i == 0 else 'top' if i == n_nodes - 1 else 'center'
                xo, yo = (4 if j == 0 else -4 if j == n_nodes - 1 else 0), (4 if i == 0 else -4 if i == n_nodes - 1 else 0)
                ax.annotate(f"({j},{i})\n{T_data[i,j]:.0f}", xy=(x_vals[j], y_vals[i]), xytext=(xo, yo),
                            textcoords='offset points', ha=ha, va=va, fontsize=8, color='black',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))
    add_plot_annotations(axs[0], T1)
    add_plot_annotations(axs[1], T2)
    plt.show()

if __name__ == "__main__":
    solve_convective_combined()
    