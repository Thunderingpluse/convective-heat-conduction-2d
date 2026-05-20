import numpy as np
import matplotlib.pyplot as plt

def solve_exp4c():
    print("\n2D Conduction Solver (1 Side Fixed, 3 Sides Convective)")
    
    # Inputs
    try:
        L = float(input("Enter plate length L (m): "))
        n_div = int(input("Enter divisions (n_div): "))
        k = float(input("Enter conductivity k (W/mK): "))
        h = float(input("Enter convection coeff h (W/m^2K): "))
        t_inf = float(input("Enter fluid temperature T_inf (K): "))
        t_fixed = float(input("Enter fixed boundary temperature, Left side (K): "))
    except ValueError:
        print("Invalid input.")
        return

    n_nodes = n_div + 1
    total_nodes = n_nodes**2
    dx = L / n_div
    Bi_dx = (h * dx) / k # Biot-based factor
    
    A = np.zeros((total_nodes, total_nodes))
    B = np.zeros(total_nodes)
    def get_idx(i, j): return i * n_nodes + j

    for i in range(n_nodes):
        for j in range(n_nodes):
            idx = get_idx(i, j)
            
            # Left Boundary (Fixed Temperature)
            if j == 0: 
                A[idx, idx] = 1; B[idx] = t_fixed
            
            # Bottom-Right Corner (2 Convective boundaries)
            elif j == n_nodes - 1 and i == 0: 
                A[idx, idx] = -(4 + 4*Bi_dx)
                A[idx, get_idx(i+1, j)] = 2 # 2 * T_top ghost substitution
                A[idx, get_idx(i, j-1)] = 2 # 2 * T_left ghost substitution
                B[idx] = -4 * Bi_dx * t_inf
            
            # Top-Right Corner (2 Convective boundaries)
            elif j == n_nodes - 1 and i == n_nodes - 1: 
                A[idx, idx] = -(4 + 4*Bi_dx)
                A[idx, get_idx(i-1, j)] = 2 # 2 * T_bottom ghost substitution
                A[idx, get_idx(i, j-1)] = 2 # 2 * T_left ghost substitution
                B[idx] = -4 * Bi_dx * t_inf
            
            # Bottom Boundary (Convective)
            elif i == 0: 
                A[idx, idx] = -(4 + 2*Bi_dx)
                A[idx, get_idx(i+1, j)] = 2 # 2 * T_top ghost substitution
                A[idx, get_idx(i, j-1)] = 1 # T_left
                A[idx, get_idx(i, j+1)] = 1 # T_right
                B[idx] = -2 * Bi_dx * t_inf
            
            # Top Boundary (Convective)
            elif i == n_nodes - 1: 
                A[idx, idx] = -(4 + 2*Bi_dx)
                A[idx, get_idx(i-1, j)] = 2 # 2 * T_bottom ghost substitution
                A[idx, get_idx(i, j-1)] = 1 # T_left
                A[idx, get_idx(i, j+1)] = 1 # T_right
                B[idx] = -2 * Bi_dx * t_inf
            
            # Right Boundary (Convective)
            elif j == n_nodes - 1: 
                A[idx, idx] = -(4 + 2*Bi_dx)
                A[idx, get_idx(i, j-1)] = 2 # 2 * T_left ghost substitution
                A[idx, get_idx(i-1, j)] = 1 # T_bottom
                A[idx, get_idx(i+1, j)] = 1 # T_top
                B[idx] = -2 * Bi_dx * t_inf
            
            # Interior Nodes
            else: 
                A[idx, idx] = -4
                A[idx, get_idx(i+1, j)] = 1
                A[idx, get_idx(i-1, j)] = 1
                A[idx, get_idx(i, j+1)] = 1
                A[idx, get_idx(i, j-1)] = 1
                B[idx] = 0

    # Print Equations Table
    print("\nGenerated Equation Table")
    print(f"{'NODE':<12} | {'EQUATION':<65} | {'RHS':<12}")
    for i in range(n_nodes):
        for j in range(n_nodes):
            node_str = f"({j},{i})"
            
            if j == 0:
                eq = f"1*T({j},{i})"
                rhs = t_fixed
            elif j == n_nodes - 1 and i == 0:
                eq = f"-(4 + 4*Bi_dx)*T({j},{i}) + 2*T({j},{i+1}) + 2*T({j-1},{i})"
                rhs = -4 * Bi_dx * t_inf
            elif j == n_nodes - 1 and i == n_nodes - 1:
                eq = f"-(4 + 4*Bi_dx)*T({j},{i}) + 2*T({j},{i-1}) + 2*T({j-1},{i})"
                rhs = -4 * Bi_dx * t_inf
            elif i == 0:
                eq = f"-(4 + 2*Bi_dx)*T({j},{i}) + 2*T({j},{i+1}) + T({j-1},{i}) + T({j+1},{i})"
                rhs = -2 * Bi_dx * t_inf
            elif i == n_nodes - 1:
                eq = f"-(4 + 2*Bi_dx)*T({j},{i}) + 2*T({j},{i-1}) + T({j-1},{i}) + T({j+1},{i})"
                rhs = -2 * Bi_dx * t_inf
            elif j == n_nodes - 1:
                eq = f"-(4 + 2*Bi_dx)*T({j},{i}) + 2*T({j-1},{i}) + T({j},{i-1}) + T({j},{i+1})"
                rhs = -2 * Bi_dx * t_inf
            else:
                eq = f"-4*T({j},{i}) + T({j+1},{i}) + T({j-1},{i}) + T({j},{i+1}) + T({j},{i-1})"
                rhs = 0
            
            print(f"{node_str:<12} | {eq:<65} | {rhs:<12.2f}")

    T_flat = np.linalg.solve(A, B)
    T = T_flat.reshape((n_nodes, n_nodes))

    # Results for convective boundaries
    print(f"\nTemperatures at Convective Right Wall (x = {L}):")
    for i in range(n_nodes):
        print(f"Node ({n_nodes-1},{i}): {T[i, -1]:.2f} K")
        
    print(f"\nTemperatures at Convective Top Wall (y = {L}):")
    for j in range(n_nodes):
        print(f"Node ({j},{n_nodes-1}): {T[-1, j]:.2f} K")

    print(f"\nTemperatures at Convective Bottom Wall (y = 0):")
    for j in range(n_nodes):
        print(f"Node ({j},0): {T[0, j]:.2f} K")

    # Plotting
    x_vals = np.linspace(0, L, n_nodes)
    y_vals = np.linspace(0, L, n_nodes)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(10, 8))
    plt.contourf(x_vals, y_vals, T, cmap='inferno', levels=20)
    plt.colorbar(label='Temperature (K)')
    
    # Add grid lines to match nodes
    plt.xticks(x_vals)
    plt.yticks(y_vals)
    plt.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)

    for i in range(n_nodes):
        for j in range(n_nodes):
            ha = 'left' if j == 0 else 'right' if j == n_nodes - 1 else 'center'
            va = 'bottom' if i == 0 else 'top' if i == n_nodes - 1 else 'center'
            xo, yo = (5 if j == 0 else -5 if j == n_nodes - 1 else 0), (5 if i == 0 else -5 if i == n_nodes - 1 else 0)
            plt.annotate(f"({j},{i})\n{T[i,j]:.0f} K", xy=(x_vals[j], y_vals[i]), xytext=(xo, yo),
                         textcoords='offset points', ha=ha, va=va, fontsize=9, color='black',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))

    plt.title(f'1 Side Fixed Area & 3 Sides Convection (h={h} W/m²K)', weight='bold')
    plt.xlabel('Position x (m)', weight='bold')
    plt.ylabel('Position y (m)', weight='bold')
    plt.show()

if __name__ == "__main__":
    solve_exp4c()
