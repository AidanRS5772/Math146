import numpy as np
import matplotlib.pyplot as plt

ofs = 3
n = 7
qA = []

for i in range(ofs, n + ofs):
    print(i)
    N = 2 ** i - 1

    blFDL = np.diag(np.ones(N - 1), k=1) + np.diag(np.ones(N - 1), k=-1) - 4 * np.eye(N)
    FDL = np.kron(np.eye(N), blFDL) + np.diag(np.ones(N * (N - 1)), k=N) + np.diag(np.ones(N * (N - 1)), k=-N)

    x, y = np.meshgrid(np.linspace(1 / (N + 2), 1 - 1 / (N + 2), N), np.linspace(1 / (N + 2), 1 - 1 / (N + 2), N))
    f = -(1 / (N + 2) ** 2) * np.exp((x - 1 / 4) ** 2 + (y - 1 / 4) ** 2)
    f = f.reshape((N ** 2, 1))

    sol = np.linalg.solve(FDL, f).reshape((N, N))
    sol = np.vstack((np.zeros((1, N)), sol, np.zeros((1, N))))
    sol = np.hstack((np.zeros((N + 2, 1)), sol, np.zeros((N + 2, 1))))
    qA.append(sol)

for i in range(ofs, n + ofs):
    N = 2 ** i - 1
    x, y = np.meshgrid(np.linspace(0, 1, N + 2), np.linspace(0, 1, N + 2))
    fig = plt.figure(f'N = {i - ofs + 1}')
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, qA[i - ofs], cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
plt.show()
