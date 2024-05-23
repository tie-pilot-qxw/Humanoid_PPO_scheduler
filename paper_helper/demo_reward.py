from matplotlib import pyplot as plt
import numpy as np

def main():
    x = np.linspace(-10, 10, 100)
    y = -x**2 + 10
    plt.plot(x, y, color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    circle = plt.Circle((0, 10), 1, color='red', fill=False)
    plt.gca().add_patch(circle)
    plt.text(0, 10, 'desired area', ha='center', va='bottom')
    plt.show()
    plt.close()

    x = np.linspace(-4, 4, 100)
    y = -x**2 + 10
    plt.plot(x, y, color='blue')
    x1 = np.linspace(-10, -4, 100)
    y1 = -(x1 + 6)**2 - 2
    plt.plot(x1, y1, color = 'blue')
    x2 = np.linspace(4, 10, 100)
    y2 = -(x2 - 6)**2 -2
    plt.plot(x2, y2, color = 'blue')

    plt.xlabel('x')
    plt.ylabel('y')
    circle1 = plt.Circle((-6, -2), 1, color='red', fill=False)
    plt.gca().add_patch(circle1)
    plt.text(-6, -2, 'suboptimal', ha='center', va='bottom')
    circle2 = plt.Circle((6, -2), 1, color='red', fill=False)
    plt.gca().add_patch(circle2)
    plt.text(6, -2, 'suboptimal', ha='center', va='bottom')
    plt.show()


if __name__ == '__main__':
    main()