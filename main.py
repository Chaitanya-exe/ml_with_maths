from modules.linear_algebra import LinearAlgebra
from modules.algorithms import Algorithms

def main():
    
    try:
        Algorithms().batch_gradient_descent()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Some error occured{e}")

if __name__ == '__main__':
    main()