from modules.linear_algebra import LinearAlgebra
from modules.algorithms import Algorithms
from modules.probability import Probability

def main():
    
    try:
        Algorithms().logistic_regression_dataset()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Some error occured: {e}")

if __name__ == '__main__':
    main()