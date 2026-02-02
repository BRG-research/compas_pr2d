from IPython.display import display, Math


def print_matrix(array):
    matrix = ""
    for row in array:
        try:
            for number in row:
                matrix += f"{number}&"
        except TypeError:
            matrix += f"{row}&"
        matrix = matrix[:-1] + r"\\"
    display(Math(r"\begin{bmatrix}" + matrix + r"\end{bmatrix}"))
