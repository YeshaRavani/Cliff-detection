try:
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    from numba import jit
    from sklearn.preprocessing import StandardScaler

except Exception as e:
    print(f"Some module are missing: {e}\n")

PERIODIC_TABLE = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}


def read_xyz(filename: Path):
    """Legge un file XYZ e restituisce i numeri atomici e le coordinate."""
    periodic_table = PERIODIC_TABLE

    with open(str(filename), "r") as f:
        lines = f.readlines()
        num_atoms = int(lines[0].strip())
        atom_data = lines[2 : 2 + num_atoms]

    atomic_numbers = []
    coordinates = []

    for line in atom_data:
        parts = line.split()
        element = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atomic_numbers.append(periodic_table[element])
        coordinates.append([x, y, z])

    return np.array(atomic_numbers), np.array(coordinates)


@jit(nopython=True)
def calculate_coulomb_matrix(atomic_numbers, coordinates):
    """Calcola la matrice di Coulomb data la lista dei numeri atomici e le coordinate."""
    num_atoms = len(atomic_numbers)
    coulomb_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                # Diagonale principale
                coulomb_matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
            else:
                # Elementi fuori diagonale
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                coulomb_matrix[i, j] = atomic_numbers[i] * atomic_numbers[j] / distance

    return coulomb_matrix


def sort_by_row_norm(matrix):
    """Ordina la matrice di Coulomb in base alla norma delle righe, in ordine decrescente."""
    row_norms = np.linalg.norm(matrix, axis=1)
    sorted_indices = np.argsort(-row_norms)
    sorted_matrix = matrix[sorted_indices][:, sorted_indices]
    return sorted_matrix


def sort_by_atomic_number(atomic_numbers, matrix):
    """Ordina la matrice di Coulomb in base al numero atomico, in ordine decrescente."""
    sorted_indices = np.argsort(-atomic_numbers)
    sorted_matrix = matrix[sorted_indices][:, sorted_indices]
    return sorted_matrix


def normalize_min_max(matrix):
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))


def padd_matrix(matrix, resolution):
    padded_matrix = np.zeros((resolution, resolution))
    padded_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix

    return padded_matrix

def padd_eig(matrix, resolution):
    padded_matrix = np.zeros((resolution, 1))
    padded_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix

    return padded_matrix

def randomly_sort_matrix(matrix):
    permutation = np.random.permutation(matrix.shape[0])
    matrix_sorted = matrix[permutation, :][:, permutation]
    return matrix_sorted


def standardize_matrix(matrix):
    mean = np.mean(matrix, axis=0)
    std_dev = np.std(matrix, axis=0)
    standardized_matrix = (matrix - mean) / std_dev

    # scaler = StandardScaler()
    # standardized_matrix = scaler.fit_transform(matrix)
    return standardized_matrix


def log_normalize(matrix):
    # Applica la normalizzazione logaritmica
    return np.log1p(np.abs(matrix))


def compute_eigenvalues(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return eigenvalues


if __name__ == "__main__":
    pass
