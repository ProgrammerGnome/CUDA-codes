#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:37:11 2023

@author: kmark7
"""

import random

def generate_matrix(rows, cols):
    return [[random.randint(1, 100) for _ in range(cols)] for _ in range(rows)]

def write_matrix_to_file(matrix, filename):
    with open(filename, 'w') as file:
        rows = len(matrix)
        cols = len(matrix[0])
        file.write(f"{rows} {cols}\n")
        for row in matrix:
            file.write(" ".join(map(str, row))+ "\n")

# Generáljuk a mátrixokat
matrix_A = generate_matrix(1200, 1000)
matrix_B = generate_matrix(1000, 2300)

# Írjuk ki a mátrixokat fájlokba
write_matrix_to_file(matrix_A, "matrix_A.txt")
write_matrix_to_file(matrix_B, "matrix_B.txt")
