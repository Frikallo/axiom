#!/usr/bin/env python3
"""
Generate NumPy .npy test fixtures for Axiom IO tests.
Run this script to regenerate fixtures when needed.
"""

import numpy as np
import os

# Output directory
FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))

def save_fixture(name, arr):
    """Save array as .npy file"""
    path = os.path.join(FIXTURES_DIR, name)
    np.save(path, arr)
    print(f"  Created: {name} - shape={arr.shape}, dtype={arr.dtype}")

print("Generating NumPy fixtures for Axiom IO tests...")
print(f"Output directory: {FIXTURES_DIR}")
print()

# Float types
print("Float types:")
save_fixture("float32_2d.npy", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
save_fixture("float64_2d.npy", np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64))
save_fixture("float16_2d.npy", np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float16))

# Integer types
print("\nInteger types:")
save_fixture("int8_1d.npy", np.array([-128, -1, 0, 1, 127], dtype=np.int8))
save_fixture("int16_1d.npy", np.array([-32768, 0, 32767], dtype=np.int16))
save_fixture("int32_2d.npy", np.arange(12, dtype=np.int32).reshape(3, 4))
save_fixture("int64_2d.npy", np.arange(6, dtype=np.int64).reshape(2, 3))

# Unsigned integer types
print("\nUnsigned integer types:")
save_fixture("uint8_1d.npy", np.array([0, 128, 255], dtype=np.uint8))
save_fixture("uint16_1d.npy", np.array([0, 32768, 65535], dtype=np.uint16))
save_fixture("uint32_1d.npy", np.array([0, 1000000, 4294967295], dtype=np.uint32))
save_fixture("uint64_1d.npy", np.array([0, 1000000000000], dtype=np.uint64))

# Boolean
print("\nBoolean:")
save_fixture("bool_2d.npy", np.array([[True, False], [False, True]], dtype=np.bool_))

# Complex types
print("\nComplex types:")
save_fixture("complex64_1d.npy", np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64))
save_fixture("complex128_1d.npy", np.array([1.5+2.5j, 3.5+4.5j], dtype=np.complex128))

# Memory orders
print("\nMemory orders:")
c_order = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order='C')
f_order = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order='F')
save_fixture("c_order.npy", np.ascontiguousarray(c_order))
save_fixture("f_order.npy", np.asfortranarray(f_order))

# Various shapes
print("\nVarious shapes:")
save_fixture("scalar.npy", np.array(42.0, dtype=np.float32))
save_fixture("1d.npy", np.arange(10, dtype=np.float32))
save_fixture("3d.npy", np.arange(24, dtype=np.float32).reshape(2, 3, 4))
save_fixture("4d.npy", np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5))

# Edge cases
print("\nEdge cases:")
save_fixture("empty_1d.npy", np.array([], dtype=np.float32))
save_fixture("single_element.npy", np.array([42.0], dtype=np.float32))
save_fixture("large_1d.npy", np.arange(10000, dtype=np.float32))

# Special values
print("\nSpecial values:")
save_fixture("special_float32.npy", np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32))

print("\nDone! Fixtures generated successfully.")
print(f"Total files: {len([f for f in os.listdir(FIXTURES_DIR) if f.endswith('.npy')])}")
