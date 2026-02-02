#include <axiom/linalg.hpp>
#include <axiom/operations.hpp>

#include "backends/cpu/blas/blas_backend.hpp"
#include "backends/cpu/lapack/lapack_backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace axiom {
namespace linalg {

using backends::cpu::lapack::get_lapack_backend;
using backends::cpu::lapack::LapackBackend;

// ============================================================================
// Internal Helpers
// ============================================================================

namespace {

// Check that tensor is on CPU and make contiguous column-major copy if needed
// LAPACK expects column-major (Fortran) order
Tensor ensure_cpu_contiguous_colmajor(const Tensor &t) {
    auto cpu_t = t.device() == Device::CPU ? t : t.cpu();
    // For LAPACK we need column-major, but we store row-major
    // We'll work with transposed data where needed
    return cpu_t.is_contiguous() ? cpu_t : cpu_t.ascontiguousarray();
}

// Compute batch size from shape (all dimensions except last 2)
size_t compute_batch_size(const Shape &shape) {
    if (shape.size() < 2) {
        return 1;
    }
    size_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= shape[i];
    }
    return batch;
}

// Get batch shape (all dimensions except last 2)
Shape get_batch_shape(const Shape &shape) {
    if (shape.size() <= 2) {
        return {};
    }
    Shape batch_shape;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch_shape.push_back(shape[i]);
    }
    return batch_shape;
}

// Validate square matrix for operations like det, inv
void validate_square_matrix(const Tensor &a, const std::string &op_name) {
    if (a.ndim() < 2) {
        throw ShapeError(op_name + " requires at least 2D tensor, got " +
                         std::to_string(a.ndim()) + "D");
    }
    size_t m = a.shape()[a.ndim() - 2];
    size_t n = a.shape()[a.ndim() - 1];
    if (m != n) {
        throw ShapeError(op_name + " requires square matrix, got (" +
                         std::to_string(m) + ", " + std::to_string(n) + ")");
    }
}

// Check LAPACK return code and throw if error
void check_lapack_info(int info, const std::string &routine) {
    if (info < 0) {
        throw RuntimeError("LAPACK " + routine + ": illegal argument " +
                           std::to_string(-info));
    }
    if (info > 0) {
        throw RuntimeError("LAPACK " + routine + ": computation failed (info=" +
                           std::to_string(info) + ")");
    }
}

} // namespace

// ============================================================================
// Utility Functions
// ============================================================================

bool has_lapack() { return backends::cpu::lapack::has_lapack(); }

const char *lapack_backend_name() { return get_lapack_backend().name(); }

// ============================================================================
// Core Operations
// ============================================================================

Tensor det(const Tensor &a) {
    validate_square_matrix(a, "det");

    auto &backend = get_lapack_backend();
    auto a_work = ensure_cpu_contiguous_colmajor(a).clone();

    size_t n = a.shape()[a.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());
    Shape batch_shape = get_batch_shape(a.shape());

    // Result has batch shape
    Shape result_shape = batch_shape.empty() ? Shape({}) : batch_shape;
    Tensor result;

    if (a.dtype() == DType::Float32) {
        result = Tensor(result_shape, DType::Float32);
        float *a_data = a_work.typed_data<float>();
        float *result_data = result.typed_data<float>();
        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            float *a_batch = a_data + b * n * n;
            int lda = static_cast<int>(n);

            // LU factorization: P * A = L * U
            // For row-major, we compute A^T = L^T * U^T * P^T
            // We'll transpose the data for LAPACK (column-major)
            // Actually, since det(A) = det(A^T), we can work directly

            // Make a transposed copy for column-major LAPACK
            std::vector<float> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info = backend.sgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            if (info > 0) {
                // Singular matrix, det = 0
                result_data[b] = 0.0f;
            } else if (info < 0) {
                check_lapack_info(info, "sgetrf");
            } else {
                // det = product of diagonal * sign from permutation
                float det_val = 1.0f;
                int sign = 1;
                for (size_t i = 0; i < n; ++i) {
                    det_val *= a_col[i * n + i];
                    if (ipiv[i] != static_cast<int>(i + 1)) {
                        sign = -sign;
                    }
                }
                result_data[b] = det_val * static_cast<float>(sign);
            }
        }
    } else if (a.dtype() == DType::Float64) {
        result = Tensor(result_shape, DType::Float64);
        // Need to convert or work with double
        auto a_double = a_work.astype(DType::Float64);
        double *a_data = a_double.typed_data<double>();
        double *result_data = result.typed_data<double>();
        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            double *a_batch = a_data + b * n * n;
            int lda = static_cast<int>(n);

            std::vector<double> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info = backend.dgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            if (info > 0) {
                result_data[b] = 0.0;
            } else if (info < 0) {
                check_lapack_info(info, "dgetrf");
            } else {
                double det_val = 1.0;
                int sign = 1;
                for (size_t i = 0; i < n; ++i) {
                    det_val *= a_col[i * n + i];
                    if (ipiv[i] != static_cast<int>(i + 1)) {
                        sign = -sign;
                    }
                }
                result_data[b] = det_val * static_cast<double>(sign);
            }
        }
    } else if (a.dtype() == DType::Complex64) {
        result = Tensor(result_shape, DType::Complex64);
        using complex64_t = std::complex<float>;
        auto a_cplx = a_work;
        complex64_t *a_data = a_cplx.typed_data<complex64_t>();
        complex64_t *result_data = result.typed_data<complex64_t>();
        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            complex64_t *a_batch = a_data + b * n * n;
            int lda = static_cast<int>(n);

            std::vector<complex64_t> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info = backend.cgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            if (info > 0) {
                result_data[b] = complex64_t(0.0f, 0.0f);
            } else if (info < 0) {
                check_lapack_info(info, "cgetrf");
            } else {
                complex64_t det_val(1.0f, 0.0f);
                int sign = 1;
                for (size_t i = 0; i < n; ++i) {
                    det_val *= a_col[i * n + i];
                    if (ipiv[i] != static_cast<int>(i + 1)) {
                        sign = -sign;
                    }
                }
                result_data[b] = det_val * static_cast<float>(sign);
            }
        }
    } else if (a.dtype() == DType::Complex128) {
        result = Tensor(result_shape, DType::Complex128);
        using complex128_t = std::complex<double>;
        auto a_cplx = a_work.astype(DType::Complex128);
        complex128_t *a_data = a_cplx.typed_data<complex128_t>();
        complex128_t *result_data = result.typed_data<complex128_t>();
        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            complex128_t *a_batch = a_data + b * n * n;
            int lda = static_cast<int>(n);

            std::vector<complex128_t> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info = backend.zgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            if (info > 0) {
                result_data[b] = complex128_t(0.0, 0.0);
            } else if (info < 0) {
                check_lapack_info(info, "zgetrf");
            } else {
                complex128_t det_val(1.0, 0.0);
                int sign = 1;
                for (size_t i = 0; i < n; ++i) {
                    det_val *= a_col[i * n + i];
                    if (ipiv[i] != static_cast<int>(i + 1)) {
                        sign = -sign;
                    }
                }
                result_data[b] = det_val * static_cast<double>(sign);
            }
        }
    } else {
        // Convert to float64 for other dtypes
        return det(a.astype(DType::Float64));
    }

    return result;
}

Tensor inv(const Tensor &a) {
    validate_square_matrix(a, "inv");

    auto &backend = get_lapack_backend();

    size_t n = a.shape()[a.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());

    // Result has same shape as input
    Tensor result = Tensor(a.shape(), a.dtype());

    if (a.dtype() == DType::Float32) {
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        float *a_data = a_work.typed_data<float>();
        float *result_data = result.typed_data<float>();
        std::vector<int> ipiv(n);
        std::vector<float> work(n * 64); // Oversize workspace

        for (size_t b = 0; b < batch_size; ++b) {
            float *a_batch = a_data + b * n * n;
            float *result_batch = result_data + b * n * n;
            int lda = static_cast<int>(n);

            // Transpose to column-major for LAPACK
            std::vector<float> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            // LU factorization
            int info = backend.sgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            check_lapack_info(info, "sgetrf");

            // Compute inverse
            info = backend.sgetri(static_cast<int>(n), a_col.data(), lda,
                                  ipiv.data(), work.data(),
                                  static_cast<int>(work.size()));
            check_lapack_info(info, "sgetri");

            // Transpose back to row-major
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    result_batch[i * n + j] = a_col[j * n + i];
                }
            }
        }
    } else if (a.dtype() == DType::Float64) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        double *a_data = a_work.typed_data<double>();
        result = Tensor(a.shape(), DType::Float64);
        double *result_data = result.typed_data<double>();
        std::vector<int> ipiv(n);
        std::vector<double> work(n * 64);

        for (size_t b = 0; b < batch_size; ++b) {
            double *a_batch = a_data + b * n * n;
            double *result_batch = result_data + b * n * n;
            int lda = static_cast<int>(n);

            std::vector<double> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info = backend.dgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            check_lapack_info(info, "dgetrf");

            info = backend.dgetri(static_cast<int>(n), a_col.data(), lda,
                                  ipiv.data(), work.data(),
                                  static_cast<int>(work.size()));
            check_lapack_info(info, "dgetri");

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    result_batch[i * n + j] = a_col[j * n + i];
                }
            }
        }
    } else if (a.dtype() == DType::Complex64) {
        using complex64_t = std::complex<float>;
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        complex64_t *a_data = a_work.typed_data<complex64_t>();
        complex64_t *result_data = result.typed_data<complex64_t>();
        std::vector<int> ipiv(n);
        std::vector<complex64_t> work(n * 64);

        for (size_t b = 0; b < batch_size; ++b) {
            complex64_t *a_batch = a_data + b * n * n;
            complex64_t *result_batch = result_data + b * n * n;
            int lda = static_cast<int>(n);

            std::vector<complex64_t> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info = backend.cgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            check_lapack_info(info, "cgetrf");

            info = backend.cgetri(static_cast<int>(n), a_col.data(), lda,
                                  ipiv.data(), work.data(),
                                  static_cast<int>(work.size()));
            check_lapack_info(info, "cgetri");

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    result_batch[i * n + j] = a_col[j * n + i];
                }
            }
        }
    } else if (a.dtype() == DType::Complex128) {
        using complex128_t = std::complex<double>;
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Complex128)).clone();
        complex128_t *a_data = a_work.typed_data<complex128_t>();
        result = Tensor(a.shape(), DType::Complex128);
        complex128_t *result_data = result.typed_data<complex128_t>();
        std::vector<int> ipiv(n);
        std::vector<complex128_t> work(n * 64);

        for (size_t b = 0; b < batch_size; ++b) {
            complex128_t *a_batch = a_data + b * n * n;
            complex128_t *result_batch = result_data + b * n * n;
            int lda = static_cast<int>(n);

            std::vector<complex128_t> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info = backend.zgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            check_lapack_info(info, "zgetrf");

            info = backend.zgetri(static_cast<int>(n), a_col.data(), lda,
                                  ipiv.data(), work.data(),
                                  static_cast<int>(work.size()));
            check_lapack_info(info, "zgetri");

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    result_batch[i * n + j] = a_col[j * n + i];
                }
            }
        }
    } else {
        return inv(a.astype(DType::Float64));
    }

    return result;
}

Tensor solve(const Tensor &a, const Tensor &b) {
    validate_square_matrix(a, "solve");

    size_t n = a.shape()[a.ndim() - 1];
    size_t nrhs = (b.ndim() == a.ndim() - 1) ? 1 : b.shape()[b.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());

    auto &backend = get_lapack_backend();

    // Result shape matches b
    Tensor result = Tensor(b.shape(), b.dtype());

    if (a.dtype() == DType::Float32 || b.dtype() == DType::Float32) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float32)).clone();
        auto b_work =
            ensure_cpu_contiguous_colmajor(b.astype(DType::Float32)).clone();
        result = Tensor(b.shape(), DType::Float32);

        float *a_data = a_work.typed_data<float>();
        float *b_data = b_work.typed_data<float>();
        float *result_data = result.typed_data<float>();
        std::vector<int> ipiv(n);

        bool b_is_vector = (b.ndim() == a.ndim() - 1);
        size_t b_stride = b_is_vector ? n : n * nrhs;

        for (size_t batch = 0; batch < batch_size; ++batch) {
            float *a_batch = a_data + batch * n * n;
            float *b_batch = b_data + batch * b_stride;
            float *result_batch = result_data + batch * b_stride;
            int lda = static_cast<int>(n);
            int ldb = static_cast<int>(n);

            // Transpose A and B to column-major
            std::vector<float> a_col(n * n);
            std::vector<float> b_col(n * nrhs);

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            if (b_is_vector) {
                std::copy(b_batch, b_batch + n, b_col.data());
            } else {
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < nrhs; ++j) {
                        b_col[j * n + i] = b_batch[i * nrhs + j];
                    }
                }
            }

            int info = backend.sgesv(static_cast<int>(n),
                                     static_cast<int>(nrhs), a_col.data(), lda,
                                     ipiv.data(), b_col.data(), ldb);
            check_lapack_info(info, "sgesv");

            // Transpose result back
            if (b_is_vector) {
                std::copy(b_col.data(), b_col.data() + n, result_batch);
            } else {
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < nrhs; ++j) {
                        result_batch[i * nrhs + j] = b_col[j * n + i];
                    }
                }
            }
        }
    } else if (a.dtype() == DType::Float64 || b.dtype() == DType::Float64) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        auto b_work =
            ensure_cpu_contiguous_colmajor(b.astype(DType::Float64)).clone();
        result = Tensor(b.shape(), DType::Float64);

        double *a_data = a_work.typed_data<double>();
        double *b_data = b_work.typed_data<double>();
        double *result_data = result.typed_data<double>();
        std::vector<int> ipiv(n);

        bool b_is_vector = (b.ndim() == a.ndim() - 1);
        size_t b_stride = b_is_vector ? n : n * nrhs;

        for (size_t batch = 0; batch < batch_size; ++batch) {
            double *a_batch = a_data + batch * n * n;
            double *b_batch = b_data + batch * b_stride;
            double *result_batch = result_data + batch * b_stride;
            int lda = static_cast<int>(n);
            int ldb = static_cast<int>(n);

            std::vector<double> a_col(n * n);
            std::vector<double> b_col(n * nrhs);

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            if (b_is_vector) {
                std::copy(b_batch, b_batch + n, b_col.data());
            } else {
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < nrhs; ++j) {
                        b_col[j * n + i] = b_batch[i * nrhs + j];
                    }
                }
            }

            int info = backend.dgesv(static_cast<int>(n),
                                     static_cast<int>(nrhs), a_col.data(), lda,
                                     ipiv.data(), b_col.data(), ldb);
            check_lapack_info(info, "dgesv");

            if (b_is_vector) {
                std::copy(b_col.data(), b_col.data() + n, result_batch);
            } else {
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < nrhs; ++j) {
                        result_batch[i * nrhs + j] = b_col[j * n + i];
                    }
                }
            }
        }
    } else {
        return solve(a.astype(DType::Float64), b.astype(DType::Float64));
    }

    return result;
}

// ============================================================================
// Matrix Decompositions
// ============================================================================

SVDResult svd(const Tensor &a, bool full_matrices) {
    if (a.ndim() < 2) {
        throw ShapeError("svd requires at least 2D tensor");
    }

    auto &backend = get_lapack_backend();

    size_t m = a.shape()[a.ndim() - 2];
    size_t n = a.shape()[a.ndim() - 1];
    size_t k = std::min(m, n);
    size_t batch_size = compute_batch_size(a.shape());
    Shape batch_shape = get_batch_shape(a.shape());

    // Output shapes
    size_t u_cols = full_matrices ? m : k;
    size_t vt_rows = full_matrices ? n : k;

    Shape u_shape = batch_shape;
    u_shape.push_back(m);
    u_shape.push_back(u_cols);

    Shape s_shape = batch_shape;
    s_shape.push_back(k);

    Shape vt_shape = batch_shape;
    vt_shape.push_back(vt_rows);
    vt_shape.push_back(n);

    SVDResult result;
    char jobz = full_matrices ? 'A' : 'S';

    if (a.dtype() == DType::Float32) {
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        float *a_data = a_work.typed_data<float>();

        result.U = Tensor(u_shape, DType::Float32);
        result.S = Tensor(s_shape, DType::Float32);
        result.Vh = Tensor(vt_shape, DType::Float32);

        float *u_data = result.U.typed_data<float>();
        float *s_data = result.S.typed_data<float>();
        float *vt_data = result.Vh.typed_data<float>();

        int lda = static_cast<int>(m);
        int ldu = static_cast<int>(m);
        int ldvt = static_cast<int>(vt_rows);

        // Workspace query: call with lwork=-1 to get optimal size
        float work_query;
        std::vector<int> iwork(8 * k);
        std::vector<float> dummy_s(k);
        std::vector<float> dummy_u(m * u_cols);
        std::vector<float> dummy_vt(vt_rows * n);
        std::vector<float> dummy_a(m * n);

        int info = backend.sgesdd(
            jobz, static_cast<int>(m), static_cast<int>(n), dummy_a.data(), lda,
            dummy_s.data(), dummy_u.data(), ldu, dummy_vt.data(), ldvt,
            &work_query, -1, iwork.data());
        check_lapack_info(info, "sgesdd workspace query");

        int lwork = static_cast<int>(work_query) + 1;
        std::vector<float> work(lwork);

        for (size_t b = 0; b < batch_size; ++b) {
            float *a_batch = a_data + b * m * n;

            // Transpose to column-major
            std::vector<float> a_col(m * n);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * m + i] = a_batch[i * n + j];
                }
            }

            std::vector<float> u_col(m * u_cols);
            std::vector<float> vt_col(vt_rows * n);

            info = backend.sgesdd(
                jobz, static_cast<int>(m), static_cast<int>(n), a_col.data(),
                lda, s_data + b * k, u_col.data(), ldu, vt_col.data(), ldvt,
                work.data(), lwork, iwork.data());
            check_lapack_info(info, "sgesdd");

            // Transpose U and Vt back to row-major
            float *u_batch = u_data + b * m * u_cols;
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < u_cols; ++j) {
                    u_batch[i * u_cols + j] = u_col[j * m + i];
                }
            }

            float *vt_batch = vt_data + b * vt_rows * n;
            for (size_t i = 0; i < vt_rows; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    vt_batch[i * n + j] = vt_col[j * vt_rows + i];
                }
            }
        }
    } else if (a.dtype() == DType::Float64) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        double *a_data = a_work.typed_data<double>();

        result.U = Tensor(u_shape, DType::Float64);
        result.S = Tensor(s_shape, DType::Float64);
        result.Vh = Tensor(vt_shape, DType::Float64);

        double *u_data = result.U.typed_data<double>();
        double *s_data = result.S.typed_data<double>();
        double *vt_data = result.Vh.typed_data<double>();

        int lda = static_cast<int>(m);
        int ldu = static_cast<int>(m);
        int ldvt = static_cast<int>(vt_rows);

        // Workspace query
        double work_query;
        std::vector<int> iwork(8 * k);
        std::vector<double> dummy_s(k);
        std::vector<double> dummy_u(m * u_cols);
        std::vector<double> dummy_vt(vt_rows * n);
        std::vector<double> dummy_a(m * n);

        int info = backend.dgesdd(
            jobz, static_cast<int>(m), static_cast<int>(n), dummy_a.data(), lda,
            dummy_s.data(), dummy_u.data(), ldu, dummy_vt.data(), ldvt,
            &work_query, -1, iwork.data());
        check_lapack_info(info, "dgesdd workspace query");

        int lwork = static_cast<int>(work_query) + 1;
        std::vector<double> work(lwork);

        for (size_t b = 0; b < batch_size; ++b) {
            double *a_batch = a_data + b * m * n;

            std::vector<double> a_col(m * n);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * m + i] = a_batch[i * n + j];
                }
            }

            std::vector<double> u_col(m * u_cols);
            std::vector<double> vt_col(vt_rows * n);

            info = backend.dgesdd(
                jobz, static_cast<int>(m), static_cast<int>(n), a_col.data(),
                lda, s_data + b * k, u_col.data(), ldu, vt_col.data(), ldvt,
                work.data(), lwork, iwork.data());
            check_lapack_info(info, "dgesdd");

            double *u_batch = u_data + b * m * u_cols;
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < u_cols; ++j) {
                    u_batch[i * u_cols + j] = u_col[j * m + i];
                }
            }

            double *vt_batch = vt_data + b * vt_rows * n;
            for (size_t i = 0; i < vt_rows; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    vt_batch[i * n + j] = vt_col[j * vt_rows + i];
                }
            }
        }
    } else if (a.dtype() == DType::Complex64) {
        using complex64_t = std::complex<float>;
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        complex64_t *a_data = a_work.typed_data<complex64_t>();

        result.U = Tensor(u_shape, DType::Complex64);
        result.S = Tensor(s_shape, DType::Float32); // Singular values are real
        result.Vh = Tensor(vt_shape, DType::Complex64);

        complex64_t *u_data = result.U.typed_data<complex64_t>();
        float *s_data = result.S.typed_data<float>();
        complex64_t *vt_data = result.Vh.typed_data<complex64_t>();

        size_t lwork =
            3 * std::max(m, n) + std::max(m, n) * std::max(m, n) + 1000;
        std::vector<complex64_t> work(lwork);
        // rwork size for complex gesdd
        size_t lrwork = std::max(5 * k * k + 5 * k,
                                 2 * std::max(m, n) * k + 2 * k * k + k) +
                        1000;
        std::vector<float> rwork(lrwork);
        std::vector<int> iwork(8 * k);

        for (size_t b = 0; b < batch_size; ++b) {
            complex64_t *a_batch = a_data + b * m * n;

            // Transpose to column-major
            std::vector<complex64_t> a_col(m * n);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * m + i] = a_batch[i * n + j];
                }
            }

            std::vector<complex64_t> u_col(m * u_cols);
            std::vector<complex64_t> vt_col(vt_rows * n);

            int lda = static_cast<int>(m);
            int ldu = static_cast<int>(m);
            int ldvt = static_cast<int>(vt_rows);

            int info = backend.cgesdd(
                jobz, static_cast<int>(m), static_cast<int>(n), a_col.data(),
                lda, s_data + b * k, u_col.data(), ldu, vt_col.data(), ldvt,
                work.data(), static_cast<int>(work.size()), rwork.data(),
                iwork.data());
            check_lapack_info(info, "cgesdd");

            // Transpose U and Vt back to row-major
            complex64_t *u_batch = u_data + b * m * u_cols;
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < u_cols; ++j) {
                    u_batch[i * u_cols + j] = u_col[j * m + i];
                }
            }

            complex64_t *vt_batch = vt_data + b * vt_rows * n;
            for (size_t i = 0; i < vt_rows; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    vt_batch[i * n + j] = vt_col[j * vt_rows + i];
                }
            }
        }
    } else if (a.dtype() == DType::Complex128) {
        using complex128_t = std::complex<double>;
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        complex128_t *a_data = a_work.typed_data<complex128_t>();

        result.U = Tensor(u_shape, DType::Complex128);
        result.S = Tensor(s_shape, DType::Float64); // Singular values are real
        result.Vh = Tensor(vt_shape, DType::Complex128);

        complex128_t *u_data = result.U.typed_data<complex128_t>();
        double *s_data = result.S.typed_data<double>();
        complex128_t *vt_data = result.Vh.typed_data<complex128_t>();

        size_t lwork =
            3 * std::max(m, n) + std::max(m, n) * std::max(m, n) + 1000;
        std::vector<complex128_t> work(lwork);
        size_t lrwork = std::max(5 * k * k + 5 * k,
                                 2 * std::max(m, n) * k + 2 * k * k + k) +
                        1000;
        std::vector<double> rwork(lrwork);
        std::vector<int> iwork(8 * k);

        for (size_t b = 0; b < batch_size; ++b) {
            complex128_t *a_batch = a_data + b * m * n;

            std::vector<complex128_t> a_col(m * n);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * m + i] = a_batch[i * n + j];
                }
            }

            std::vector<complex128_t> u_col(m * u_cols);
            std::vector<complex128_t> vt_col(vt_rows * n);

            int lda = static_cast<int>(m);
            int ldu = static_cast<int>(m);
            int ldvt = static_cast<int>(vt_rows);

            int info = backend.zgesdd(
                jobz, static_cast<int>(m), static_cast<int>(n), a_col.data(),
                lda, s_data + b * k, u_col.data(), ldu, vt_col.data(), ldvt,
                work.data(), static_cast<int>(work.size()), rwork.data(),
                iwork.data());
            check_lapack_info(info, "zgesdd");

            complex128_t *u_batch = u_data + b * m * u_cols;
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < u_cols; ++j) {
                    u_batch[i * u_cols + j] = u_col[j * m + i];
                }
            }

            complex128_t *vt_batch = vt_data + b * vt_rows * n;
            for (size_t i = 0; i < vt_rows; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    vt_batch[i * n + j] = vt_col[j * vt_rows + i];
                }
            }
        }
    } else {
        return svd(a.astype(DType::Float64), full_matrices);
    }

    return result;
}

QRResult qr(const Tensor &a) {
    if (a.ndim() < 2) {
        throw ShapeError("qr requires at least 2D tensor");
    }

    auto &backend = get_lapack_backend();

    size_t m = a.shape()[a.ndim() - 2];
    size_t n = a.shape()[a.ndim() - 1];
    size_t k = std::min(m, n);
    size_t batch_size = compute_batch_size(a.shape());
    Shape batch_shape = get_batch_shape(a.shape());

    Shape q_shape = batch_shape;
    q_shape.push_back(m);
    q_shape.push_back(k);

    Shape r_shape = batch_shape;
    r_shape.push_back(k);
    r_shape.push_back(n);

    QRResult result;

    if (a.dtype() == DType::Float32) {
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        float *a_data = a_work.typed_data<float>();

        result.Q = Tensor(q_shape, DType::Float32);
        result.R = Tensor(r_shape, DType::Float32);
        result.R.fill(0.0f);

        float *q_data = result.Q.typed_data<float>();
        float *r_data = result.R.typed_data<float>();

        std::vector<float> tau(k);
        std::vector<float> work(n * 64);

        for (size_t b = 0; b < batch_size; ++b) {
            float *a_batch = a_data + b * m * n;

            // Transpose to column-major
            std::vector<float> a_col(m * n);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * m + i] = a_batch[i * n + j];
                }
            }

            int lda = static_cast<int>(m);

            // QR factorization
            int info = backend.sgeqrf(
                static_cast<int>(m), static_cast<int>(n), a_col.data(), lda,
                tau.data(), work.data(), static_cast<int>(work.size()));
            check_lapack_info(info, "sgeqrf");

            // Extract R (upper triangular)
            float *r_batch = r_data + b * k * n;
            for (size_t i = 0; i < k; ++i) {
                for (size_t j = i; j < n; ++j) {
                    r_batch[i * n + j] = a_col[j * m + i];
                }
            }

            // Generate Q
            info = backend.sorgqr(static_cast<int>(m), static_cast<int>(k),
                                  static_cast<int>(k), a_col.data(), lda,
                                  tau.data(), work.data(),
                                  static_cast<int>(work.size()));
            check_lapack_info(info, "sorgqr");

            // Transpose Q back
            float *q_batch = q_data + b * m * k;
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    q_batch[i * k + j] = a_col[j * m + i];
                }
            }
        }
    } else if (a.dtype() == DType::Float64) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        double *a_data = a_work.typed_data<double>();

        result.Q = Tensor(q_shape, DType::Float64);
        result.R = Tensor(r_shape, DType::Float64);
        result.R.fill(0.0);

        double *q_data = result.Q.typed_data<double>();
        double *r_data = result.R.typed_data<double>();

        std::vector<double> tau(k);
        std::vector<double> work(n * 64);

        for (size_t b = 0; b < batch_size; ++b) {
            double *a_batch = a_data + b * m * n;

            std::vector<double> a_col(m * n);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * m + i] = a_batch[i * n + j];
                }
            }

            int lda = static_cast<int>(m);

            int info = backend.dgeqrf(
                static_cast<int>(m), static_cast<int>(n), a_col.data(), lda,
                tau.data(), work.data(), static_cast<int>(work.size()));
            check_lapack_info(info, "dgeqrf");

            double *r_batch = r_data + b * k * n;
            for (size_t i = 0; i < k; ++i) {
                for (size_t j = i; j < n; ++j) {
                    r_batch[i * n + j] = a_col[j * m + i];
                }
            }

            info = backend.dorgqr(static_cast<int>(m), static_cast<int>(k),
                                  static_cast<int>(k), a_col.data(), lda,
                                  tau.data(), work.data(),
                                  static_cast<int>(work.size()));
            check_lapack_info(info, "dorgqr");

            double *q_batch = q_data + b * m * k;
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    q_batch[i * k + j] = a_col[j * m + i];
                }
            }
        }
    } else {
        return qr(a.astype(DType::Float64));
    }

    return result;
}

Tensor cholesky(const Tensor &a, bool upper) {
    validate_square_matrix(a, "cholesky");

    auto &backend = get_lapack_backend();

    size_t n = a.shape()[a.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());

    Tensor result = Tensor(a.shape(), a.dtype());
    char uplo = upper ? 'U' : 'L';

    if (a.dtype() == DType::Float32) {
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        float *a_data = a_work.typed_data<float>();
        float *result_data = result.typed_data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            float *a_batch = a_data + b * n * n;
            float *result_batch = result_data + b * n * n;

            // Transpose to column-major
            std::vector<float> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int lda = static_cast<int>(n);
            int info =
                backend.spotrf(uplo, static_cast<int>(n), a_col.data(), lda);
            check_lapack_info(info, "spotrf");

            // Zero out the other triangle and transpose back
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    if (upper) {
                        result_batch[i * n + j] =
                            (j >= i) ? a_col[j * n + i] : 0.0f;
                    } else {
                        result_batch[i * n + j] =
                            (j <= i) ? a_col[j * n + i] : 0.0f;
                    }
                }
            }
        }
    } else if (a.dtype() == DType::Float64) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        double *a_data = a_work.typed_data<double>();
        result = Tensor(a.shape(), DType::Float64);
        double *result_data = result.typed_data<double>();

        for (size_t b = 0; b < batch_size; ++b) {
            double *a_batch = a_data + b * n * n;
            double *result_batch = result_data + b * n * n;

            std::vector<double> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int lda = static_cast<int>(n);
            int info =
                backend.dpotrf(uplo, static_cast<int>(n), a_col.data(), lda);
            check_lapack_info(info, "dpotrf");

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    if (upper) {
                        result_batch[i * n + j] =
                            (j >= i) ? a_col[j * n + i] : 0.0;
                    } else {
                        result_batch[i * n + j] =
                            (j <= i) ? a_col[j * n + i] : 0.0;
                    }
                }
            }
        }
    } else {
        return cholesky(a.astype(DType::Float64), upper);
    }

    return result;
}

LUResult lu(const Tensor &a) {
    validate_square_matrix(a, "lu");

    auto &backend = get_lapack_backend();

    size_t n = a.shape()[a.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());
    Shape batch_shape = get_batch_shape(a.shape());

    Shape piv_shape = batch_shape;
    piv_shape.push_back(n);

    LUResult result;
    result.L = Tensor(a.shape(), a.dtype());
    result.U = Tensor(a.shape(), a.dtype());
    result.P = Tensor(a.shape(), a.dtype());
    result.piv = Tensor(piv_shape, DType::Int32);

    if (a.dtype() == DType::Float32) {
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        float *a_data = a_work.typed_data<float>();
        float *l_data = result.L.typed_data<float>();
        float *u_data = result.U.typed_data<float>();
        float *p_data = result.P.typed_data<float>();
        int32_t *piv_data = result.piv.typed_data<int32_t>();

        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            float *a_batch = a_data + b * n * n;
            float *l_batch = l_data + b * n * n;
            float *u_batch = u_data + b * n * n;
            float *p_batch = p_data + b * n * n;
            int32_t *piv_batch = piv_data + b * n;

            std::vector<float> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int lda = static_cast<int>(n);
            int info = backend.sgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            check_lapack_info(info, "sgetrf");

            // Extract L and U from column-major result
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float val = a_col[j * n + i];
                    if (i > j) {
                        l_batch[i * n + j] = val;
                        u_batch[i * n + j] = 0.0f;
                    } else if (i == j) {
                        l_batch[i * n + j] = 1.0f;
                        u_batch[i * n + j] = val;
                    } else {
                        l_batch[i * n + j] = 0.0f;
                        u_batch[i * n + j] = val;
                    }
                }
                piv_batch[i] = ipiv[i];
            }

            // Build permutation matrix from pivot indices
            std::fill(p_batch, p_batch + n * n, 0.0f);
            std::vector<size_t> perm(n);
            for (size_t i = 0; i < n; ++i) {
                perm[i] = i;
            }
            for (size_t i = 0; i < n; ++i) {
                size_t swap_idx = static_cast<size_t>(ipiv[i] - 1);
                std::swap(perm[i], perm[swap_idx]);
            }
            for (size_t i = 0; i < n; ++i) {
                p_batch[i * n + perm[i]] = 1.0f;
            }
        }
    } else if (a.dtype() == DType::Float64) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        double *a_data = a_work.typed_data<double>();
        result.L = Tensor(a.shape(), DType::Float64);
        result.U = Tensor(a.shape(), DType::Float64);
        result.P = Tensor(a.shape(), DType::Float64);
        double *l_data = result.L.typed_data<double>();
        double *u_data = result.U.typed_data<double>();
        double *p_data = result.P.typed_data<double>();
        int32_t *piv_data = result.piv.typed_data<int32_t>();

        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            double *a_batch = a_data + b * n * n;
            double *l_batch = l_data + b * n * n;
            double *u_batch = u_data + b * n * n;
            double *p_batch = p_data + b * n * n;
            int32_t *piv_batch = piv_data + b * n;

            std::vector<double> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int lda = static_cast<int>(n);
            int info = backend.dgetrf(static_cast<int>(n), static_cast<int>(n),
                                      a_col.data(), lda, ipiv.data());
            check_lapack_info(info, "dgetrf");

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double val = a_col[j * n + i];
                    if (i > j) {
                        l_batch[i * n + j] = val;
                        u_batch[i * n + j] = 0.0;
                    } else if (i == j) {
                        l_batch[i * n + j] = 1.0;
                        u_batch[i * n + j] = val;
                    } else {
                        l_batch[i * n + j] = 0.0;
                        u_batch[i * n + j] = val;
                    }
                }
                piv_batch[i] = ipiv[i];
            }

            std::fill(p_batch, p_batch + n * n, 0.0);
            std::vector<size_t> perm(n);
            for (size_t i = 0; i < n; ++i) {
                perm[i] = i;
            }
            for (size_t i = 0; i < n; ++i) {
                size_t swap_idx = static_cast<size_t>(ipiv[i] - 1);
                std::swap(perm[i], perm[swap_idx]);
            }
            for (size_t i = 0; i < n; ++i) {
                p_batch[i * n + perm[i]] = 1.0;
            }
        }
    } else {
        return lu(a.astype(DType::Float64));
    }

    return result;
}

// ============================================================================
// Eigendecomposition & Advanced Operations
// ============================================================================

EigResult eig(const Tensor &a) {
    validate_square_matrix(a, "eig");

    auto &backend = get_lapack_backend();

    size_t n = a.shape()[a.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());
    Shape batch_shape = get_batch_shape(a.shape());

    Shape eigenval_shape = batch_shape;
    eigenval_shape.push_back(n);

    EigResult result;

    if (a.dtype() == DType::Float32 || a.dtype() == DType::Float64) {
        // Real input -> complex eigenvalues possible
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        double *a_data = a_work.typed_data<double>();

        result.eigenvalues = Tensor(eigenval_shape, DType::Complex128);
        result.eigenvectors = Tensor(a.shape(), DType::Complex128);

        using complex128_t = std::complex<double>;
        complex128_t *eigenval_data =
            result.eigenvalues.typed_data<complex128_t>();
        complex128_t *eigenvec_data =
            result.eigenvectors.typed_data<complex128_t>();

        std::vector<double> wr(n), wi(n);
        std::vector<double> vr(n * n);
        std::vector<double> work(4 * n);

        for (size_t b = 0; b < batch_size; ++b) {
            double *a_batch = a_data + b * n * n;
            complex128_t *eigenval_batch = eigenval_data + b * n;
            complex128_t *eigenvec_batch = eigenvec_data + b * n * n;

            std::vector<double> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int lda = static_cast<int>(n);
            int ldvr = static_cast<int>(n);

            int info =
                backend.dgeev('N', 'V', static_cast<int>(n), a_col.data(), lda,
                              wr.data(), wi.data(), nullptr, 1, vr.data(), ldvr,
                              work.data(), static_cast<int>(work.size()));
            check_lapack_info(info, "dgeev");

            // Convert real/imag parts to complex
            for (size_t i = 0; i < n; ++i) {
                eigenval_batch[i] = complex128_t(wr[i], wi[i]);
            }

            // Convert eigenvectors to complex (LAPACK stores complex conjugate
            // pairs)
            for (size_t j = 0; j < n;) {
                if (wi[j] == 0.0) {
                    // Real eigenvalue
                    for (size_t i = 0; i < n; ++i) {
                        eigenvec_batch[i * n + j] =
                            complex128_t(vr[j * n + i], 0.0);
                    }
                    ++j;
                } else {
                    // Complex conjugate pair
                    for (size_t i = 0; i < n; ++i) {
                        eigenvec_batch[i * n + j] =
                            complex128_t(vr[j * n + i], vr[(j + 1) * n + i]);
                        eigenvec_batch[i * n + j + 1] =
                            complex128_t(vr[j * n + i], -vr[(j + 1) * n + i]);
                    }
                    j += 2;
                }
            }
        }
    } else {
        throw TypeError::unsupported_dtype(dtype_name(a.dtype()), "eig");
    }

    return result;
}

EigResult eigh(const Tensor &a) {
    validate_square_matrix(a, "eigh");

    auto &backend = get_lapack_backend();

    size_t n = a.shape()[a.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());
    Shape batch_shape = get_batch_shape(a.shape());

    Shape eigenval_shape = batch_shape;
    eigenval_shape.push_back(n);

    EigResult result;

    if (a.dtype() == DType::Float32) {
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        float *a_data = a_work.typed_data<float>();

        result.eigenvalues = Tensor(eigenval_shape, DType::Float32);
        result.eigenvectors = Tensor(a.shape(), DType::Float32);

        float *eigenval_data = result.eigenvalues.typed_data<float>();
        float *eigenvec_data = result.eigenvectors.typed_data<float>();

        std::vector<float> work(3 * n);

        for (size_t b = 0; b < batch_size; ++b) {
            float *a_batch = a_data + b * n * n;
            float *eigenval_batch = eigenval_data + b * n;
            float *eigenvec_batch = eigenvec_data + b * n * n;

            std::vector<float> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int lda = static_cast<int>(n);
            int info = backend.ssyev(
                'V', 'U', static_cast<int>(n), a_col.data(), lda,
                eigenval_batch, work.data(), static_cast<int>(work.size()));
            check_lapack_info(info, "ssyev");

            // Transpose eigenvectors back
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    eigenvec_batch[i * n + j] = a_col[j * n + i];
                }
            }
        }
    } else if (a.dtype() == DType::Float64) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        double *a_data = a_work.typed_data<double>();

        result.eigenvalues = Tensor(eigenval_shape, DType::Float64);
        result.eigenvectors = Tensor(a.shape(), DType::Float64);

        double *eigenval_data = result.eigenvalues.typed_data<double>();
        double *eigenvec_data = result.eigenvectors.typed_data<double>();

        std::vector<double> work(3 * n);

        for (size_t b = 0; b < batch_size; ++b) {
            double *a_batch = a_data + b * n * n;
            double *eigenval_batch = eigenval_data + b * n;
            double *eigenvec_batch = eigenvec_data + b * n * n;

            std::vector<double> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int lda = static_cast<int>(n);
            int info = backend.dsyev(
                'V', 'U', static_cast<int>(n), a_col.data(), lda,
                eigenval_batch, work.data(), static_cast<int>(work.size()));
            check_lapack_info(info, "dsyev");

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    eigenvec_batch[i * n + j] = a_col[j * n + i];
                }
            }
        }
    } else {
        return eigh(a.astype(DType::Float64));
    }

    return result;
}

Tensor lstsq(const Tensor &a, const Tensor &b, double rcond) {
    if (a.ndim() < 2) {
        throw ShapeError("lstsq requires at least 2D tensor for a");
    }

    auto &backend = get_lapack_backend();

    size_t m = a.shape()[a.ndim() - 2];
    size_t n = a.shape()[a.ndim() - 1];
    size_t nrhs = (b.ndim() == a.ndim() - 1) ? 1 : b.shape()[b.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());

    if (rcond < 0) {
        rcond = std::max(m, n) * std::numeric_limits<double>::epsilon();
    }

    // Result shape: (..., n, nrhs) or (..., n) for vector b
    Shape result_shape = get_batch_shape(a.shape());
    result_shape.push_back(n);
    if (b.ndim() == a.ndim()) {
        result_shape.push_back(nrhs);
    }

    Tensor result;

    if (a.dtype() == DType::Float64 || b.dtype() == DType::Float64) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        auto b_work =
            ensure_cpu_contiguous_colmajor(b.astype(DType::Float64)).clone();
        result = Tensor(result_shape, DType::Float64);

        double *a_data = a_work.typed_data<double>();
        double *b_data = b_work.typed_data<double>();
        double *result_data = result.typed_data<double>();

        bool b_is_vector = (b.ndim() == a.ndim() - 1);
        size_t b_stride = b_is_vector ? m : m * nrhs;
        size_t result_stride = b_is_vector ? n : n * nrhs;

        std::vector<double> s(std::min(m, n));
        // Larger workspace - DGELSD requires substantial work space
        size_t minmn = std::min(m, n);
        size_t maxmn = std::max(m, n);
        size_t nlvl =
            std::max(static_cast<size_t>(1),
                     static_cast<size_t>(std::log2(minmn / 26.0)) + 1);
        size_t smlsiz = 25;
        size_t liwork =
            std::max(static_cast<size_t>(1), 3 * minmn * nlvl + 11 * minmn);
        size_t lwork = 12 * minmn + 2 * minmn * smlsiz + 8 * minmn * nlvl +
                       minmn * nrhs + (smlsiz + 1) * (smlsiz + 1) + 1000;
        std::vector<double> work(std::max(lwork, maxmn * 100));
        std::vector<int> iwork(std::max(liwork, 8 * minmn + 100));

        for (size_t batch = 0; batch < batch_size; ++batch) {
            double *a_batch = a_data + batch * m * n;
            double *b_batch = b_data + batch * b_stride;
            double *result_batch = result_data + batch * result_stride;

            // Transpose to column-major
            std::vector<double> a_col(m * n);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * m + i] = a_batch[i * n + j];
                }
            }

            // b_col has size max(m,n) * nrhs for the solution
            size_t ldb = std::max(m, n);
            std::vector<double> b_col(ldb * nrhs, 0.0);
            if (b_is_vector) {
                std::copy(b_batch, b_batch + m, b_col.data());
            } else {
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < nrhs; ++j) {
                        b_col[j * ldb + i] = b_batch[i * nrhs + j];
                    }
                }
            }

            int rank = 0;
            int info = backend.dgelsd(
                static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(nrhs), a_col.data(), static_cast<int>(m),
                b_col.data(), static_cast<int>(ldb), s.data(), rcond, &rank,
                work.data(), static_cast<int>(work.size()), iwork.data());
            check_lapack_info(info, "dgelsd");

            // Extract solution (first n rows of b_col)
            if (b_is_vector) {
                std::copy(b_col.data(), b_col.data() + n, result_batch);
            } else {
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < nrhs; ++j) {
                        result_batch[i * nrhs + j] = b_col[j * ldb + i];
                    }
                }
            }
        }
    } else {
        return lstsq(a.astype(DType::Float64), b.astype(DType::Float64), rcond);
    }

    return result;
}

Tensor pinv(const Tensor &a, double rcond) {
    if (a.ndim() < 2) {
        throw ShapeError("pinv requires at least 2D tensor");
    }

    // Pseudoinverse via SVD: A^+ = V @ diag(1/s) @ U^H
    auto [U, S, Vh] = svd(a, false);

    size_t m = a.shape()[a.ndim() - 2];
    size_t n = a.shape()[a.ndim() - 1];
    size_t k = S.shape()[S.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());
    Shape batch_shape = get_batch_shape(a.shape());

    // Threshold small singular values (use max across all batches)
    double threshold =
        rcond * S.max().item<double>(); // max singular value * rcond

    // Create reciprocal of singular values, zeroing small ones
    Tensor S_inv = Tensor::zeros(S.shape(), S.dtype());

    if (S.dtype() == DType::Float32) {
        float *s_data = S.typed_data<float>();
        float *s_inv_data = S_inv.typed_data<float>();
        float thresh_f = static_cast<float>(threshold);
        for (size_t i = 0; i < S.size(); ++i) {
            s_inv_data[i] = (s_data[i] > thresh_f) ? (1.0f / s_data[i]) : 0.0f;
        }
    } else {
        double *s_data = S.typed_data<double>();
        double *s_inv_data = S_inv.typed_data<double>();
        for (size_t i = 0; i < S.size(); ++i) {
            s_inv_data[i] = (s_data[i] > threshold) ? (1.0 / s_data[i]) : 0.0;
        }
    }

    // A^+ = Vh^H @ diag(S_inv) @ U^H
    // Vh^H is just transpose for real matrices
    // U has shape (..., M, K), Vh has shape (..., K, N)
    // Transpose gives: VhT (..., N, K), UT (..., K, M)
    auto VhT = Vh.transpose(); // (..., N, K)
    auto UT = U.transpose();   // (..., K, M)

    // Result shape: (..., N, M)
    Shape result_shape = batch_shape;
    result_shape.push_back(n);
    result_shape.push_back(m);
    Tensor result = Tensor::zeros(result_shape, a.dtype());

    // Scale VhT columns by S_inv and multiply by UT
    // VhT_scaled[..., i, j] = VhT[..., i, j] * S_inv[..., j]
    if (a.dtype() == DType::Float32) {
        float *vht_data = VhT.typed_data<float>();
        float *ut_data = UT.typed_data<float>();
        float *s_inv_data = S_inv.typed_data<float>();
        float *result_data = result.typed_data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            float *vht_batch = vht_data + b * n * k;
            float *ut_batch = ut_data + b * k * m;
            float *s_inv_batch = s_inv_data + b * k;
            float *result_batch = result_data + b * n * m;

            // Create scaled VhT for this batch
            std::vector<float> vht_scaled(n * k);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    vht_scaled[i * k + j] =
                        vht_batch[i * k + j] * s_inv_batch[j];
                }
            }

            // Matrix multiply: result = VhT_scaled @ UT
            // (N, K) @ (K, M) -> (N, M)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; ++l) {
                        sum += vht_scaled[i * k + l] * ut_batch[l * m + j];
                    }
                    result_batch[i * m + j] = sum;
                }
            }
        }
    } else if (a.dtype() == DType::Float64) {
        double *vht_data = VhT.typed_data<double>();
        double *ut_data = UT.typed_data<double>();
        double *s_inv_data = S_inv.typed_data<double>();
        double *result_data = result.typed_data<double>();

        for (size_t b = 0; b < batch_size; ++b) {
            double *vht_batch = vht_data + b * n * k;
            double *ut_batch = ut_data + b * k * m;
            double *s_inv_batch = s_inv_data + b * k;
            double *result_batch = result_data + b * n * m;

            // Create scaled VhT for this batch
            std::vector<double> vht_scaled(n * k);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    vht_scaled[i * k + j] =
                        vht_batch[i * k + j] * s_inv_batch[j];
                }
            }

            // Matrix multiply: result = VhT_scaled @ UT
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    double sum = 0.0;
                    for (size_t l = 0; l < k; ++l) {
                        sum += vht_scaled[i * k + l] * ut_batch[l * m + j];
                    }
                    result_batch[i * m + j] = sum;
                }
            }
        }
    } else if (a.dtype() == DType::Complex64) {
        // For complex: A^+ = V @ diag(1/S) @ U^H
        // V = conj(Vh^T), U^H = conj(U^T)
        using complex64_t = std::complex<float>;

        // Apply conjugation for Hermitian transpose
        auto VhT_conj = ops::conj(VhT); // conj(Vh^T) = V
        auto UT_conj = ops::conj(UT);   // conj(U^T) = U^H

        complex64_t *vht_data = VhT_conj.typed_data<complex64_t>();
        complex64_t *ut_data = UT_conj.typed_data<complex64_t>();
        float *s_inv_data = S_inv.typed_data<float>();
        complex64_t *result_data = result.typed_data<complex64_t>();

        for (size_t b = 0; b < batch_size; ++b) {
            complex64_t *vht_batch = vht_data + b * n * k;
            complex64_t *ut_batch = ut_data + b * k * m;
            float *s_inv_batch = s_inv_data + b * k;
            complex64_t *result_batch = result_data + b * n * m;

            // Create scaled V for this batch
            std::vector<complex64_t> v_scaled(n * k);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    v_scaled[i * k + j] = vht_batch[i * k + j] * s_inv_batch[j];
                }
            }

            // Matrix multiply: result = V_scaled @ U^H
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    complex64_t sum(0.0f, 0.0f);
                    for (size_t l = 0; l < k; ++l) {
                        sum += v_scaled[i * k + l] * ut_batch[l * m + j];
                    }
                    result_batch[i * m + j] = sum;
                }
            }
        }
    } else if (a.dtype() == DType::Complex128) {
        using complex128_t = std::complex<double>;

        auto VhT_conj = ops::conj(VhT);
        auto UT_conj = ops::conj(UT);

        complex128_t *vht_data = VhT_conj.typed_data<complex128_t>();
        complex128_t *ut_data = UT_conj.typed_data<complex128_t>();
        double *s_inv_data = S_inv.typed_data<double>();
        complex128_t *result_data = result.typed_data<complex128_t>();

        for (size_t b = 0; b < batch_size; ++b) {
            complex128_t *vht_batch = vht_data + b * n * k;
            complex128_t *ut_batch = ut_data + b * k * m;
            double *s_inv_batch = s_inv_data + b * k;
            complex128_t *result_batch = result_data + b * n * m;

            std::vector<complex128_t> v_scaled(n * k);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    v_scaled[i * k + j] = vht_batch[i * k + j] * s_inv_batch[j];
                }
            }

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    complex128_t sum(0.0, 0.0);
                    for (size_t l = 0; l < k; ++l) {
                        sum += v_scaled[i * k + l] * ut_batch[l * m + j];
                    }
                    result_batch[i * m + j] = sum;
                }
            }
        }
    }

    return result;
}

Tensor norm(const Tensor &a, const std::string &ord) {
    if (ord == "fro") {
        // Frobenius norm: sqrt(sum(|x|^2))
        // For complex: |x|^2 = conj(x) * x = real(x)^2 + imag(x)^2
        if (a.dtype() == DType::Complex64 || a.dtype() == DType::Complex128) {
            auto abs_vals = a.abs(); // abs() gives magnitude for complex
            auto squared = ops::multiply(abs_vals, abs_vals);
            return ops::sqrt(squared.sum());
        } else {
            auto squared = ops::multiply(a, a);
            return ops::sqrt(squared.sum());
        }
    } else if (ord == "nuc") {
        // Nuclear norm: sum of singular values
        auto [U, S, Vh] = svd(a, false);
        return S.sum();
    } else {
        throw ValueError("Unknown norm type: " + ord);
    }
}

Tensor norm(const Tensor &a, int ord) {
    if (a.ndim() == 1) {
        // Vector norms
        if (ord == 0) {
            // L0 "norm": count of non-zero elements
            auto non_zero = ops::not_equal(a, Tensor::full({}, 0.0f));
            return non_zero.sum().astype(DType::Float64);
        } else if (ord == 1) {
            return a.abs().sum();
        } else if (ord == 2) {
            // For complex: ||x||_2 = sqrt(sum(|x|^2))
            auto abs_vals = a.abs();
            return ops::sqrt(ops::multiply(abs_vals, abs_vals).sum());
        } else {
            // General p-norm: (sum(|x|^p))^(1/p)
            auto p = static_cast<double>(ord);
            return ops::power(ops::power(a.abs(), Tensor::full({}, p)).sum(),
                              Tensor::full({}, 1.0 / p));
        }
    } else {
        // Matrix norms
        if (ord == 1) {
            // Max column sum
            return a.abs().sum(0).max();
        } else if (ord == 2) {
            // Spectral norm (largest singular value)
            auto [U, S, Vh] = svd(a, false);
            return S.max();
        } else {
            throw ValueError("Unsupported matrix norm order: " +
                             std::to_string(ord));
        }
    }
}

Tensor norm(const Tensor &a, double ord) {
    // General Lp norm for vectors
    // Already works with complex since a.abs() returns magnitude
    if (a.ndim() != 1) {
        throw ShapeError("Float ord only supported for vector norms");
    }
    if (ord == std::numeric_limits<double>::infinity()) {
        return a.abs().max();
    } else if (ord == -std::numeric_limits<double>::infinity()) {
        return a.abs().min();
    } else {
        return ops::power(ops::power(a.abs(), Tensor::full({}, ord)).sum(),
                          Tensor::full({}, 1.0 / ord));
    }
}

Tensor matrix_rank(const Tensor &a, double tol) {
    auto [U, S, Vh] = svd(a, false);

    if (tol < 0) {
        size_t k = std::min(a.shape()[a.ndim() - 2], a.shape()[a.ndim() - 1]);
        tol =
            k * std::numeric_limits<double>::epsilon() * S.max().item<double>();
    }

    // Count singular values > tol
    size_t rank = 0;
    if (S.dtype() == DType::Float32) {
        float *s_data = S.typed_data<float>();
        for (size_t i = 0; i < S.size(); ++i) {
            if (s_data[i] > tol) {
                ++rank;
            }
        }
    } else {
        double *s_data = S.typed_data<double>();
        for (size_t i = 0; i < S.size(); ++i) {
            if (s_data[i] > tol) {
                ++rank;
            }
        }
    }

    return Tensor::full({}, static_cast<int64_t>(rank), Device::CPU)
        .astype(DType::Int64);
}

Tensor cond(const Tensor &a, int p) {
    if (p == 2) {
        auto [U, S, Vh] = svd(a, false);
        return ops::divide(S.max(), S.min());
    } else if (p == 1 || p == -1) {
        // cond(A, 1) = norm(A, 1) * norm(inv(A), 1)
        auto a_inv = inv(a);
        return ops::multiply(norm(a, p), norm(a_inv, p));
    } else {
        throw ValueError("cond only supports p = 1, 2, or inf");
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

Tensor trace(const Tensor &a, int offset) { return a.trace(offset); }

Tensor matrix_power(const Tensor &a, int n) {
    validate_square_matrix(a, "matrix_power");

    if (n == 0) {
        size_t size = a.shape()[a.ndim() - 1];
        return Tensor::eye(size, a.dtype());
    } else if (n == 1) {
        return a.clone();
    } else if (n == -1) {
        return inv(a);
    } else if (n > 0) {
        // Binary exponentiation
        Tensor result = Tensor::eye(a.shape()[a.ndim() - 1], a.dtype());
        Tensor base = a.clone();

        while (n > 0) {
            if (n % 2 == 1) {
                result = result.matmul(base);
            }
            base = base.matmul(base);
            n /= 2;
        }
        return result;
    } else {
        // Negative power: inv(A)^|n|
        return matrix_power(inv(a), -n);
    }
}

std::pair<Tensor, Tensor> multi_dot(const std::vector<Tensor> &tensors) {
    if (tensors.empty()) {
        throw ValueError("multi_dot requires at least one tensor");
    }
    if (tensors.size() == 1) {
        return {tensors[0], Tensor()};
    }

    // Simple left-to-right multiplication (optimal ordering is NP-hard)
    Tensor result = tensors[0];
    for (size_t i = 1; i < tensors.size(); ++i) {
        result = result.matmul(tensors[i]);
    }
    return {result, Tensor()};
}

// ============================================================================
// Matrix and Vector Products (NumPy parity)
// ============================================================================

Tensor dot(const Tensor &a, const Tensor &b) {
    // NumPy dot semantics:
    // 1D @ 1D -> scalar (inner product)
    // 1D @ 2D -> (N,) @ (N,M) = (M,) vector-matrix
    // 2D @ 1D -> (M,N) @ (N,) = (M,) matrix-vector
    // 2D @ 2D -> matmul
    // ND @ ND -> sum product over last axis of a, second-to-last of b

    auto a_work = ensure_cpu_contiguous_colmajor(a);
    auto b_work = ensure_cpu_contiguous_colmajor(b);

    // 1D @ 1D: inner product -> scalar
    if (a.ndim() == 1 && b.ndim() == 1) {
        if (a.shape()[0] != b.shape()[0]) {
            throw ShapeError("dot: vectors must have same length, got " +
                             std::to_string(a.shape()[0]) + " and " +
                             std::to_string(b.shape()[0]));
        }

        size_t n = a.shape()[0];

        // Use BLAS for float/double
        if (a.dtype() == DType::Float32 && b.dtype() == DType::Float32) {
            auto &backend = backends::cpu::blas::get_blas_backend();
            float result = backend.sdot(n, a_work.typed_data<float>(), 1,
                                        b_work.typed_data<float>(), 1);
            return Tensor::full({}, result);
        } else if (a.dtype() == DType::Float64 && b.dtype() == DType::Float64) {
            auto &backend = backends::cpu::blas::get_blas_backend();
            double result = backend.ddot(n, a_work.typed_data<double>(), 1,
                                         b_work.typed_data<double>(), 1);
            return Tensor::full({}, result);
        } else {
            // Fallback for other types: promote and compute manually
            auto promoted_dtype = ops::promote_types(a.dtype(), b.dtype());
            auto a_promoted = a.astype(promoted_dtype);
            auto b_promoted = b.astype(promoted_dtype);

            if (promoted_dtype == DType::Complex64) {
                using cplx = std::complex<float>;
                const cplx *a_data = a_promoted.typed_data<cplx>();
                const cplx *b_data = b_promoted.typed_data<cplx>();
                cplx sum(0.0f, 0.0f);
                for (size_t i = 0; i < n; ++i) {
                    sum += a_data[i] * b_data[i];
                }
                return Tensor::full({}, sum);
            } else if (promoted_dtype == DType::Complex128) {
                using cplx = std::complex<double>;
                const cplx *a_data = a_promoted.typed_data<cplx>();
                const cplx *b_data = b_promoted.typed_data<cplx>();
                cplx sum(0.0, 0.0);
                for (size_t i = 0; i < n; ++i) {
                    sum += a_data[i] * b_data[i];
                }
                return Tensor::full({}, sum);
            } else {
                // Convert to float64 for other types
                return dot(a.astype(DType::Float64), b.astype(DType::Float64));
            }
        }
    }

    // 2D @ 2D or higher: use matmul
    if (a.ndim() >= 2 && b.ndim() >= 2) {
        return a.matmul(b);
    }

    // 1D @ 2D: (N,) @ (N, M) -> (M,)
    if (a.ndim() == 1 && b.ndim() == 2) {
        // Reshape a to (1, N), matmul, then squeeze
        auto a_reshaped = a.reshape({1, a.shape()[0]});
        auto result = a_reshaped.matmul(b);
        return result.squeeze(0);
    }

    // 2D @ 1D: (M, N) @ (N,) -> (M,)
    if (a.ndim() == 2 && b.ndim() == 1) {
        return matvec(a, b);
    }

    // ND @ 1D: sum product over last axis of a with b
    if (a.ndim() > 2 && b.ndim() == 1) {
        if (a.shape()[a.ndim() - 1] != b.shape()[0]) {
            throw ShapeError("dot: last axis of a must match b length");
        }
        // Reshape a to (..., K) as 2D: (prod_batch, K)
        size_t k = a.shape()[a.ndim() - 1];
        size_t batch = a.size() / k;
        auto a_2d = a.reshape({batch, k});
        auto result_1d = matvec(a_2d, b);
        // Reshape back to batch_shape
        Shape result_shape;
        for (size_t i = 0; i + 1 < a.ndim(); ++i) {
            result_shape.push_back(a.shape()[i]);
        }
        return result_1d.reshape(result_shape);
    }

    // 1D @ ND: not standard NumPy behavior, treat as (1, N) @ ND
    if (a.ndim() == 1 && b.ndim() > 2) {
        auto a_2d = a.reshape({1, a.shape()[0]});
        auto result = a_2d.matmul(b);
        return result.squeeze(-2);
    }

    throw ShapeError("dot: unsupported dimension combination");
}

Tensor vdot(const Tensor &a, const Tensor &b) {
    // vdot flattens both inputs and conjugates the first argument
    auto a_flat = a.flatten();
    auto b_flat = b.flatten();

    if (a_flat.shape()[0] != b_flat.shape()[0]) {
        throw ShapeError("vdot: arrays must have same number of elements");
    }

    // Conjugate first argument for complex types
    Tensor a_conj;
    if (a.dtype() == DType::Complex64 || a.dtype() == DType::Complex128) {
        a_conj = ops::conj(a_flat);
    } else {
        a_conj = a_flat;
    }

    return dot(a_conj, b_flat);
}

Tensor inner(const Tensor &a, const Tensor &b) {
    // Inner product: sum product over last axes
    // For 1D arrays: ordinary dot product (same as dot)
    // For ND arrays: sum over last axis of a and last axis of b

    if (a.ndim() == 1 && b.ndim() == 1) {
        return dot(a, b);
    }

    if (a.shape()[a.ndim() - 1] != b.shape()[b.ndim() - 1]) {
        throw ShapeError("inner: last dimensions must match, got " +
                         std::to_string(a.shape()[a.ndim() - 1]) + " and " +
                         std::to_string(b.shape()[b.ndim() - 1]));
    }

    // For ND arrays, result shape is a.shape[:-1] + b.shape[:-1]
    // We implement via transpose + matmul
    // inner(a, b) = a @ b.T for 2D case

    if (a.ndim() == 2 && b.ndim() == 2) {
        return a.matmul(b.transpose());
    }

    // General case: batch processing
    // Result shape: a.shape[:-1] + b.shape[:-1]
    Shape result_shape;
    for (size_t i = 0; i + 1 < a.ndim(); ++i) {
        result_shape.push_back(a.shape()[i]);
    }
    for (size_t i = 0; i + 1 < b.ndim(); ++i) {
        result_shape.push_back(b.shape()[i]);
    }

    size_t k = a.shape()[a.ndim() - 1];
    size_t a_batch = a.size() / k;
    size_t b_batch = b.size() / k;

    auto promoted_dtype = ops::promote_types(a.dtype(), b.dtype());
    auto a_work = a.astype(promoted_dtype).reshape({a_batch, k});
    auto b_work = b.astype(promoted_dtype).reshape({b_batch, k});

    Tensor result = Tensor(result_shape, promoted_dtype);

    if (promoted_dtype == DType::Float32) {
        float *result_data = result.typed_data<float>();
        const float *a_data = a_work.typed_data<float>();
        const float *b_data = b_work.typed_data<float>();
        auto &backend = backends::cpu::blas::get_blas_backend();

        for (size_t i = 0; i < a_batch; ++i) {
            for (size_t j = 0; j < b_batch; ++j) {
                result_data[i * b_batch + j] =
                    backend.sdot(k, a_data + i * k, 1, b_data + j * k, 1);
            }
        }
    } else if (promoted_dtype == DType::Float64) {
        double *result_data = result.typed_data<double>();
        const double *a_data = a_work.typed_data<double>();
        const double *b_data = b_work.typed_data<double>();
        auto &backend = backends::cpu::blas::get_blas_backend();

        for (size_t i = 0; i < a_batch; ++i) {
            for (size_t j = 0; j < b_batch; ++j) {
                result_data[i * b_batch + j] =
                    backend.ddot(k, a_data + i * k, 1, b_data + j * k, 1);
            }
        }
    } else {
        // Manual computation for complex types
        return inner(a.astype(DType::Float64), b.astype(DType::Float64));
    }

    return result;
}

Tensor outer(const Tensor &a, const Tensor &b) {
    // Outer product: flatten both inputs, compute outer product
    auto a_flat = a.flatten();
    auto b_flat = b.flatten();

    size_t m = a_flat.shape()[0];
    size_t n = b_flat.shape()[0];

    auto promoted_dtype = ops::promote_types(a.dtype(), b.dtype());
    auto a_work = a_flat.astype(promoted_dtype);
    auto b_work = b_flat.astype(promoted_dtype);

    Tensor result = Tensor({m, n}, promoted_dtype);

    if (promoted_dtype == DType::Float32) {
        float *result_data = result.typed_data<float>();
        const float *a_data = a_work.typed_data<float>();
        const float *b_data = b_work.typed_data<float>();

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result_data[i * n + j] = a_data[i] * b_data[j];
            }
        }
    } else if (promoted_dtype == DType::Float64) {
        double *result_data = result.typed_data<double>();
        const double *a_data = a_work.typed_data<double>();
        const double *b_data = b_work.typed_data<double>();

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result_data[i * n + j] = a_data[i] * b_data[j];
            }
        }
    } else if (promoted_dtype == DType::Complex64) {
        using cplx = std::complex<float>;
        cplx *result_data = result.typed_data<cplx>();
        const cplx *a_data = a_work.typed_data<cplx>();
        const cplx *b_data = b_work.typed_data<cplx>();

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result_data[i * n + j] = a_data[i] * b_data[j];
            }
        }
    } else if (promoted_dtype == DType::Complex128) {
        using cplx = std::complex<double>;
        cplx *result_data = result.typed_data<cplx>();
        const cplx *a_data = a_work.typed_data<cplx>();
        const cplx *b_data = b_work.typed_data<cplx>();

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result_data[i * n + j] = a_data[i] * b_data[j];
            }
        }
    } else {
        return outer(a.astype(DType::Float64), b.astype(DType::Float64));
    }

    return result;
}

Tensor matvec(const Tensor &x1, const Tensor &x2) {
    // Matrix-vector product: (..., M, N) @ (..., N) -> (..., M)
    if (x1.ndim() < 2) {
        throw ShapeError("matvec: first argument must be at least 2D matrix");
    }
    if (x2.ndim() < 1) {
        throw ShapeError("matvec: second argument must be at least 1D vector");
    }

    size_t m = x1.shape()[x1.ndim() - 2];
    size_t n = x1.shape()[x1.ndim() - 1];

    if (x2.shape()[x2.ndim() - 1] != n) {
        throw ShapeError("matvec: matrix columns (" + std::to_string(n) +
                         ") must match vector length (" +
                         std::to_string(x2.shape()[x2.ndim() - 1]) + ")");
    }

    auto promoted_dtype = ops::promote_types(x1.dtype(), x2.dtype());
    auto mat = ensure_cpu_contiguous_colmajor(x1.astype(promoted_dtype));
    auto vec = ensure_cpu_contiguous_colmajor(x2.astype(promoted_dtype));

    // Handle batching
    size_t batch_size = compute_batch_size(x1.shape());
    Shape batch_shape = get_batch_shape(x1.shape());

    Shape result_shape = batch_shape;
    result_shape.push_back(m);

    Tensor result = Tensor(result_shape, promoted_dtype);

    if (promoted_dtype == DType::Float32) {
        auto &backend = backends::cpu::blas::get_blas_backend();
        const float *mat_data = mat.typed_data<float>();
        const float *vec_data = vec.typed_data<float>();
        float *result_data = result.typed_data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            const float *mat_batch = mat_data + b * m * n;
            const float *vec_batch = vec_data + (x2.ndim() > 1 ? b * n : 0);
            float *result_batch = result_data + b * m;

            // BLAS gemv with row-major: y = A * x
            // A is (m, n), x is (n,), y is (m,)
            // sgemv(trans=false, M=m, N=n, alpha, A, lda=n, x, incx, beta, y,
            // incy)
            backend.sgemv(false, m, n, 1.0f, mat_batch, n, vec_batch, 1, 0.0f,
                          result_batch, 1);
        }
    } else if (promoted_dtype == DType::Float64) {
        auto &backend = backends::cpu::blas::get_blas_backend();
        const double *mat_data = mat.typed_data<double>();
        const double *vec_data = vec.typed_data<double>();
        double *result_data = result.typed_data<double>();

        for (size_t b = 0; b < batch_size; ++b) {
            const double *mat_batch = mat_data + b * m * n;
            const double *vec_batch = vec_data + (x2.ndim() > 1 ? b * n : 0);
            double *result_batch = result_data + b * m;

            backend.dgemv(false, m, n, 1.0, mat_batch, n, vec_batch, 1, 0.0,
                          result_batch, 1);
        }
    } else {
        // Manual computation for complex types
        if (promoted_dtype == DType::Complex64) {
            using cplx = std::complex<float>;
            const cplx *mat_data = mat.typed_data<cplx>();
            const cplx *vec_data = vec.typed_data<cplx>();
            cplx *result_data = result.typed_data<cplx>();

            for (size_t b = 0; b < batch_size; ++b) {
                const cplx *mat_batch = mat_data + b * m * n;
                const cplx *vec_batch = vec_data + (x2.ndim() > 1 ? b * n : 0);
                cplx *result_batch = result_data + b * m;

                for (size_t i = 0; i < m; ++i) {
                    cplx sum(0.0f, 0.0f);
                    for (size_t j = 0; j < n; ++j) {
                        sum += mat_batch[i * n + j] * vec_batch[j];
                    }
                    result_batch[i] = sum;
                }
            }
        } else if (promoted_dtype == DType::Complex128) {
            using cplx = std::complex<double>;
            const cplx *mat_data = mat.typed_data<cplx>();
            const cplx *vec_data = vec.typed_data<cplx>();
            cplx *result_data = result.typed_data<cplx>();

            for (size_t b = 0; b < batch_size; ++b) {
                const cplx *mat_batch = mat_data + b * m * n;
                const cplx *vec_batch = vec_data + (x2.ndim() > 1 ? b * n : 0);
                cplx *result_batch = result_data + b * m;

                for (size_t i = 0; i < m; ++i) {
                    cplx sum(0.0, 0.0);
                    for (size_t j = 0; j < n; ++j) {
                        sum += mat_batch[i * n + j] * vec_batch[j];
                    }
                    result_batch[i] = sum;
                }
            }
        } else {
            return matvec(x1.astype(DType::Float64), x2.astype(DType::Float64));
        }
    }

    return result;
}

Tensor vecmat(const Tensor &x1, const Tensor &x2) {
    // Vector-matrix product: (..., M) @ (..., M, N) -> (..., N)
    if (x1.ndim() < 1) {
        throw ShapeError("vecmat: first argument must be at least 1D vector");
    }
    if (x2.ndim() < 2) {
        throw ShapeError("vecmat: second argument must be at least 2D matrix");
    }

    size_t m = x2.shape()[x2.ndim() - 2];
    size_t n = x2.shape()[x2.ndim() - 1];

    if (x1.shape()[x1.ndim() - 1] != m) {
        throw ShapeError("vecmat: vector length (" +
                         std::to_string(x1.shape()[x1.ndim() - 1]) +
                         ") must match matrix rows (" + std::to_string(m) +
                         ")");
    }

    auto promoted_dtype = ops::promote_types(x1.dtype(), x2.dtype());
    auto vec = ensure_cpu_contiguous_colmajor(x1.astype(promoted_dtype));
    auto mat = ensure_cpu_contiguous_colmajor(x2.astype(promoted_dtype));

    // Handle batching
    size_t batch_size = compute_batch_size(x2.shape());
    Shape batch_shape = get_batch_shape(x2.shape());

    Shape result_shape = batch_shape;
    result_shape.push_back(n);

    Tensor result = Tensor(result_shape, promoted_dtype);

    if (promoted_dtype == DType::Float32) {
        auto &backend = backends::cpu::blas::get_blas_backend();
        const float *vec_data = vec.typed_data<float>();
        const float *mat_data = mat.typed_data<float>();
        float *result_data = result.typed_data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            const float *vec_batch = vec_data + (x1.ndim() > 1 ? b * m : 0);
            const float *mat_batch = mat_data + b * m * n;
            float *result_batch = result_data + b * n;

            // vecmat: v @ A where v is (m,), A is (m,n) -> (n,)
            // This equals A^T @ v for column vector interpretation
            // BLAS row-major: sgemv(trans=true, M=m, N=n, ...) computes y = A^T
            // @ x
            backend.sgemv(true, m, n, 1.0f, mat_batch, n, vec_batch, 1, 0.0f,
                          result_batch, 1);
        }
    } else if (promoted_dtype == DType::Float64) {
        auto &backend = backends::cpu::blas::get_blas_backend();
        const double *vec_data = vec.typed_data<double>();
        const double *mat_data = mat.typed_data<double>();
        double *result_data = result.typed_data<double>();

        for (size_t b = 0; b < batch_size; ++b) {
            const double *vec_batch = vec_data + (x1.ndim() > 1 ? b * m : 0);
            const double *mat_batch = mat_data + b * m * n;
            double *result_batch = result_data + b * n;

            backend.dgemv(true, m, n, 1.0, mat_batch, n, vec_batch, 1, 0.0,
                          result_batch, 1);
        }
    } else {
        // Manual computation for complex types
        if (promoted_dtype == DType::Complex64) {
            using cplx = std::complex<float>;
            const cplx *vec_data = vec.typed_data<cplx>();
            const cplx *mat_data = mat.typed_data<cplx>();
            cplx *result_data = result.typed_data<cplx>();

            for (size_t b = 0; b < batch_size; ++b) {
                const cplx *vec_batch = vec_data + (x1.ndim() > 1 ? b * m : 0);
                const cplx *mat_batch = mat_data + b * m * n;
                cplx *result_batch = result_data + b * n;

                for (size_t j = 0; j < n; ++j) {
                    cplx sum(0.0f, 0.0f);
                    for (size_t i = 0; i < m; ++i) {
                        sum += vec_batch[i] * mat_batch[i * n + j];
                    }
                    result_batch[j] = sum;
                }
            }
        } else if (promoted_dtype == DType::Complex128) {
            using cplx = std::complex<double>;
            const cplx *vec_data = vec.typed_data<cplx>();
            const cplx *mat_data = mat.typed_data<cplx>();
            cplx *result_data = result.typed_data<cplx>();

            for (size_t b = 0; b < batch_size; ++b) {
                const cplx *vec_batch = vec_data + (x1.ndim() > 1 ? b * m : 0);
                const cplx *mat_batch = mat_data + b * m * n;
                cplx *result_batch = result_data + b * n;

                for (size_t j = 0; j < n; ++j) {
                    cplx sum(0.0, 0.0);
                    for (size_t i = 0; i < m; ++i) {
                        sum += vec_batch[i] * mat_batch[i * n + j];
                    }
                    result_batch[j] = sum;
                }
            }
        } else {
            return vecmat(x1.astype(DType::Float64), x2.astype(DType::Float64));
        }
    }

    return result;
}

Tensor vecdot(const Tensor &x1, const Tensor &x2, int axis) {
    // Vector dot product along specified axis
    if (x1.shape() != x2.shape()) {
        throw ShapeError("vecdot: inputs must have the same shape");
    }

    // Normalize axis
    int ndim = static_cast<int>(x1.ndim());
    if (axis < 0) {
        axis += ndim;
    }
    if (axis < 0 || axis >= ndim) {
        throw ShapeError("vecdot: axis out of range");
    }

    size_t k = x1.shape()[axis];
    auto promoted_dtype = ops::promote_types(x1.dtype(), x2.dtype());
    auto a = x1.astype(promoted_dtype);
    auto b = x2.astype(promoted_dtype);

    // Result shape: remove the contracted axis
    Shape result_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i != axis) {
            result_shape.push_back(x1.shape()[i]);
        }
    }
    if (result_shape.empty()) {
        result_shape = {};
    }

    // Move contracted axis to last position for efficient iteration
    std::vector<int> perm;
    for (int i = 0; i < ndim; ++i) {
        if (i != axis) {
            perm.push_back(i);
        }
    }
    perm.push_back(axis);

    auto a_perm = a.transpose(perm).ascontiguousarray();
    auto b_perm = b.transpose(perm).ascontiguousarray();

    size_t outer_size = a_perm.size() / k;
    Tensor result = Tensor(result_shape, promoted_dtype);

    if (promoted_dtype == DType::Float32) {
        auto &backend = backends::cpu::blas::get_blas_backend();
        const float *a_data = a_perm.typed_data<float>();
        const float *b_data = b_perm.typed_data<float>();
        float *result_data = result.typed_data<float>();

        for (size_t i = 0; i < outer_size; ++i) {
            result_data[i] =
                backend.sdot(k, a_data + i * k, 1, b_data + i * k, 1);
        }
    } else if (promoted_dtype == DType::Float64) {
        auto &backend = backends::cpu::blas::get_blas_backend();
        const double *a_data = a_perm.typed_data<double>();
        const double *b_data = b_perm.typed_data<double>();
        double *result_data = result.typed_data<double>();

        for (size_t i = 0; i < outer_size; ++i) {
            result_data[i] =
                backend.ddot(k, a_data + i * k, 1, b_data + i * k, 1);
        }
    } else if (promoted_dtype == DType::Complex64) {
        using cplx = std::complex<float>;
        const cplx *a_data = a_perm.typed_data<cplx>();
        const cplx *b_data = b_perm.typed_data<cplx>();
        cplx *result_data = result.typed_data<cplx>();

        for (size_t i = 0; i < outer_size; ++i) {
            cplx sum(0.0f, 0.0f);
            for (size_t j = 0; j < k; ++j) {
                sum += a_data[i * k + j] * b_data[i * k + j];
            }
            result_data[i] = sum;
        }
    } else if (promoted_dtype == DType::Complex128) {
        using cplx = std::complex<double>;
        const cplx *a_data = a_perm.typed_data<cplx>();
        const cplx *b_data = b_perm.typed_data<cplx>();
        cplx *result_data = result.typed_data<cplx>();

        for (size_t i = 0; i < outer_size; ++i) {
            cplx sum(0.0, 0.0);
            for (size_t j = 0; j < k; ++j) {
                sum += a_data[i * k + j] * b_data[i * k + j];
            }
            result_data[i] = sum;
        }
    } else {
        return vecdot(x1.astype(DType::Float64), x2.astype(DType::Float64),
                      axis);
    }

    return result;
}

// ============================================================================
// Complex Products
// ============================================================================

Tensor tensordot(const Tensor &a, const Tensor &b, int axes) {
    // Contract last 'axes' dimensions of a with first 'axes' dimensions of b
    if (axes < 0) {
        throw ValueError("tensordot: axes must be non-negative");
    }
    if (static_cast<size_t>(axes) > a.ndim() ||
        static_cast<size_t>(axes) > b.ndim()) {
        throw ShapeError("tensordot: axes exceeds tensor dimensions");
    }

    // Validate contracted dimensions match
    for (int i = 0; i < axes; ++i) {
        size_t a_dim = a.shape()[a.ndim() - axes + i];
        size_t b_dim = b.shape()[i];
        if (a_dim != b_dim) {
            throw ShapeError("tensordot: dimension mismatch at axis " +
                             std::to_string(i));
        }
    }

    // Build axes pair
    std::vector<int> a_axes, b_axes;
    for (int i = 0; i < axes; ++i) {
        a_axes.push_back(static_cast<int>(a.ndim()) - axes + i);
        b_axes.push_back(i);
    }

    return tensordot(a, b, {a_axes, b_axes});
}

Tensor tensordot(const Tensor &a, const Tensor &b,
                 const std::pair<std::vector<int>, std::vector<int>> &axes) {
    // General tensor contraction
    // 1. Move contracted axes to end of a, beginning of b
    // 2. Reshape to 2D matrices
    // 3. matmul
    // 4. Reshape to result

    const auto &a_axes = axes.first;
    const auto &b_axes = axes.second;

    if (a_axes.size() != b_axes.size()) {
        throw ValueError("tensordot: axes lists must have same length");
    }

    int n_contract = static_cast<int>(a_axes.size());

    // Normalize and validate axes
    std::vector<int> a_axes_norm, b_axes_norm;
    for (int ax : a_axes) {
        int norm_ax = ax < 0 ? ax + static_cast<int>(a.ndim()) : ax;
        if (norm_ax < 0 || norm_ax >= static_cast<int>(a.ndim())) {
            throw ShapeError("tensordot: invalid axis for first tensor");
        }
        a_axes_norm.push_back(norm_ax);
    }
    for (int ax : b_axes) {
        int norm_ax = ax < 0 ? ax + static_cast<int>(b.ndim()) : ax;
        if (norm_ax < 0 || norm_ax >= static_cast<int>(b.ndim())) {
            throw ShapeError("tensordot: invalid axis for second tensor");
        }
        b_axes_norm.push_back(norm_ax);
    }

    // Check contracted dimensions match
    for (size_t i = 0; i < a_axes_norm.size(); ++i) {
        if (a.shape()[a_axes_norm[i]] != b.shape()[b_axes_norm[i]]) {
            throw ShapeError("tensordot: contracted dimensions must match");
        }
    }

    // Build permutation for a: free axes first, then contracted
    std::vector<int> a_perm;
    std::set<int> a_contracted(a_axes_norm.begin(), a_axes_norm.end());
    for (int i = 0; i < static_cast<int>(a.ndim()); ++i) {
        if (a_contracted.find(i) == a_contracted.end()) {
            a_perm.push_back(i);
        }
    }
    for (int ax : a_axes_norm) {
        a_perm.push_back(ax);
    }

    // Build permutation for b: contracted first, then free
    std::vector<int> b_perm;
    for (int ax : b_axes_norm) {
        b_perm.push_back(ax);
    }
    std::set<int> b_contracted(b_axes_norm.begin(), b_axes_norm.end());
    for (int i = 0; i < static_cast<int>(b.ndim()); ++i) {
        if (b_contracted.find(i) == b_contracted.end()) {
            b_perm.push_back(i);
        }
    }

    // Calculate shapes
    size_t a_free_size = 1, a_contract_size = 1;
    Shape a_free_shape;
    for (int i = 0; i < static_cast<int>(a.ndim()) - n_contract; ++i) {
        size_t dim = a.shape()[a_perm[i]];
        a_free_size *= dim;
        a_free_shape.push_back(dim);
    }
    for (int i = 0; i < n_contract; ++i) {
        a_contract_size *= a.shape()[a_axes_norm[i]];
    }

    size_t b_free_size = 1;
    Shape b_free_shape;
    for (int i = n_contract; i < static_cast<int>(b.ndim()); ++i) {
        size_t dim = b.shape()[b_perm[i]];
        b_free_size *= dim;
        b_free_shape.push_back(dim);
    }

    // Permute and reshape
    auto a_perm_t = a.transpose(a_perm).ascontiguousarray();
    auto b_perm_t = b.transpose(b_perm).ascontiguousarray();

    auto a_2d = a_perm_t.reshape({a_free_size, a_contract_size});
    auto b_2d = b_perm_t.reshape({a_contract_size, b_free_size});

    // Matrix multiply
    auto result_2d = a_2d.matmul(b_2d);

    // Reshape to final shape
    Shape result_shape;
    for (size_t dim : a_free_shape) {
        result_shape.push_back(dim);
    }
    for (size_t dim : b_free_shape) {
        result_shape.push_back(dim);
    }

    if (result_shape.empty()) {
        // Scalar result
        return result_2d.reshape({});
    }

    return result_2d.reshape(result_shape);
}

Tensor kron(const Tensor &a, const Tensor &b) {
    // Kronecker product
    // For 2D: result[i*p+k, j*q+l] = a[i,j] * b[k,l]
    // General approach: use broadcasting and reshape

    auto promoted_dtype = ops::promote_types(a.dtype(), b.dtype());
    auto a_work = a.astype(promoted_dtype);
    auto b_work = b.astype(promoted_dtype);

    // Handle dimensions - extend to at least 2D
    size_t ndim = std::max(a.ndim(), b.ndim());
    ndim = std::max(ndim, static_cast<size_t>(2));

    // Pad shapes to same ndim
    Shape a_shape = a.shape();
    Shape b_shape = b.shape();
    while (a_shape.size() < ndim) {
        a_shape.insert(a_shape.begin(), 1);
    }
    while (b_shape.size() < ndim) {
        b_shape.insert(b_shape.begin(), 1);
    }

    auto a_reshaped = a_work.reshape(a_shape);
    auto b_reshaped = b_work.reshape(b_shape);

    // For Kronecker product, we interleave dimensions:
    // a.reshape(..., m, 1, n, 1) * b.reshape(..., 1, p, 1, q)
    // -> reshape to (..., m*p, n*q)

    // Build interleaved shapes
    Shape a_interleaved, b_interleaved;
    for (size_t i = 0; i < ndim; ++i) {
        a_interleaved.push_back(a_shape[i]);
        a_interleaved.push_back(1);
    }
    for (size_t i = 0; i < ndim; ++i) {
        b_interleaved.push_back(1);
        b_interleaved.push_back(b_shape[i]);
    }

    auto a_int = a_reshaped.reshape(a_interleaved);
    auto b_int = b_reshaped.reshape(b_interleaved);

    // Multiply with broadcasting
    auto product = ops::multiply(a_int, b_int);

    // Reshape to final result
    Shape result_shape;
    for (size_t i = 0; i < ndim; ++i) {
        result_shape.push_back(a_shape[i] * b_shape[i]);
    }

    return product.reshape(result_shape);
}

Tensor cross(const Tensor &x1, const Tensor &x2, int axis) {
    // Cross product of two 3-element vectors
    if (x1.shape() != x2.shape()) {
        throw ShapeError("cross: inputs must have the same shape");
    }

    // Normalize axis
    int ndim = static_cast<int>(x1.ndim());
    if (axis < 0) {
        axis += ndim;
    }
    if (axis < 0 || axis >= ndim) {
        throw ShapeError("cross: axis out of range");
    }

    if (x1.shape()[axis] != 3) {
        throw ShapeError("cross: dimension at axis must be 3, got " +
                         std::to_string(x1.shape()[axis]));
    }

    auto promoted_dtype = ops::promote_types(x1.dtype(), x2.dtype());
    auto a = x1.astype(promoted_dtype);
    auto b = x2.astype(promoted_dtype);

    // Move axis to last position for easier indexing
    std::vector<int> perm;
    for (int i = 0; i < ndim; ++i) {
        if (i != axis) {
            perm.push_back(i);
        }
    }
    perm.push_back(axis);

    auto a_perm = a.transpose(perm).ascontiguousarray();
    auto b_perm = b.transpose(perm).ascontiguousarray();

    // Result has same shape
    Tensor result = Tensor(a_perm.shape(), promoted_dtype);

    size_t outer_size = a_perm.size() / 3;

    // Cross product formula: c = a x b
    // c[0] = a[1]*b[2] - a[2]*b[1]
    // c[1] = a[2]*b[0] - a[0]*b[2]
    // c[2] = a[0]*b[1] - a[1]*b[0]

    if (promoted_dtype == DType::Float32) {
        const float *a_data = a_perm.typed_data<float>();
        const float *b_data = b_perm.typed_data<float>();
        float *result_data = result.typed_data<float>();

        for (size_t i = 0; i < outer_size; ++i) {
            const float *ai = a_data + i * 3;
            const float *bi = b_data + i * 3;
            float *ci = result_data + i * 3;

            ci[0] = ai[1] * bi[2] - ai[2] * bi[1];
            ci[1] = ai[2] * bi[0] - ai[0] * bi[2];
            ci[2] = ai[0] * bi[1] - ai[1] * bi[0];
        }
    } else if (promoted_dtype == DType::Float64) {
        const double *a_data = a_perm.typed_data<double>();
        const double *b_data = b_perm.typed_data<double>();
        double *result_data = result.typed_data<double>();

        for (size_t i = 0; i < outer_size; ++i) {
            const double *ai = a_data + i * 3;
            const double *bi = b_data + i * 3;
            double *ci = result_data + i * 3;

            ci[0] = ai[1] * bi[2] - ai[2] * bi[1];
            ci[1] = ai[2] * bi[0] - ai[0] * bi[2];
            ci[2] = ai[0] * bi[1] - ai[1] * bi[0];
        }
    } else if (promoted_dtype == DType::Complex64) {
        using cplx = std::complex<float>;
        const cplx *a_data = a_perm.typed_data<cplx>();
        const cplx *b_data = b_perm.typed_data<cplx>();
        cplx *result_data = result.typed_data<cplx>();

        for (size_t i = 0; i < outer_size; ++i) {
            const cplx *ai = a_data + i * 3;
            const cplx *bi = b_data + i * 3;
            cplx *ci = result_data + i * 3;

            ci[0] = ai[1] * bi[2] - ai[2] * bi[1];
            ci[1] = ai[2] * bi[0] - ai[0] * bi[2];
            ci[2] = ai[0] * bi[1] - ai[1] * bi[0];
        }
    } else if (promoted_dtype == DType::Complex128) {
        using cplx = std::complex<double>;
        const cplx *a_data = a_perm.typed_data<cplx>();
        const cplx *b_data = b_perm.typed_data<cplx>();
        cplx *result_data = result.typed_data<cplx>();

        for (size_t i = 0; i < outer_size; ++i) {
            const cplx *ai = a_data + i * 3;
            const cplx *bi = b_data + i * 3;
            cplx *ci = result_data + i * 3;

            ci[0] = ai[1] * bi[2] - ai[2] * bi[1];
            ci[1] = ai[2] * bi[0] - ai[0] * bi[2];
            ci[2] = ai[0] * bi[1] - ai[1] * bi[0];
        }
    } else {
        return cross(x1.astype(DType::Float64), x2.astype(DType::Float64),
                     axis);
    }

    // Permute back to original axis order
    std::vector<int> inv_perm(ndim);
    for (int i = 0; i < ndim; ++i) {
        inv_perm[perm[i]] = i;
    }

    return result.transpose(inv_perm);
}

// ============================================================================
// Decomposition Variants
// ============================================================================

Tensor svdvals(const Tensor &x) {
    // Singular values only - more efficient wrapper around svd
    auto [U, S, Vh] = svd(x, false);
    return S;
}

Tensor eigvals(const Tensor &a) {
    // Eigenvalues only - wrapper around eig
    auto [eigenvalues, eigenvectors] = eig(a);
    return eigenvalues;
}

Tensor eigvalsh(const Tensor &a) {
    // Eigenvalues of symmetric/Hermitian matrix only
    auto [eigenvalues, eigenvectors] = eigh(a);
    return eigenvalues;
}

std::pair<Tensor, Tensor> slogdet(const Tensor &a) {
    // Sign and log determinant using LU decomposition
    // More numerically stable than computing det directly
    validate_square_matrix(a, "slogdet");

    auto &backend = get_lapack_backend();

    size_t n = a.shape()[a.ndim() - 1];
    size_t batch_size = compute_batch_size(a.shape());
    Shape batch_shape = get_batch_shape(a.shape());

    Shape result_shape = batch_shape.empty() ? Shape({}) : batch_shape;

    Tensor sign_result, logabsdet_result;

    if (a.dtype() == DType::Float32) {
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        float *a_data = a_work.typed_data<float>();

        sign_result = Tensor(result_shape, DType::Float32);
        logabsdet_result = Tensor(result_shape, DType::Float32);
        float *sign_data = sign_result.typed_data<float>();
        float *logdet_data = logabsdet_result.typed_data<float>();

        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            float *a_batch = a_data + b * n * n;

            // Transpose to column-major
            std::vector<float> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info =
                backend.sgetrf(static_cast<int>(n), static_cast<int>(n),
                               a_col.data(), static_cast<int>(n), ipiv.data());

            if (info > 0) {
                // Singular matrix
                sign_data[b] = 0.0f;
                logdet_data[b] = -std::numeric_limits<float>::infinity();
            } else if (info < 0) {
                check_lapack_info(info, "sgetrf");
            } else {
                // Compute sign and log|det| from diagonal of U
                float log_abs_det = 0.0f;
                int sign = 1;

                for (size_t i = 0; i < n; ++i) {
                    float diag = a_col[i * n + i];
                    if (diag < 0) {
                        sign = -sign;
                        log_abs_det += std::log(-diag);
                    } else if (diag > 0) {
                        log_abs_det += std::log(diag);
                    } else {
                        // Zero diagonal -> singular
                        sign = 0;
                        log_abs_det = -std::numeric_limits<float>::infinity();
                        break;
                    }

                    // Account for pivoting
                    if (ipiv[i] != static_cast<int>(i + 1)) {
                        sign = -sign;
                    }
                }

                sign_data[b] = static_cast<float>(sign);
                logdet_data[b] = log_abs_det;
            }
        }
    } else if (a.dtype() == DType::Float64) {
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Float64)).clone();
        double *a_data = a_work.typed_data<double>();

        sign_result = Tensor(result_shape, DType::Float64);
        logabsdet_result = Tensor(result_shape, DType::Float64);
        double *sign_data = sign_result.typed_data<double>();
        double *logdet_data = logabsdet_result.typed_data<double>();

        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            double *a_batch = a_data + b * n * n;

            std::vector<double> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info =
                backend.dgetrf(static_cast<int>(n), static_cast<int>(n),
                               a_col.data(), static_cast<int>(n), ipiv.data());

            if (info > 0) {
                sign_data[b] = 0.0;
                logdet_data[b] = -std::numeric_limits<double>::infinity();
            } else if (info < 0) {
                check_lapack_info(info, "dgetrf");
            } else {
                double log_abs_det = 0.0;
                int sign = 1;

                for (size_t i = 0; i < n; ++i) {
                    double diag = a_col[i * n + i];
                    if (diag < 0) {
                        sign = -sign;
                        log_abs_det += std::log(-diag);
                    } else if (diag > 0) {
                        log_abs_det += std::log(diag);
                    } else {
                        sign = 0;
                        log_abs_det = -std::numeric_limits<double>::infinity();
                        break;
                    }

                    if (ipiv[i] != static_cast<int>(i + 1)) {
                        sign = -sign;
                    }
                }

                sign_data[b] = static_cast<double>(sign);
                logdet_data[b] = log_abs_det;
            }
        }
    } else if (a.dtype() == DType::Complex64) {
        using complex64_t = std::complex<float>;
        auto a_work = ensure_cpu_contiguous_colmajor(a).clone();
        complex64_t *a_data = a_work.typed_data<complex64_t>();

        sign_result = Tensor(result_shape, DType::Complex64);
        logabsdet_result = Tensor(result_shape, DType::Float32);
        complex64_t *sign_data = sign_result.typed_data<complex64_t>();
        float *logdet_data = logabsdet_result.typed_data<float>();

        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            complex64_t *a_batch = a_data + b * n * n;

            std::vector<complex64_t> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info =
                backend.cgetrf(static_cast<int>(n), static_cast<int>(n),
                               a_col.data(), static_cast<int>(n), ipiv.data());

            if (info > 0) {
                sign_data[b] = complex64_t(0.0f, 0.0f);
                logdet_data[b] = -std::numeric_limits<float>::infinity();
            } else if (info < 0) {
                check_lapack_info(info, "cgetrf");
            } else {
                float log_abs_det = 0.0f;
                complex64_t sign_val(1.0f, 0.0f);

                for (size_t i = 0; i < n; ++i) {
                    complex64_t diag = a_col[i * n + i];
                    float abs_diag = std::abs(diag);

                    if (abs_diag == 0.0f) {
                        sign_val = complex64_t(0.0f, 0.0f);
                        log_abs_det = -std::numeric_limits<float>::infinity();
                        break;
                    }

                    log_abs_det += std::log(abs_diag);
                    sign_val *= diag / abs_diag; // Unit complex number

                    if (ipiv[i] != static_cast<int>(i + 1)) {
                        sign_val = -sign_val;
                    }
                }

                sign_data[b] = sign_val;
                logdet_data[b] = log_abs_det;
            }
        }
    } else if (a.dtype() == DType::Complex128) {
        using complex128_t = std::complex<double>;
        auto a_work =
            ensure_cpu_contiguous_colmajor(a.astype(DType::Complex128)).clone();
        complex128_t *a_data = a_work.typed_data<complex128_t>();

        sign_result = Tensor(result_shape, DType::Complex128);
        logabsdet_result = Tensor(result_shape, DType::Float64);
        complex128_t *sign_data = sign_result.typed_data<complex128_t>();
        double *logdet_data = logabsdet_result.typed_data<double>();

        std::vector<int> ipiv(n);

        for (size_t b = 0; b < batch_size; ++b) {
            complex128_t *a_batch = a_data + b * n * n;

            std::vector<complex128_t> a_col(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    a_col[j * n + i] = a_batch[i * n + j];
                }
            }

            int info =
                backend.zgetrf(static_cast<int>(n), static_cast<int>(n),
                               a_col.data(), static_cast<int>(n), ipiv.data());

            if (info > 0) {
                sign_data[b] = complex128_t(0.0, 0.0);
                logdet_data[b] = -std::numeric_limits<double>::infinity();
            } else if (info < 0) {
                check_lapack_info(info, "zgetrf");
            } else {
                double log_abs_det = 0.0;
                complex128_t sign_val(1.0, 0.0);

                for (size_t i = 0; i < n; ++i) {
                    complex128_t diag = a_col[i * n + i];
                    double abs_diag = std::abs(diag);

                    if (abs_diag == 0.0) {
                        sign_val = complex128_t(0.0, 0.0);
                        log_abs_det = -std::numeric_limits<double>::infinity();
                        break;
                    }

                    log_abs_det += std::log(abs_diag);
                    sign_val *= diag / abs_diag;

                    if (ipiv[i] != static_cast<int>(i + 1)) {
                        sign_val = -sign_val;
                    }
                }

                sign_data[b] = sign_val;
                logdet_data[b] = log_abs_det;
            }
        }
    } else {
        return slogdet(a.astype(DType::Float64));
    }

    return {sign_result, logabsdet_result};
}

} // namespace linalg
} // namespace axiom
