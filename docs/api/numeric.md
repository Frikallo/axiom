# Numeric Constants

Constants, special value handling, and safe arithmetic in the `axiom::numeric` namespace.

## Constants

### NaN and Infinity

```cpp
constexpr float  numeric::nan_f;          // float quiet NaN
constexpr double numeric::nan_d;          // double quiet NaN
constexpr float  numeric::inf_f;          // float +infinity
constexpr double numeric::inf_d;          // double +infinity
constexpr float  numeric::neg_inf_f;      // float -infinity
constexpr double numeric::neg_inf_d;      // double -infinity

template <typename T> constexpr T numeric::nan();
template <typename T> constexpr T numeric::inf();
template <typename T> constexpr T numeric::neg_inf();
```

### Machine Epsilon

```cpp
constexpr float  numeric::epsilon_f;
constexpr double numeric::epsilon_d;

template <typename T> constexpr T numeric::epsilon();
```

### Mathematical Constants

```cpp
constexpr float  numeric::pi_f;
constexpr double numeric::pi_d;
constexpr float  numeric::e_f;
constexpr double numeric::e_d;
constexpr float  numeric::euler_gamma_f;
constexpr double numeric::euler_gamma_d;

template <typename T> constexpr T numeric::pi();
template <typename T> constexpr T numeric::e();
template <typename T> constexpr T numeric::euler_gamma();
```

---

## Value Classification

```cpp
template <typename T> bool numeric::is_nan(T value);
template <typename T> bool numeric::is_inf(T value);
template <typename T> bool numeric::is_pos_inf(T value);
template <typename T> bool numeric::is_neg_inf(T value);
template <typename T> bool numeric::is_finite(T value);
template <typename T> bool numeric::is_normal(T value);
```

These work on scalar values. For tensor-level checks, use `tensor.has_nan()`, `tensor.has_inf()`, or `ops::isnan()`.

Convenience aliases in the `axiom` namespace: `is_nan`, `is_inf`, `is_finite`.

---

## Safe Arithmetic

```cpp
template <typename T> T numeric::safe_div(T a, T b);
template <typename T> T numeric::safe_log(T value);
template <typename T> T numeric::safe_sqrt(T value);
```

Return NaN/Inf instead of throwing on edge cases:
- `safe_div(0, 0)` returns NaN; `safe_div(1, 0)` returns Inf.
- `safe_log(x < 0)` returns NaN; `safe_log(0)` returns -Inf.
- `safe_sqrt(x < 0)` returns NaN.

---

## Approximate Comparison

```cpp
template <typename T>
bool numeric::approx_equal(T a, T b,
                           T rel_tol = epsilon<T>() * 100,
                           T abs_tol = epsilon<T>());
```

Floating-point approximate equality. Returns `false` for NaN inputs.

---

## Numeric Formatting

```cpp
struct NumericFormat {
    const char *nan_str = "nan";
    const char *pos_inf_str = "inf";
    const char *neg_inf_str = "-inf";
    int precision = 4;
    bool fixed = true;
};

NumericFormat &numeric::default_format();

template <typename T>
std::string numeric::to_string(T value, const NumericFormat &fmt = default_format());
```

Format values with proper NaN/Inf handling.

**See Also:** [Data Types](dtypes), [Errors](errors)
