#pragma once

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <variant>

#include "axiom/float16.hpp"

namespace axiom {

// float16_t is defined in axiom/float16.hpp
using complex64_t = std::complex<float>;
using complex128_t = std::complex<double>;

// The overload pattern template
template <class... Ts> struct overload : Ts... {
    using Ts::operator()...; // Bring all operator() into scope
};

template <class T> class DType {
  public:
    using value_type = T;

    template <typename E> struct is_complex_t : public std::false_type {};

    template <typename E>
    struct is_complex_t<std::complex<E>> : public std::true_type {};

    template <class F, class... Args> void dispatch(F &&f, Args &&...args) {
        f(*this, std::forward<Args>(args)...);
    }
    constexpr size_t dtype_size() { return sizeof(T); }
    constexpr bool is_complex() { return is_complex_t<value_type>::value; }
    constexpr bool is_float() { return std::is_floating_point_v<value_type>; }
    constexpr bool is_int() { return std::is_integral_v<value_type>; }
    constexpr bool is_unsigned() { return std::is_unsigned_v<value_type>; }
    constexpr bool is_signed() { return std::is_signed_v<value_type>; }
};

class Bool : public DType<bool> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "Bool"; }
};

class Int8 : public DType<int8_t> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "Int8"; }
};

class Int16 : public DType<int16_t> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "Int16"; }
};
class Int32 : public DType<int32_t> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "int32"; }
};
class Int64 : public DType<int64_t> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "Int64"; }
};

class UInt8 : public DType<uint8_t> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "UInt8"; }
};

class UInt16 : public DType<uint16_t> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "UInt16"; }
};
class UInt32 : public DType<uint32_t> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "UInt32"; }
};
class UInt64 : public DType<uint64_t> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "UInt64"; }
};
class Float16 : public DType<float16_t> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "Float16"; }
};
class Float32 : public DType<float> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "Float32"; }
};
class Float64 : public DType<double> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "Float64"; }
};

class Complex64 : public DType<std::complex<float>> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "Complex64"; }
};

class Complex128 : public DType<std::complex<double>> {
  public:
    using value_type = typename DType::value_type;
    using DType::dispatch;
    using DType::dtype_size;
    using DType::is_complex;
    using DType::is_float;
    using DType::is_int;
    using DType::is_signed;
    using DType::is_unsigned;
    value_type one() const;
    value_type zeros() const;
    std::string name() const { return "Complex128"; }
};

using DTypes = ::std::variant<Bool, Int8, Int16, Int32, Int64,

                              UInt8, UInt16, UInt32, UInt64,

                              Float16, Float32, Float64,

                              Complex64, Complex128>;

bool is_complex(DTypes dtype);
bool is_float(DTypes dtype);
bool is_int(DTypes dtype);
bool is_signed(DTypes dtype);
bool is_unsigned(DTypes dtype);
bool is_float(DTypes dtype);
size_t dtype_size(DTypes dtype);
std::string dtype_name(DTypes dtype);

namespace detail {
template <class T, class... INNER>
constexpr auto dtype_of_v(std::variant<INNER...> /*args*/) {
    std::optional<DTypes> init;
    std::tuple<INNER...> t;
    auto func = [&]<class E>(E &&t) {
        if constexpr (std::is_same_v<std::decay_t<T>,
                                     typename std::decay_t<E>::value_type>) {
            init = std::make_optional(t);
        }
    };
    std::apply([&]<typename... E>(E... es) { (func(es), ...); }, t);

    return std::visit(overload{[](auto t) { return DTypes{t}; }}, init.value());
}
} // namespace detail

template <class T> constexpr DTypes dtype_of_v() {
    return detail::dtype_of_v<T>(DTypes{Bool()});
}

} // namespace axiom