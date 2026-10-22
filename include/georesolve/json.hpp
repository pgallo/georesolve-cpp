#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Minimal JSON reader, sufficient for the JSONL corpora in data/.
// Kept dependency-free on purpose: the project builds with only a compiler,
// CMake and GoogleTest.
namespace georesolve::json {

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::vector<std::pair<std::string, Value>> object;

    bool isString() const { return type == Type::String; }
    bool isArray() const { return type == Type::Array; }
    bool isObject() const { return type == Type::Object; }

    // Returns nullptr when the key is absent or this is not an object.
    const Value* find(const std::string& key) const;
    // Throws ParseError when the key is absent.
    const Value& at(const std::string& key) const;
};

Value parse(const std::string& text);

}  // namespace georesolve::json
