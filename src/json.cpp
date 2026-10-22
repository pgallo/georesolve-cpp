#include "georesolve/json.hpp"

#include <cctype>
#include <cstdlib>

namespace georesolve::json {

const Value* Value::find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& [k, v] : object) {
        if (k == key) return &v;
    }
    return nullptr;
}

const Value& Value::at(const std::string& key) const {
    const Value* v = find(key);
    if (v == nullptr) throw ParseError("missing key: " + key);
    return *v;
}

namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    Value parseDocument() {
        Value v = parseValue();
        skipWs();
        if (pos_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    const std::string& s_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw ParseError("json: " + what + " at offset " + std::to_string(pos_));
    }

    void skipWs() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }

    char peek() {
        skipWs();
        if (pos_ >= s_.size()) fail("unexpected end of input");
        return s_[pos_];
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++pos_;
    }

    bool consumeLiteral(const char* lit) {
        std::size_t n = std::char_traits<char>::length(lit);
        if (s_.compare(pos_, n, lit) == 0) {
            pos_ += n;
            return true;
        }
        return false;
    }

    Value parseValue() {
        char c = peek();
        Value v;
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') {
            v.type = Value::Type::String;
            v.string = parseString();
            return v;
        }
        if (consumeLiteral("true")) {
            v.type = Value::Type::Bool;
            v.boolean = true;
            return v;
        }
        if (consumeLiteral("false")) {
            v.type = Value::Type::Bool;
            return v;
        }
        if (consumeLiteral("null")) return v;
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parseNumber();
        fail("unexpected character");
    }

    Value parseObject() {
        Value v;
        v.type = Value::Type::Object;
        expect('{');
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        while (true) {
            if (peek() != '"') fail("expected object key");
            std::string key = parseString();
            expect(':');
            v.object.emplace_back(std::move(key), parseValue());
            char c = peek();
            ++pos_;
            if (c == '}') break;
            if (c != ',') fail("expected ',' or '}'");
        }
        return v;
    }

    Value parseArray() {
        Value v;
        v.type = Value::Type::Array;
        expect('[');
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        while (true) {
            v.array.push_back(parseValue());
            char c = peek();
            ++pos_;
            if (c == ']') break;
            if (c != ',') fail("expected ',' or ']'");
        }
        return v;
    }

    Value parseNumber() {
        const char* begin = s_.c_str() + pos_;
        char* end = nullptr;
        double d = std::strtod(begin, &end);
        if (end == begin) fail("bad number");
        pos_ += static_cast<std::size_t>(end - begin);
        Value v;
        v.type = Value::Type::Number;
        v.number = d;
        return v;
    }

    static void appendUtf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) fail("unterminated string");
            char c = s_[pos_++];
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size()) fail("bad escape");
            char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (pos_ + 4 > s_.size()) fail("bad unicode escape");
                    unsigned cp = static_cast<unsigned>(std::stoul(s_.substr(pos_, 4), nullptr, 16));
                    pos_ += 4;
                    appendUtf8(out, cp);
                    break;
                }
                default: fail("unknown escape");
            }
        }
        return out;
    }
};

}  // namespace

Value parse(const std::string& text) { return Parser(text).parseDocument(); }

}  // namespace georesolve::json
