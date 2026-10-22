#include "georesolve/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <unordered_set>

namespace georesolve {

namespace {

bool isWordByte(unsigned char c) { return std::isalnum(c) != 0 || c >= 0x80; }

// Fold the second byte of a UTF-8 sequence starting with 0xC3 (U+00C0..U+00FF)
// to its ASCII base letter. Returns 0 when there is no sensible ASCII fold.
char foldLatin1(unsigned char second) {
    // Index = second - 0x80, covering U+00C0..U+00FF.
    static constexpr std::array<char, 64> table = {
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i',  // C0-CF
        'd', 'n', 'o', 'o', 'o', 'o', 'o', 0,   'o', 'u', 'u', 'u', 'u', 'y', 0,   's',  // D0-DF
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i',  // E0-EF
        'd', 'n', 'o', 'o', 'o', 'o', 'o', 0,   'o', 'u', 'u', 'u', 'u', 'y', 0,   'y',  // F0-FF
    };
    if (second < 0x80 || second > 0xBF) return 0;
    return table[second - 0x80];
}

bool startsUpper(std::string_view w) {
    if (w.empty()) return false;
    auto c0 = static_cast<unsigned char>(w[0]);
    if (std::isupper(c0)) return true;
    // Upper-case Latin-1 letters: 0xC3 0x80..0x9E
    return w.size() > 1 && c0 == 0xC3 && static_cast<unsigned char>(w[1]) >= 0x80 &&
           static_cast<unsigned char>(w[1]) <= 0x9E;
}

}  // namespace

std::string normalize(std::string_view word) {
    std::string out;
    out.reserve(word.size());
    for (std::size_t i = 0; i < word.size(); ++i) {
        auto c = static_cast<unsigned char>(word[i]);
        if (c == 0xC3 && i + 1 < word.size()) {
            char f = foldLatin1(static_cast<unsigned char>(word[i + 1]));
            if (f != 0) {
                out += f;
                ++i;
                continue;
            }
        }
        out += static_cast<char>(std::tolower(c));
    }
    return out;
}

std::vector<Token> tokenize(std::string_view text) {
    std::vector<Token> tokens;
    std::size_t i = 0;
    char pendingPunct = 0;
    while (i < text.size()) {
        auto c = static_cast<unsigned char>(text[i]);
        if (!isWordByte(c)) {
            if (pendingPunct == 0 && std::ispunct(c)) pendingPunct = static_cast<char>(c);
            ++i;
            continue;
        }
        std::size_t start = i;
        while (i < text.size() && isWordByte(static_cast<unsigned char>(text[i]))) ++i;
        Token t;
        t.text = std::string(text.substr(start, i - start));
        t.norm = normalize(t.text);
        t.begin = start;
        t.end = i;
        t.capitalized = startsUpper(t.text);
        t.punctBefore = pendingPunct;
        pendingPunct = 0;
        tokens.push_back(std::move(t));
    }
    return tokens;
}

std::vector<std::string> normalizedTokens(std::string_view text) {
    std::vector<std::string> out;
    for (auto& t : tokenize(text)) out.push_back(std::move(t.norm));
    return out;
}

bool isStopword(std::string_view norm) {
    static const std::unordered_set<std::string_view> stop = {
        "a",     "about", "after", "all",   "along", "also",  "an",    "and",   "are",   "around",
        "as",    "at",    "be",    "before", "between", "both", "but",  "by",    "can",   "every",
        "for",   "from",  "had",   "has",   "have",  "he",    "her",   "his",   "i",     "in",
        "into",  "is",    "it",    "its",   "just",  "many",  "more",  "most",  "my",    "near",
        "next",  "not",   "of",    "often", "on",    "once",  "one",   "or",    "other", "our",
        "out",   "over",  "s",     "same",  "several", "she", "so",    "some",  "than",  "that",
        "the",   "their", "them",  "then",  "there", "these", "they",  "this",  "those", "through",
        "to",    "too",   "two",   "up",    "very",  "was",   "we",    "were",  "what",  "when",
        "where", "which", "while", "who",   "with",  "would", "you",   "your",  "each",  "other",
    };
    return stop.count(norm) > 0;
}

std::vector<std::string> contextTerms(const std::vector<Token>& tokens, std::size_t skipBegin,
                                      std::size_t skipEnd) {
    std::vector<std::string> terms;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i >= skipBegin && i < skipEnd) continue;
        const std::string& n = tokens[i].norm;
        if (n.size() < 2 || isStopword(n)) continue;
        if (std::all_of(n.begin(), n.end(), [](unsigned char ch) { return std::isdigit(ch); })) continue;
        terms.push_back(n);
    }
    return terms;
}

}  // namespace georesolve
