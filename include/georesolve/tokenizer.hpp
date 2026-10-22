#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace georesolve {

struct Token {
    std::string text;     // surface form as it appears in the input
    std::string norm;     // lower-cased, Latin-1 diacritics folded to ASCII
    std::size_t begin = 0;  // byte offsets into the input
    std::size_t end = 0;
    bool capitalized = false;
    char punctBefore = 0;  // first punctuation mark between the previous word and this one
};

// Splits text into word tokens. Punctuation is not emitted as tokens; instead the
// first punctuation mark preceding a word is recorded on it, so multi-word names
// such as "St. Louis" still match while "Paris, Texas" remains detectable.
std::vector<Token> tokenize(std::string_view text);

// Lower-case + fold accented Latin-1 letters ("Córdoba" -> "cordoba").
std::string normalize(std::string_view word);

// Normalized tokens of a name, using the same rules as tokenize().
std::vector<std::string> normalizedTokens(std::string_view text);

bool isStopword(std::string_view norm);

// Context terms for a mention: normalized non-stopword tokens outside [skipBegin, skipEnd).
std::vector<std::string> contextTerms(const std::vector<Token>& tokens, std::size_t skipBegin,
                                      std::size_t skipEnd);

}  // namespace georesolve
