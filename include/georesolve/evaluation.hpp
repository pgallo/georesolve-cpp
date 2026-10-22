#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "georesolve/corpus.hpp"
#include "georesolve/resolver.hpp"

namespace georesolve {

struct Metrics {
    Method method = Method::Hmm;
    std::string label;  // display name, defaults to methodName(method)
    std::size_t gold = 0;
    std::size_t correct = 0;
    std::size_t ambiguous = 0;         // gold mentions with 2+ candidates
    std::size_t ambiguousCorrect = 0;
    std::size_t missed = 0;            // gold mentions the matcher did not find
    std::size_t within161 = 0;         // prediction within 161 km (100 miles) of the gold point
    double sumErrorKm = 0.0;           // over found mentions

    double accuracy() const { return gold ? static_cast<double>(correct) / static_cast<double>(gold) : 0.0; }
    double ambiguousAccuracy() const {
        return ambiguous ? static_cast<double>(ambiguousCorrect) / static_cast<double>(ambiguous) : 0.0;
    }
    double accAt161() const { return gold ? static_cast<double>(within161) / static_cast<double>(gold) : 0.0; }
    double meanErrorKm() const {
        const std::size_t found = gold - missed;
        return found ? sumErrorKm / static_cast<double>(found) : 0.0;
    }
};

Metrics evaluate(const Resolver& resolver, const std::vector<Document>& docs, Method method);

void printTable(std::ostream& os, const std::vector<Metrics>& rows);

// Lists HMM errors, for error analysis.
void printErrors(std::ostream& os, const Resolver& resolver, const std::vector<Document>& docs, Method method);

}  // namespace georesolve
