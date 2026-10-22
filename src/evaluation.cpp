#include "georesolve/evaluation.hpp"

#include <cstdio>

#include "georesolve/geo.hpp"

namespace georesolve {

Metrics evaluate(const Resolver& resolver, const std::vector<Document>& docs, Method method) {
    const Gazetteer& g = resolver.gazetteer();
    Metrics m;
    m.method = method;
    m.label = methodName(method);
    for (const auto& doc : docs) {
        auto res = resolver.resolve(doc.text, method);
        std::vector<Mention> mentions;
        mentions.reserve(res.size());
        for (const auto& r : res) mentions.push_back(r.mention);
        auto align = alignGold(mentions, doc.gold);
        for (std::size_t i = 0; i < doc.gold.size(); ++i) {
            ++m.gold;
            const int goldIdx = g.indexOf(doc.gold[i].id);
            if (align[i] < 0 || goldIdx < 0) {
                ++m.missed;
                continue;
            }
            const Resolution& r = res[static_cast<std::size_t>(align[i])];
            const bool ambiguous = r.mention.candidates.size() > 1;
            const bool ok = r.entry == goldIdx;
            if (ambiguous) ++m.ambiguous;
            if (ok) {
                ++m.correct;
                if (ambiguous) ++m.ambiguousCorrect;
            }
            const Entry& a = g.at(r.entry);
            const Entry& b = g.at(goldIdx);
            const double km = haversineKm(a.lat, a.lon, b.lat, b.lon);
            m.sumErrorKm += km;
            if (km <= 161.0) ++m.within161;
        }
    }
    return m;
}

void printTable(std::ostream& os, const std::vector<Metrics>& rows) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%-13s %9s %13s %11s %15s %8s\n", "method", "accuracy", "ambiguous acc",
                  "acc@161km", "mean error km", "missed");
    os << buf;
    os << std::string(74, '-') << '\n';
    for (const auto& m : rows) {
        std::snprintf(buf, sizeof buf, "%-13s %8.1f%% %12.1f%% %10.1f%% %15.0f %8zu\n", m.label.c_str(),
                      100.0 * m.accuracy(), 100.0 * m.ambiguousAccuracy(), 100.0 * m.accAt161(), m.meanErrorKm(),
                      m.missed);
        os << buf;
    }
    if (!rows.empty()) {
        std::snprintf(buf, sizeof buf, "(%zu gold mentions, %zu ambiguous)\n", rows[0].gold, rows[0].ambiguous);
        os << buf;
    }
}

void printErrors(std::ostream& os, const Resolver& resolver, const std::vector<Document>& docs, Method method) {
    const Gazetteer& g = resolver.gazetteer();
    for (const auto& doc : docs) {
        auto res = resolver.resolve(doc.text, method);
        std::vector<Mention> mentions;
        for (const auto& r : res) mentions.push_back(r.mention);
        auto align = alignGold(mentions, doc.gold);
        for (std::size_t i = 0; i < doc.gold.size(); ++i) {
            const int goldIdx = g.indexOf(doc.gold[i].id);
            if (align[i] < 0) {
                os << "  MISSED " << doc.gold[i].surface << " in: " << doc.text << '\n';
                continue;
            }
            const Resolution& r = res[static_cast<std::size_t>(align[i])];
            if (r.entry != goldIdx) {
                os << "  " << r.mention.surface << ": predicted " << g.describe(r.entry) << ", gold "
                   << (goldIdx >= 0 ? g.describe(goldIdx) : doc.gold[i].id) << "\n    " << doc.text << '\n';
            }
        }
    }
}

}  // namespace georesolve
