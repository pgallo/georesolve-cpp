// georesolve command-line interface.
//
//   georesolve "I flew from Naples to Rome, then drove to Florence."
//   georesolve --method knn "Paris is lovely in spring."
//   georesolve --eval [--errors]
//
// Data files are read from --data DIR, else $GEORESOLVE_DATA, else the source tree's data/.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "georesolve/corpus.hpp"
#include "georesolve/evaluation.hpp"
#include "georesolve/resolver.hpp"

#ifndef GEORESOLVE_DATA_DIR
#define GEORESOLVE_DATA_DIR "data"
#endif

namespace {

using namespace georesolve;

void usage() {
    std::cerr << "usage: georesolve [--data DIR] [--method population|knn|independent|hmm] [--top N] TEXT...\n"
                 "       georesolve [--data DIR] --eval [--errors]\n";
}

bool parseMethod(const std::string& s, Method& out) {
    for (Method m : {Method::Population, Method::Knn, Method::Independent, Method::Hmm}) {
        if (s == methodName(m)) {
            out = m;
            return true;
        }
    }
    return false;
}

void printResolutions(const Resolver& r, const std::vector<Resolution>& res, int top) {
    const Gazetteer& g = r.gazetteer();
    if (res.empty()) {
        std::cout << "(no place names found)\n";
        return;
    }
    char buf[256];
    for (const auto& x : res) {
        const Entry& e = g.at(x.entry);
        std::snprintf(buf, sizeof buf, "%-22s -> %-28s %-24s lat %7.2f lon %8.2f  conf %.2f  (%zu candidate%s)\n",
                      x.mention.surface.c_str(), g.describe(x.entry).c_str(), e.id.c_str(), e.lat, e.lon,
                      x.confidence, x.mention.candidates.size(), x.mention.candidates.size() == 1 ? "" : "s");
        std::cout << buf;
        for (int i = 1; i < top && i < static_cast<int>(x.ranked.size()); ++i) {
            const auto& [idx, p] = x.ranked[static_cast<std::size_t>(i)];
            std::snprintf(buf, sizeof buf, "%-22s    %-28s %-24s %.2f\n", "", g.describe(idx).c_str(),
                          g.at(idx).id.c_str(), p);
            std::cout << buf;
        }
    }
}

int runEval(Resolver& resolver, const std::string& dataDir, bool showErrors) {
    auto test = loadJsonl(dataDir + "/test.jsonl");
    std::vector<Metrics> rows;
    for (Method m : {Method::Population, Method::Knn, Method::Independent, Method::Hmm}) {
        rows.push_back(evaluate(resolver, test, m));
    }
    std::cout << "\nHeld-out evaluation on " << test.size() << " sentences (data/test.jsonl)\n\n";
    printTable(std::cout, rows);
    // Ablation: identical model, transitions normalized over each mention's own
    // candidates instead of over the whole gazetteer.
    const TransitionParams tuned = resolver.transitionParams();
    const double weight = resolver.transitionWeight();
    TransitionParams local = tuned;
    local.normalization = Normalization::Local;
    resolver.setTransition(local, weight);
    Metrics ablation = evaluate(resolver, test, Method::Hmm);
    ablation.label = "hmm-local";
    resolver.setTransition(tuned, weight);
    std::cout << "\nAblation (transition rows normalized over the next mention's candidates only):\n";
    printTable(std::cout, {rows.back(), ablation});

    if (showErrors) {
        std::cout << "\nHMM errors:\n";
        printErrors(std::cout, resolver, test, Method::Hmm);
    }
    // Regression gate for CI: the full model must beat every baseline.
    const double hmm = rows.back().accuracy();
    for (std::size_t i = 0; i + 1 < rows.size(); ++i) {
        if (rows[i].method != Method::Independent && !(hmm > rows[i].accuracy())) {
            std::cerr << "\nFAIL: hmm does not beat the " << methodName(rows[i].method) << " baseline\n";
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dataDir = GEORESOLVE_DATA_DIR;
    if (const char* env = std::getenv("GEORESOLVE_DATA")) dataDir = env;
    Method method = Method::Hmm;
    bool eval = false, showErrors = false;
    int top = 1;
    std::string text;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto needValue = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << flag << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--data") {
            dataDir = needValue("--data");
        } else if (a == "--method") {
            if (!parseMethod(needValue("--method"), method)) {
                usage();
                return 2;
            }
        } else if (a == "--top") {
            top = std::atoi(needValue("--top").c_str());
        } else if (a == "--eval") {
            eval = true;
        } else if (a == "--errors") {
            showErrors = true;
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            if (!text.empty()) text += ' ';
            text += a;
        }
    }
    if (!eval && text.empty()) {
        usage();
        return 2;
    }

    try {
        Resolver resolver(Gazetteer::loadCsv(dataDir + "/gazetteer.csv"));
        auto train = loadJsonl(dataDir + "/train.jsonl");
        TrainingReport rep = resolver.train(train);

        if (eval) {
            std::cout << "Gazetteer: " << resolver.gazetteer().size() << " entries\n"
                      << "Training:  " << rep.documents << " sentences, " << rep.alignedMentions << "/"
                      << rep.goldMentions << " gold mentions matched, " << rep.ambiguousInstances
                      << " ambiguous instances for the emission model\n"
                      << "Emission weights (conditional logit):\n";
            const auto& w = resolver.emissionModel().weights();
            for (std::size_t k = 0; k < kNumFeatures; ++k) {
                std::printf("  %-20s %+7.3f\n", featureNames()[k], w[k]);
            }
            std::printf("Transitions: weight %.2f, distance scale %.0f km (train accuracy %.1f%%)\n",
                        rep.transitionWeight, rep.transitionScaleKm, 100.0 * rep.tuningAccuracy);
            return runEval(resolver, dataDir, showErrors);
        }
        printResolutions(resolver, resolver.resolve(text, method), top);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
