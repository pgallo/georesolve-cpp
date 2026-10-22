#include "georesolve/resolver.hpp"

#include <algorithm>
#include <cmath>

#include "georesolve/hmm.hpp"

namespace georesolve {

const char* methodName(Method m) {
    switch (m) {
        case Method::Population: return "population";
        case Method::Knn: return "knn";
        case Method::Independent: return "independent";
        case Method::Hmm: return "hmm";
    }
    return "?";
}

namespace {

std::string normSurface(const std::string& s) {
    std::string out;
    for (const auto& t : normalizedTokens(s)) {
        if (!out.empty()) out += ' ';
        out += t;
    }
    return out;
}

std::vector<double> softmax(const std::vector<double>& logits) {
    const double z = logSumExp(logits);
    std::vector<double> p;
    p.reserve(logits.size());
    for (double l : logits) p.push_back(std::exp(l - z));
    return p;
}

}  // namespace

std::vector<int> alignGold(const std::vector<Mention>& mentions, const std::vector<GoldMention>& gold) {
    std::vector<int> out(gold.size(), -1);
    std::size_t cursor = 0;
    for (std::size_t g = 0; g < gold.size(); ++g) {
        const std::string want = normSurface(gold[g].surface);
        for (std::size_t m = cursor; m < mentions.size(); ++m) {
            if (normSurface(mentions[m].surface) == want) {
                out[g] = static_cast<int>(m);
                cursor = m + 1;
                break;
            }
        }
    }
    return out;
}

Resolver::Resolver(Gazetteer gazetteer, const TransitionParams& params)
    : gazetteer_(std::move(gazetteer)), matcher_(gazetteer_), transitions_(gazetteer_, params) {}

Resolver::Prepared Resolver::prepare(std::string_view text, int excludeDoc) const {
    Prepared p;
    p.tokens = tokenize(text);
    p.mentions = matcher_.find(text, p.tokens);
    for (const auto& m : p.mentions) {
        auto terms = contextTerms(p.tokens, m.tokenBegin, m.tokenEnd);
        p.knn.push_back(knn_.predict(knn_.vectorize(terms), m.candidates, excludeDoc));
    }
    p.features = extractFeatures(gazetteer_, p.tokens, p.mentions, p.knn);
    return p;
}

TrainingReport Resolver::train(const std::vector<Document>& docs, bool tuneTransitions) {
    TrainingReport report;
    report.documents = docs.size();

    // 1. KNN index over every aligned gold mention.
    std::vector<LabelledContext> examples;
    for (std::size_t d = 0; d < docs.size(); ++d) {
        auto tokens = tokenize(docs[d].text);
        auto mentions = matcher_.find(docs[d].text, tokens);
        auto align = alignGold(mentions, docs[d].gold);
        report.goldMentions += docs[d].gold.size();
        for (std::size_t g = 0; g < align.size(); ++g) {
            if (align[g] < 0) continue;
            const Mention& m = mentions[static_cast<std::size_t>(align[g])];
            int label = gazetteer_.indexOf(docs[d].gold[g].id);
            if (label < 0) continue;
            ++report.alignedMentions;
            examples.push_back({static_cast<int>(d), label, contextTerms(tokens, m.tokenBegin, m.tokenEnd)});
        }
    }
    knn_.fit(std::move(examples));

    // 2. Emission model on ambiguous mentions, KNN scored leave-one-document-out.
    std::vector<Prepared> prepared;
    std::vector<EmissionModel::Instance> instances;
    for (std::size_t d = 0; d < docs.size(); ++d) {
        prepared.push_back(prepare(docs[d].text, static_cast<int>(d)));
        const Prepared& p = prepared.back();
        auto align = alignGold(p.mentions, docs[d].gold);
        for (std::size_t g = 0; g < align.size(); ++g) {
            if (align[g] < 0) continue;
            auto mi = static_cast<std::size_t>(align[g]);
            const auto& cands = p.mentions[mi].candidates;
            if (cands.size() < 2) continue;
            auto it = std::find(cands.begin(), cands.end(), gazetteer_.indexOf(docs[d].gold[g].id));
            if (it == cands.end()) continue;
            instances.push_back({p.features[mi], static_cast<std::size_t>(it - cands.begin())});
        }
    }
    report.ambiguousInstances = instances.size();
    emission_ = EmissionModel();
    report.emissionLogLik = emission_.fit(instances);

    // 3. Transition weight and distance scale, chosen on training accuracy.
    if (tuneTransitions) {
        double bestAcc = -1.0;
        TransitionParams bestP = transitions_.params();
        double bestBeta = beta_;
        for (double beta : {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}) {
            for (double scale : {50.0, 100.0, 250.0, 500.0}) {
                TransitionParams tp = transitions_.params();
                tp.scaleKm = scale;
                setTransition(tp, beta);
                double acc = trainingAccuracy(prepared, docs);
                if (acc > bestAcc + 1e-12) {
                    bestAcc = acc;
                    bestP = tp;
                    bestBeta = beta;
                }
            }
        }
        setTransition(bestP, bestBeta);
        report.tuningAccuracy = bestAcc;
    } else {
        report.tuningAccuracy = trainingAccuracy(prepared, docs);
    }
    report.transitionWeight = beta_;
    report.transitionScaleKm = transitions_.params().scaleKm;
    return report;
}

double Resolver::trainingAccuracy(const std::vector<Prepared>& prepared, const std::vector<Document>& docs) const {
    std::size_t correct = 0, total = 0;
    for (std::size_t d = 0; d < docs.size(); ++d) {
        auto res = decode(prepared[d], Method::Hmm);
        auto align = alignGold(prepared[d].mentions, docs[d].gold);
        for (std::size_t g = 0; g < align.size(); ++g) {
            ++total;
            if (align[g] >= 0 &&
                res[static_cast<std::size_t>(align[g])].entry == gazetteer_.indexOf(docs[d].gold[g].id)) {
                ++correct;
            }
        }
    }
    return total == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
}

std::vector<Resolution> Resolver::resolve(std::string_view text, Method method) const {
    return decode(prepare(text, -1), method);
}

std::vector<Resolution> Resolver::decode(const Prepared& p, Method method) const {
    const std::size_t T = p.mentions.size();
    std::vector<std::vector<double>> probs(T);  // per-mention distribution used for ranking/confidence
    std::vector<std::size_t> choice(T, 0);

    if (method == Method::Hmm && T > 0) {
        LogHmm hmm;
        for (std::size_t t = 0; t < T; ++t) hmm.logEmit.push_back(emission_.logProbs(p.features[t]));
        for (std::size_t t = 1; t < T; ++t) {
            Matrix a = transitions_.logMatrix(gazetteer_, p.mentions[t - 1].candidates, p.mentions[t].candidates);
            for (auto& row : a) {
                for (double& v : row) v *= beta_;
            }
            hmm.logTrans.push_back(std::move(a));
        }
        auto path = viterbi(hmm).path;
        auto gamma = posteriors(hmm);
        for (std::size_t t = 0; t < T; ++t) {
            choice[t] = path[t];
            probs[t] = std::move(gamma[t]);
        }
    } else {
        for (std::size_t t = 0; t < T; ++t) {
            const auto& cands = p.mentions[t].candidates;
            std::vector<double> score(cands.size());
            switch (method) {
                case Method::Population: {
                    // Score by log population; report the population share as confidence.
                    double total = 0.0;
                    for (std::size_t c = 0; c < cands.size(); ++c) {
                        score[c] = p.features[t][c][kLogPopulation];
                        probs[t].push_back(static_cast<double>(gazetteer_.at(cands[c]).population) + 1.0);
                        total += probs[t].back();
                    }
                    for (double& v : probs[t]) v /= total;
                    break;
                }
                case Method::Knn:
                    // Population only breaks ties (KNN is uniform when no training context is similar).
                    for (std::size_t c = 0; c < cands.size(); ++c) {
                        score[c] = p.knn[t][c] + 1e-6 * p.features[t][c][kLogPopulation];
                    }
                    probs[t] = p.knn[t];
                    break;
                default:  // Method::Independent
                    score = emission_.logProbs(p.features[t]);
                    probs[t] = softmax(score);
                    break;
            }
            choice[t] = static_cast<std::size_t>(std::max_element(score.begin(), score.end()) - score.begin());
        }
    }

    std::vector<Resolution> out;
    out.reserve(T);
    for (std::size_t t = 0; t < T; ++t) {
        Resolution r;
        r.mention = p.mentions[t];
        r.entry = r.mention.candidates[choice[t]];
        r.confidence = probs[t][choice[t]];
        for (std::size_t c = 0; c < r.mention.candidates.size(); ++c) {
            r.ranked.emplace_back(r.mention.candidates[c], probs[t][c]);
        }
        std::stable_sort(r.ranked.begin(), r.ranked.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        out.push_back(std::move(r));
    }
    return out;
}

}  // namespace georesolve
