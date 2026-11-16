import Foundation

// The app side of m3: a finished word and the pause pass, as the WordResult / SentenceFix the m2 path produces, with
// the m3 tier attached (Policy maps AUTO to auto-apply, SUGGEST to an underline, anything else to keep).

extension M3Corrector {
    /// the sentence being typed, ending with the word just finished: from the last sentence end, leading words dropped
    /// until it fits the encoder
    func message(_ full: String) -> String {
        let cs = Array(full)
        var s = SentenceCheck.split(full).last.map { String(cs[$0]) } ?? full
        while let f = s.first, f.isWhitespace { s.removeFirst() }
        while s.utf8.count > model.meta.maxSrc - 1 {
            guard let sp = s.firstIndex(where: { $0.isWhitespace }) else { break }
            s = String(s[sp...]).trimmingCharacters(in: .whitespaces)
        }
        return s
    }

    /// A finished word (the m2 Corrector.correctWord's place). prevTyped: the previous word as it is in the field (its
    /// re-check is the m3 verdict for it, now that this word follows); prevText is set only for an AUTO change.
    public func correctWord(_ w: WordInput, prevTyped: String?, style: String, user: [String]) throws -> WordResult {
        let t0 = Date()
        let typed = w.typed
        let keepSlot = Slot(typed: typed, cands: [typed], sp: [0], typedIdx: 0)
        func keep(_ tier: M3Tier = .keep) -> WordResult {
            let r = WordResult(slot: keepSlot, scores: [0], lm: nil, chosen: 0, out: typed, skipped: false, cheap: false, nScored: 0, nCands: 1,
                               ms: Date().timeIntervalSince(t0) * 1000)
            r.m3Tier = tier
            return r
        }
        guard M3Text.segWords(typed) == [typed] else { return keep() }
        let msg = message(w.left + w.sep + typed)
        let res = try correct(msg, style: style, user: user)
        let tw = res.typedWords
        guard let j = tw.indices.last, tw[j] == typed else { return keep() }
        var prevText: String? = nil
        if let pt = prevTyped, j >= 1, tw[j - 1] == pt, let pv = verdict(res, j: j - 1), pv.tier == .auto, pv.best != pt {
            prevText = pv.best
        }
        guard let v = verdict(res, j: j), v.tier != .keep, v.best != typed else {
            let r = keep()
            if prevText != nil {
                let k = WordResult(slot: keepSlot, scores: [0], lm: nil, chosen: 0, out: typed, skipped: false, cheap: false, nScored: 0,
                                   nCands: 1, ms: r.ms, prevText: prevText)
                k.m3Tier = .keep
                return k
            }
            return r
        }
        var fixes = v.fixes
        if let i = fixes.firstIndex(of: v.best) { fixes.remove(at: i) }
        fixes.insert(v.best, at: 0)
        fixes = Array(fixes.prefix(3))
        // scores: the typed word 0, the fixes in order, the best one the m3 margin above the copy
        let scores = [0.0] + fixes.indices.map { v.margin - Double($0) * 0.001 }
        let slot = Slot(typed: typed, cands: [typed] + fixes, sp: [Double](repeating: 0, count: fixes.count + 1), typedIdx: 0)
        let r = WordResult(slot: slot, scores: scores, lm: nil, chosen: 1, out: v.best, skipped: false, cheap: false, nScored: fixes.count + 1,
                           nCands: fixes.count + 1, ms: Date().timeIntervalSince(t0) * 1000, prevText: prevText)
        r.m3Tier = v.tier
        return r
    }

    /// The pause pass for one sentence: every word m3 would change (AUTO) or suggest (SUGGEST), Character offsets in
    /// req.sentence, gap = the m3 margin. AUTO changes the user undid before come back as SUGGEST.
    public func checkSentence(_ req: SentenceReq, style: String, user: UserData?) throws -> [SentenceFix] {
        let cs = Array(req.sentence)
        var lead = 0
        while lead < cs.count && cs[lead].isWhitespace { lead += 1 }
        let msg = String(cs[lead...])
        if msg.utf8.count > model.meta.maxSrc - 1 { return [] }
        let res = try correct(msg, style: style, user: user?.all ?? [])
        // scalar offset -> Character offset in msg
        var charAt: [Int: Int] = [:]
        var u = 0
        for (ci, ch) in msg.enumerated() { charAt[u] = ci; u += ch.unicodeScalars.count }
        let segs = M3Text.seg(M3Text.scalars(msg))
        var out: [SentenceFix] = []
        for j in res.typedWords.indices {
            guard let v = verdict(res, j: j), v.tier != .keep, j < segs.count, let c0 = charAt[segs[j].0] else { continue }
            if v.best == v.typed { continue }
            var tier = v.tier
            if tier == .auto, let ud = user, ud.penalty(v.typed, v.best) > 0 { tier = .suggest }
            var fixes = v.fixes
            if let i = fixes.firstIndex(of: v.best) { fixes.remove(at: i) }
            fixes.insert(v.best, at: 0)
            out.append(SentenceFix(start: lead + c0, from: v.typed, to: v.best, gap: v.margin, fixes: Array(fixes.prefix(3)), tier: tier))
        }
        return out
    }
}
