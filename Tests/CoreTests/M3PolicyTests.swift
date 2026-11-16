import Engine
import Foundation
import KeylimeCore
import XCTest

/// m3 in the app: models_v2 entries in the picker, and the decoder's tiers through Policy.
final class M3PolicyTests: XCTestCase {
    let manifest: [String: Any] = [
        "default": "m2s_int4",
        "models": [["id": "m2s_int4"], ["id": "block4"]],
        "models_v2": [
            ["id": "m3", "type": "m3dict", "label": "m3 62M int4 (Preview)", "replaces": "block4", "min_app": ["android": "0.1.200", "mac": "0.2.17"],
             "base_urls": ["https://huggingface.co/MostLime/keylime-m3-preview/resolve/main/"],
             "files": [["role": "encoder", "file": "m3.encoder.int4.onnx", "size": 1, "sha256": "AA"],
                       ["file": "m3.decoder.int4.onnx", "size": 2, "sha256": "bb"],
                       ["file": "vocab_m3.json", "size": 3, "sha256": "cc"],
                       ["file": "lexicon_clean.tsv", "size": 4, "sha256": "dd"]],
             "params": ["th_auto": -5.9], "config": ["eos": 2]],
            ["id": "m9", "type": "something-new", "files": [["file": "x", "sha256": "ee"]]],
        ],
    ]

    func testModelsV2() {
        let v2 = ModelListV2.parse(manifest, appVersion: "0.2.17")
        XCTAssertEqual(v2.map { $0.id }, ["m3"], "unknown types are skipped")
        let m = v2[0]
        XCTAssertEqual(m.file("encoder")?.sha256, "aa")
        XCTAssertEqual(m.file("decoder")?.file, "m3.decoder.int4.onnx")
        XCTAssertEqual(m.file("vocab")?.file, "vocab_m3.json")
        XCTAssertEqual(m.file("lexicon")?.size, 4)
        XCTAssertEqual(m.baseUrls.count, 1)
        XCTAssertEqual(ModelListV2.order(["m2s_int4", "block4"], v2), ["m2s_int4", "m3"], "m3 takes m1's slot")
        XCTAssertEqual(ModelListV2.order(["m2s_int4"], v2), ["m2s_int4", "m3"])
        XCTAssertTrue(ModelListV2.parse(manifest, appVersion: "0.2.16").isEmpty, "older than min_app.mac")
        XCTAssertEqual(ModelListV2.parse(manifest, appVersion: "0.0.412").count, 1, "CI dev builds run everything")
        XCTAssertTrue(ModelListV2.parse(["models": []], appVersion: "1.0.0").isEmpty)
        // files as a role map
        let d = ModelListV2.parse(["models_v2": [["id": "m3", "type": "m3dict", "files": ["encoder": ["file": "e.onnx", "sha256": "1"],
                                                                                         "lexicon": ["file": "l.tsv", "sha256": "2"]]]]],
                                  appVersion: "9.9.9")
        XCTAssertEqual(d.first?.file("encoder")?.file, "e.onnx")
        XCTAssertEqual(d.first?.file("lexicon")?.file, "l.tsv")
    }

    func outcome(_ tier: M3Tier, typed: String = "teh", out: String = "the", recheck: String? = nil) -> WordOutcome {
        WordOutcome(typed: typed, out: out, margin: 1, gap: 1, fixes: out == typed ? [] : [out, "ten"], recheck: recheck, typedOOV: true, tier: tier)
    }

    func testBoundaryTiers() {
        XCTAssertEqual(Policy.boundary(outcome(.auto), mode: .auto, hasPrev: false, prevOut: nil, userWord: false),
                       .apply(out: "the", prevNew: nil, merged: false))
        XCTAssertEqual(Policy.boundary(outcome(.auto), mode: .suggest, hasPrev: false, prevOut: nil, userWord: false),
                       .flag(fixes: ["the", "ten"], spelling: true))
        XCTAssertEqual(Policy.boundary(outcome(.suggest), mode: .auto, hasPrev: false, prevOut: nil, userWord: false),
                       .flag(fixes: ["the", "ten"], spelling: true))
        XCTAssertEqual(Policy.boundary(outcome(.keep, out: "teh"), mode: .auto, hasPrev: false, prevOut: nil, userWord: false), .keep)
        XCTAssertEqual(Policy.boundary(outcome(.auto), mode: .off, hasPrev: false, prevOut: nil, userWord: false), .keep)
        XCTAssertEqual(Policy.boundary(outcome(.auto), mode: .auto, hasPrev: false, prevOut: nil, userWord: true), .keep)
        // undone before: underlined only
        XCTAssertEqual(Policy.boundary(outcome(.auto), mode: .auto, hasPrev: false, prevOut: nil, userWord: false, penalty: { _, _ in 1 }),
                       .flag(fixes: ["the", "ten"], spelling: true))
        // the previous word's AUTO re-check rides along in auto mode only
        XCTAssertEqual(Policy.boundary(outcome(.keep, typed: "cat", out: "cat", recheck: "the"), mode: .auto, hasPrev: true, prevOut: "teh",
                                       userWord: false), .apply(out: "cat", prevNew: "the", merged: false))
        XCTAssertEqual(Policy.boundary(outcome(.keep, typed: "cat", out: "cat", recheck: "the"), mode: .suggest, hasPrev: true, prevOut: "teh",
                                       userWord: false), .keep)
        // fixed words (i -> I) still come first, except in casual
        XCTAssertEqual(Policy.boundary(outcome(.keep, typed: "i", out: "i"), mode: .suggest, hasPrev: false, prevOut: nil, userWord: false),
                       .apply(out: "I", prevNew: nil, merged: false))
        XCTAssertEqual(Policy.boundary(outcome(.keep, typed: "i", out: "i"), mode: .suggest, hasPrev: false, prevOut: nil, userWord: false,
                                       capI: false), .keep)
    }

    func testIdleTiers() {
        XCTAssertEqual(Policy.idle(gap: -5, spelling: false, mode: .auto, inCurrentSentence: true, userWord: false, tier: .auto), .apply)
        XCTAssertEqual(Policy.idle(gap: -5, spelling: true, mode: .auto, inCurrentSentence: false, userWord: false, tier: .auto), .flag)
        XCTAssertEqual(Policy.idle(gap: -5, spelling: true, mode: .suggest, inCurrentSentence: true, userWord: false, tier: .auto), .flag)
        XCTAssertEqual(Policy.idle(gap: -8, spelling: true, mode: .auto, inCurrentSentence: true, userWord: false, tier: .suggest), .flag)
        XCTAssertEqual(Policy.idle(gap: 9, spelling: true, mode: .auto, inCurrentSentence: true, userWord: false, tier: .keep), .none)
        XCTAssertEqual(Policy.idle(gap: 9, spelling: true, mode: .auto, inCurrentSentence: true, userWord: true, tier: .auto), .none)
    }
}
