import Foundation

struct EngineResult: Codable, Equatable {
    var ok: Bool
    var status: String
    var message: String?
    var inputPath: String?
    var outputPath: String?
    var embedPath: String?
    var width: Int?
    var height: Int?
    var embedWidth: Int?
    var embedHeight: Int?
    var lpi: Double?
    var angles: [Double]?
    var screenMean: Double?
    var detected: Bool?
}

@MainActor
final class SessionModel {
    var imageURL: URL?
    var result: EngineResult?
    var isBusy = false
    var errorText: String?

    private let engine = EngineClient()

    func openImage(_ url: URL) {
        imageURL = url
        result = nil
        errorText = nil
        Task { await runDescreen() }
    }

    func runDescreen() async {
        guard let imageURL else { return }
        isBusy = true
        errorText = nil
        defer { isBusy = false }
        do {
            let (res, _) = try await engine.descreen(inputURL: imageURL)
            result = res
        } catch {
            errorText = error.localizedDescription
        }
    }
}
