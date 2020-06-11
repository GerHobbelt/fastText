// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FastText",
    products: [
        .library(
            name: "FastText",
            targets: ["FastText"]
        ),
        .library(
            name: "CFastText",
            targets: ["CFastText"]
        ),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "FastText",
            dependencies: ["CFastText"],
            path: "swift/Sources/SFastText"
        ),
        .target(
            name: "CFastText",
            path: "swift/Sources/CFastText"
        ),
    ]
)
